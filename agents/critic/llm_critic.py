"""
LLM-Based Semantic Critic (MedGemma)

Second-stage critic that uses MedGemma to detect:
  - Clinical overconfidence relative to epistemic uncertainty
  - Missing reasonable differential diagnoses
  - Unjustified certainty in the impression
  - Logical gaps between findings and impression

This module does NOT:
  - Generate diagnoses
  - Output probability estimates
  - Override KLE uncertainty
  - Replace the radiologist

It only critiques reasoning.
"""

import json
import logging
from typing import Optional

import torch
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

# Pydantic output model
class LLMCriticOutput(BaseModel):
    """Structured output from the MedGemma semantic critic."""
    overconfidence_reason: Optional[str] = Field(
        default=None,
        description="Reason the report is overconfident, or null if appropriate."
    )
    missing_differentials: list[str] = Field(
        default_factory=list,
        description="Reasonable differential diagnoses not mentioned."
    )
    justification_gap: Optional[str] = Field(
        default=None,
        description="Logical gap between findings and impression, or null."
    )
    suggested_hedging: Optional[str] = Field(
        default=None,
        description="Safer phrasing suggestion, or null."
    )
    semantic_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Semantic risk score from 0 (safe) to 1 (high risk)."
    )


# System prompt  (kept as a constant for auditability)


_SYSTEM_PROMPT = """\
You are a clinical safety critic reviewing a radiology report.

You are NOT diagnosing.

Your job is to detect:
1. Overconfidence relative to uncertainty
2. Missing reasonable differential diagnoses
3. Unjustified certainty in the impression
4. Logical gaps between findings and impression

Given:
- FINDINGS text
- IMPRESSION text
- Epistemic uncertainty score (0–1)

Return STRICT JSON with:

{
  "overconfidence_reason": string | null,
  "missing_differentials": list[string],
  "justification_gap": string | null,
  "suggested_hedging": string | null,
  "semantic_risk_score": float (0–1)
}

Return valid JSON only. No markdown fences, no commentary.
"""



# MedGemmaCritic class


class MedGemmaCritic:
    """
    Lazy-loaded MedGemma wrapper that performs semantic critique.

    The model is loaded on the first call to ``critique()`` and reused for
    subsequent invocations.  If model loading fails or inference errors out
    the caller receives ``None`` so the pipeline falls back to rule-based
    evaluation only.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False

    # ---- lazy model loading ------------------------------------------------

    def _load_model(self):
        """Load MedGemma model and tokenizer once."""
        if self._loaded:
            return

        if settings.MOCK_MODELS:
            logger.info("[LLM-Critic] MOCK mode — skipping model load")
            self._loaded = True
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = settings.MEDGEMMA_4B_MODEL
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info("[LLM-Critic] Loading %s on %s …", model_name, device)

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
            self._loaded = True
            logger.info("[LLM-Critic] Model loaded successfully")

        except Exception as exc:  # noqa: BLE001
            logger.warning("[LLM-Critic] Failed to load model: %s — falling back to rule-based only", exc)
            self._loaded = True  # prevent repeated load attempts

    # ---- prompt construction -----------------------------------------------

    @staticmethod
    def _build_user_prompt(findings: str, impression: str, uncertainty: float) -> str:
        return (
            f"FINDINGS:\n{findings}\n\n"
            f"IMPRESSION:\n{impression}\n\n"
            f"Epistemic uncertainty score: {uncertainty:.4f}"
        )

    # ---- inference ---------------------------------------------------------

    def _run_inference(self, user_prompt: str) -> str:
        """Generate text from the model and return raw string."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Use chat template if the tokenizer supports it
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_ids = self._tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self._model.device)
        else:
            flat = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"
            input_ids = self._tokenizer(flat, return_tensors="pt").input_ids.to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
            )

        # Decode only newly generated tokens
        generated = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ---- mock inference ----------------------------------------------------

    @staticmethod
    def _mock_inference(findings: str, impression: str, uncertainty: float) -> LLMCriticOutput:
        """Return a deterministic mock for testing without GPU."""
        # Simulate: if uncertainty is high and impression sounds certain, flag it
        is_assertive = any(
            kw in impression.lower()
            for kw in ("definite", "certain", "diagnostic of", "pathognomonic", "confirms")
        )
        if uncertainty > 0.5 and is_assertive:
            return LLMCriticOutput(
                overconfidence_reason="Report uses definitive language despite high epistemic uncertainty.",
                missing_differentials=["Atypical infection", "Malignancy"],
                justification_gap="Findings do not fully support the level of certainty expressed.",
                suggested_hedging="Consider 'findings are suggestive of' rather than definitive phrasing.",
                semantic_risk_score=0.65,
            )
        return LLMCriticOutput(
            overconfidence_reason=None,
            missing_differentials=[],
            justification_gap=None,
            suggested_hedging=None,
            semantic_risk_score=0.1,
        )

    # ---- public API --------------------------------------------------------

    def critique(
        self,
        findings: str,
        impression: str,
        kle_uncertainty: float,
    ) -> Optional[LLMCriticOutput]:
        """
        Run semantic critique on a radiology report.

        Returns ``LLMCriticOutput`` on success, or ``None`` if inference
        fails (timeout, bad JSON, exception).  The caller should treat
        ``None`` as "LLM unavailable — use rule-based output only."
        """
        self._load_model()

        # ------ mock path ---------------------------------------------------
        if settings.MOCK_MODELS:
            result = self._mock_inference(findings, impression, kle_uncertainty)
            logger.info(
                "CRITIC-LLM (mock): Semantic risk=%.2f, Missing differentials=%d",
                result.semantic_risk_score,
                len(result.missing_differentials),
            )
            return result

        # ------ real inference path -----------------------------------------
        if self._model is None:
            logger.warning("[LLM-Critic] Model unavailable — skipping semantic critique")
            return None

        try:
            user_prompt = self._build_user_prompt(findings, impression, kle_uncertainty)
            raw_output = self._run_inference(user_prompt)

            # Strip potential markdown fences
            cleaned = raw_output.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            result = LLMCriticOutput(**parsed)

            # Clamp semantic_risk_score to [0, 1]
            result.semantic_risk_score = max(0.0, min(1.0, result.semantic_risk_score))

            logger.info(
                "CRITIC-LLM: Semantic risk=%.2f, Missing differentials=%d",
                result.semantic_risk_score,
                len(result.missing_differentials),
            )
            return result

        except json.JSONDecodeError as exc:
            logger.warning("[LLM-Critic] Invalid JSON from model: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("[LLM-Critic] Inference failed: %s", exc)
            return None


medgemma_critic = MedGemmaCritic()
