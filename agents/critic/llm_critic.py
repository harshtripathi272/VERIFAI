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
from app.shared_model_loader import load_shared_medgemma, get_inference_lock

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
5. Inconsistencies with clinical history (if available)
6. Failure to consider literature evidence (if available)

Given:
- FINDINGS text
- IMPRESSION text
- Epistemic uncertainty score (0–1)
- Clinical history from FHIR (optional)
- Literature evidence (optional)

Consider the broader diagnostic context when evaluating.

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
    
    Uses shared model loader to prevent duplicate model loading.
    The model is loaded on the first call to ``critique()`` and reused for
    subsequent invocations.  If model loading fails or inference errors out
    the caller receives ``None`` so the pipeline falls back to rule-based
    evaluation only.
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._loaded = False

    # ---- lazy model loading ------------------------------------------------

    def _load_model(self):
        """Load shared MedGemma model once (singleton across agents)."""
        if self._loaded:
            return

        if settings.MOCK_MODELS:
            logger.info("[LLM-Critic] MOCK mode — skipping model load")
            self._loaded = True
            return

        try:
            logger.info("[LLM-Critic] Loading shared MedGemma model...")
            
            # Use shared model loader instead of loading separate instance
            self._model, self._processor = load_shared_medgemma()
            
            self._loaded = True
            logger.info("[LLM-Critic] Using shared model instance")

        except Exception as exc:  # noqa: BLE001
            logger.warning("[LLM-Critic] Failed to load model: %s — falling back to rule-based only", exc)
            self._loaded = True  # prevent repeated load attempts

    # ---- prompt construction -----------------------------------------------

    @staticmethod
    def _build_user_prompt(
        findings: str, 
        impression: str, 
        uncertainty: float,
        historian_output=None,
        literature_output=None
    ) -> str:
        """Build user prompt with enriched context."""
        prompt_parts = [
            f"FINDINGS:\n{findings}\n",
            f"IMPRESSION:\n{impression}\n",
            f"Epistemic uncertainty score: {uncertainty:.4f}"
        ]
        
        # Add clinical history context if available
        if historian_output:
            supporting = historian_output.supporting_facts if hasattr(historian_output, 'supporting_facts') else []
            contradicting = historian_output.contradicting_facts if hasattr(historian_output, 'contradicting_facts') else []
            
            if supporting or contradicting:
                prompt_parts.append("\nCLINICAL HISTORY (FHIR):")
                if supporting:
                    prompt_parts.append(f"Supporting facts: {len(supporting)} evidence points")
                    for fact in supporting[:3]:  # Top 3
                        prompt_parts.append(f"  - {fact.description}")
                if contradicting:
                    prompt_parts.append(f"Contradicting facts: {len(contradicting)} evidence points")
                    for fact in contradicting[:3]:  # Top 3
                        prompt_parts.append(f"  - {fact.description}")
        
        # Add literature context if available
        if literature_output:
            if isinstance(literature_output, str):
                # String summary
                if len(literature_output) > 50:  # Has meaningful content
                    prompt_parts.append(f"\nLITERATURE EVIDENCE:\n{literature_output[:500]}")
            else:
                # Structured output
                citations = literature_output.citations if hasattr(literature_output, 'citations') else []
                if citations:
                    prompt_parts.append(f"\nLITERATURE EVIDENCE: {len(citations)} relevant studies found")
        
        return "\n".join(prompt_parts)

    # ---- inference ---------------------------------------------------------

    def _run_inference(self, user_prompt: str) -> str:
        """Generate text from the model and return raw string."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded")

        # Use chat template format for MedGemma 1.5
        messages = [{"role": "user", "content": [{"type": "text", "text": f"{_SYSTEM_PROMPT}\n\n{user_prompt}"}]}]
        
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._model.device, dtype=torch.float16)
        
        input_len = inputs["input_ids"].shape[-1]

        # CRITICAL: Acquire shared lock before inference
        _inference_lock = get_inference_lock()
        with _inference_lock:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                )

        # Extract only newly generated tokens
        generated_tokens = outputs[0][input_len:]
        return self._processor.decode(generated_tokens, skip_special_tokens=True).strip()

    # ---- mock inference ----------------------------------------------------

    @staticmethod
    def _mock_inference(
        findings: str, 
        impression: str, 
        uncertainty: float,
        historian_output=None,
        literature_output=None
    ) -> LLMCriticOutput:
        """Return a deterministic mock for testing without GPU."""
        # Simulate: if uncertainty is high and impression sounds certain, flag it
        is_assertive = any(
            kw in impression.lower()
            for kw in ("definite", "certain", "diagnostic of", "pathognomonic", "confirms")
        )
        
        # Check for context mismatches
        context_issues = []
        if historian_output:
            contradicting = historian_output.contradicting_facts if hasattr(historian_output, 'contradicting_facts') else []
            if len(contradicting) > 1:
                context_issues.append("Clinical history contradictions not addressed")
        
        if uncertainty > 0.5 and is_assertive:
            return LLMCriticOutput(
                overconfidence_reason="Report uses definitive language despite high epistemic uncertainty.",
                missing_differentials=["Atypical infection", "Malignancy"],
                justification_gap="Findings do not fully support the level of certainty expressed.",
                suggested_hedging="Consider 'findings are suggestive of' rather than definitive phrasing.",
                semantic_risk_score=0.65 + (0.1 if context_issues else 0.0),
            )
        return LLMCriticOutput(
            overconfidence_reason=None,
            missing_differentials=[],
            justification_gap="\n".join(context_issues) if context_issues else None,
            suggested_hedging=None,
            semantic_risk_score=0.1,
        )

    # ---- public API --------------------------------------------------------

    def critique(
        self,
        findings: str,
        impression: str,
        kle_uncertainty: float,
        historian_output=None,  # NEW: Clinical history context
        literature_output=None   # NEW: Literature evidence
    ) -> Optional[LLMCriticOutput]:
        """
        Run semantic critique on a radiology report with enriched context.

        Args:
            findings: FINDINGS section text
            impression: IMPRESSION section text
            kle_uncertainty: Epistemic uncertainty from KLE (0-1)
            historian_output: HistorianOutput with FHIR facts (optional)
            literature_output: LiteratureOutput or string summary (optional)

        Returns ``LLMCriticOutput`` on success, or ``None`` if inference
        fails (timeout, bad JSON, exception).  The caller should treat
        ``None`` as "LLM unavailable — use rule-based output only."
        """
        self._load_model()

        # ------ mock path ---------------------------------------------------
        if settings.MOCK_MODELS:
            result = self._mock_inference(
                findings, impression, kle_uncertainty,
                historian_output, literature_output
            )
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
            user_prompt = self._build_user_prompt(
                findings, impression, kle_uncertainty,
                historian_output, literature_output
            )
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
