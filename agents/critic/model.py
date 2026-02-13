"""
Critic Model

Overconfidence detector that evaluates consistency between linguistic certainty
in the radiology report and the externally computed KLE-based uncertainty score.

Integration points (marked with # >>> LLM-CRITIC):
  - Second-stage MedGemma semantic critic gated behind settings.ENABLE_LLM_CRITIC
  - Merges LLM output into rule-based results without overriding them
"""

import logging
import re

from app.config import settings
from .llm_critic import medgemma_critic  # >>> LLM-CRITIC

logger = logging.getLogger(__name__)


class CriticModel:
    """
    Linguistic certainty evaluator that detects radiologist overconfidence.
    
    Compares the assertiveness/certainty expressed in the IMPRESSION text
    against the epistemic uncertainty score (KLE) computed from semantic sampling.
    
    Does NOT use internal model signals (logits, entropy, attention).
    """
    
    def __init__(self):
        self.mock = settings.MOCK_MODELS
        # No model loading needed for rule-based linguistic analysis
    
    def _analyze_linguistic_certainty(self, impression: str) -> tuple[float, list[str]]:
        """
        Analyze the linguistic certainty level in the IMPRESSION text.
        
        Returns:
            Tuple of (certainty_score, certainty_markers)
            - certainty_score: 0.0 (very hedged) to 1.0 (very assertive)
            - certainty_markers: List of phrases that influenced the score
        """
        impression_lower = impression.lower()
        
        # High certainty phrases (increase score)
        high_certainty_patterns = [
            r'\bdefinite\b', r'\bdefinitely\b', r'\bcertain\b', r'\bcertainly\b',
            r'\bdiagnostic of\b', r'\bpathognomonic\b', r'\bconfirm\b',
            r'\bconsistent with\b(?! differential)', r'\bdemonstrates\b',
            r'\bshows\b(?! possible)', r'\bevidence of\b(?! possible)',
        ]
        
        # Low certainty phrases (decrease score)
        low_certainty_patterns = [
            r'\bpossib\w*\b', r'\blikely\b', r'\bunlikely\b', r'\bmay\b',
            r'\bcould\b', r'\bmight\b', r'\bsuggest\w*\b', r'\braise.{0,20}concern\b',
            r'\bcannot exclude\b', r'\bconsider\b', r'\bdifferential\b',
            r'\bvs\.?\b', r'\bversus\b', r'\bunclear\b', r'\bindeterminate\b',
            r'\brecommend.{0,30}correlation\b', r'\brecommend.{0,30}follow.?up\b'
        ]
        
        high_count = sum(len(re.findall(pattern, impression_lower)) for pattern in high_certainty_patterns)
        low_count = sum(len(re.findall(pattern, impression_lower)) for pattern in low_certainty_patterns)
        
        # Base score
        base_score = 0.5
        
        # Adjust based on markers
        certainty_score = base_score + (high_count * 0.15) - (low_count * 0.15)
        certainty_score = max(0.0, min(1.0, certainty_score))
        
        # Collect markers for explanation
        markers = []
        if high_count > 0:
            markers.append(f"Strong assertions ({high_count} occurrences)")
        if low_count > 0:
            markers.append(f"Hedging language ({low_count} occurrences)")
        
        return certainty_score, markers
    
    def evaluate(
        self,
        findings: str,
        impression: str,
        kle_uncertainty: float,
        historian_output=None,  # NEW: FHIR clinical context
        literature_output=None   # NEW: Literature evidence
    ) -> tuple[bool, list[str], str | None, float]:
        """
        Evaluate whether the linguistic certainty is appropriate given the epistemic uncertainty
        AND enriched clinical context.
        
        Args:
            findings: FINDINGS text section
            impression: IMPRESSION text section
            kle_uncertainty: Epistemic uncertainty score from KLE (0.0-1.0, higher = more uncertain)
            historian_output: HistorianOutput with FHIR facts (optional)
            literature_output: LiteratureOutput with citations (optional) or string summary
            
        Returns:
            Tuple of (is_overconfident, concern_flags, recommended_hedging, safety_score)
        """
        # ----------------------------------------------------------------
        # Stage 1: Rule-based linguistic analysis  (unchanged)
        # ----------------------------------------------------------------

        # Analyze linguistic certainty in the impression
        linguistic_certainty, markers = self._analyze_linguistic_certainty(impression)
        
        # Detect overconfidence pattern:
        # High linguistic certainty BUT high epistemic uncertainty
        is_overconfident = False
        concern_flags = []
        recommended_hedging = None
        
        # Define thresholds
        HIGH_CERTAINTY_THRESHOLD = 0.65
        HIGH_UNCERTAINTY_THRESHOLD = 0.45
        
        if linguistic_certainty > HIGH_CERTAINTY_THRESHOLD and kle_uncertainty > HIGH_UNCERTAINTY_THRESHOLD:
            is_overconfident = True
            concern_flags.append(
                f"High linguistic certainty ({linguistic_certainty:.2f}) despite high epistemic uncertainty ({kle_uncertainty:.2f})"
            )
            concern_flags.extend(markers)
            recommended_hedging = (
                "Consider using more hedging language (e.g., 'suggestive of', 'most consistent with', "
                "'differential includes') to reflect the semantic instability observed across samples."
            )
        elif linguistic_certainty > 0.8 and kle_uncertainty > 0.3:
            # Moderate concern
            concern_flags.append(
                f"Relatively assertive language ({linguistic_certainty:.2f}) with moderate uncertainty ({kle_uncertainty:.2f})"
            )
            concern_flags.extend(markers)
        
        # Check for internal contradictions between findings and impression
        if "no abnormality" in findings.lower() and "consolidation" in impression.lower():
            concern_flags.append("Potential contradiction: findings vs impression")
        
        # Calculate safety score (inverse of risk)
        # Safety is high when:
        # - Low certainty + high uncertainty (appropriately cautious)
        # - High certainty + low uncertainty (appropriately confident)
        certainty_uncertainty_gap = abs(linguistic_certainty - (1.0 - kle_uncertainty))
        safety_score = 1.0 - certainty_uncertainty_gap
        safety_score = max(0.0, min(1.0, safety_score))
        
        if is_overconfident:
            safety_score = min(safety_score, 0.5)  # Cap safety if overconfident

        # ----------------------------------------------------------------
        # NEW: Stage 1.5: Context-enriched evaluation
        # ----------------------------------------------------------------
        # Evaluate consistency with clinical history and literature
        context_penalty = 0.0
        
        # Check FHIR clinical history
        if historian_output:
            contradicting = historian_output.contradicting_facts if hasattr(historian_output, 'contradicting_facts') else []
            supporting = historian_output.supporting_facts if hasattr(historian_output, 'supporting_facts') else []
            
            # Flag if many contradictions not addressed in impression
            if len(contradicting) > 2:
                concern_flags.append(
                    f"Clinical history contains {len(contradicting)} contradicting facts not addressed in impression"
                )
                context_penalty += 0.10
            
            # Flag if strong clinical support exists but impression is overly cautious
            if len(supporting) > 3 and linguistic_certainty < 0.4:
                concern_flags.append(
                    "Strong clinical support exists but impression remains overly cautious"
                )
                # This is actually GOOD (appropriate caution), so reduce penalty slightly
                context_penalty -= 0.05
        
        # Check literature evidence
        if literature_output:
            # Handle both structured and string outputs
            if isinstance(literature_output, str):
                # String summary - check if it mentions differentials
                if "differential" in literature_output.lower() or "alternative" in literature_output.lower():
                    if not self._mentions_differentials(impression):
                        concern_flags.append(
                            "Literature suggests alternative diagnoses not mentioned in impression"
                        )
                        context_penalty += 0.08
            else:
                # Structured output
                citations = literature_output.citations if hasattr(literature_output, 'citations') else []
                if len(citations) > 0:
                    # Literature found relevant studies
                    if not self._mentions_differentials(impression):
                        concern_flags.append(
                            f"Literature found {len(citations)} relevant studies but impression does not mention differentials"
                        )
                        context_penalty += 0.08
        
        # Apply context penalty to safety score
        safety_score = max(0.0, min(1.0, safety_score - context_penalty))

        # ----------------------------------------------------------------
        # Stage 2: LLM-based semantic critic  >>> LLM-CRITIC
        # ----------------------------------------------------------------
        # Only invoked when:
        #   1. ENABLE_LLM_CRITIC is True
        #   2. KLE uncertainty is high OR rule-based already flagged overconfidence
        # This prevents unnecessary latency on low-risk cases.

        if settings.ENABLE_LLM_CRITIC and (kle_uncertainty > 0.3 or is_overconfident):
            try:
                llm_output = medgemma_critic.critique(
                    findings=findings,
                    impression=impression,
                    kle_uncertainty=kle_uncertainty,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CRITIC-LLM: Semantic critic raised exception: %s — using rule-based output only", exc
                )
                llm_output = None

            if llm_output is not None:
                # -- Merge: is_overconfident ----------------------------------
                if llm_output.overconfidence_reason is not None:
                    is_overconfident = True
                    concern_flags.append(
                        f"[LLM] Overconfidence: {llm_output.overconfidence_reason}"
                    )

                # -- Merge: justification_gap ---------------------------------
                if llm_output.justification_gap:
                    concern_flags.append(
                        f"[LLM] Justification gap: {llm_output.justification_gap}"
                    )

                # -- Merge: missing_differentials -----------------------------
                if llm_output.missing_differentials:
                    formatted = ", ".join(llm_output.missing_differentials)
                    concern_flags.append(
                        f"[LLM] Missing differentials: {formatted}"
                    )

                # -- Merge: recommended_hedging (LLM takes priority) ----------
                if llm_output.suggested_hedging:
                    recommended_hedging = llm_output.suggested_hedging

                # -- Merge: safety_score adjustment ---------------------------
                # Never let LLM directly SET safety_score; only penalise.
                if llm_output.semantic_risk_score > 0.5:
                    safety_score *= (1.0 - 0.3 * llm_output.semantic_risk_score)
                    safety_score = max(0.0, min(1.0, safety_score))

                # -- Trace logging --------------------------------------------
                logger.info(
                    "CRITIC-LLM: Semantic risk=%.2f, Missing differentials=%d",
                    llm_output.semantic_risk_score,
                    len(llm_output.missing_differentials),
                )
            else:
                # LLM failed — log and continue with rule-based output only
                logger.warning(
                    "CRITIC-LLM: Semantic critic unavailable — using rule-based output only"
                )

        return is_overconfident, concern_flags, recommended_hedging, safety_score
    
    def _mentions_differentials(self, impression: str) -> bool:
        """Check if impression mentions differential diagnoses."""
        impression_lower = impression.lower()
        differential_patterns = [
            r'\bdifferential\b', r'\bconsider\b', r'\bvs\.?\b', 
            r'\bversus\b', r'\balternatively\b', r'\balternative\b'
        ]
        return any(re.search(pattern, impression_lower) for pattern in differential_patterns)



# Singleton instance
critic_model = CriticModel()
