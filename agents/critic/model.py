"""
Critic Model

Overconfidence detector that evaluates consistency between linguistic certainty
in the radiology report and the externally computed KLE-based uncertainty score.
"""

from app.config import settings
import re


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
        kle_uncertainty: float
    ) -> tuple[bool, list[str], str | None, float]:
        """
        Evaluate whether the linguistic certainty is appropriate given the epistemic uncertainty.
        
        Args:
            findings: FINDINGS text section
            impression: IMPRESSION text section
            kle_uncertainty: Epistemic uncertainty score from KLE (0.0-1.0, higher = more uncertain)
            
        Returns:
            Tuple of (is_overconfident, concern_flags, recommended_hedging, safety_score)
        """
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
        
        return is_overconfident, concern_flags, recommended_hedging, safety_score


# Singleton instance
critic_model = CriticModel()
