"""
Critic Model

Overconfidence detector trained on PCam-derived uncertainty features.
Consumes MedSigLIP embeddings + radiologist internal signals to detect miscalibration.
"""

from graph.state import InternalSignals
from app.config import settings


class CriticModel:
    """
    Uncertainty classifier that detects radiologist overconfidence.
    
    In production, this would be a trained classifier (MLP or similar)
    that learned overconfidence patterns from the PCam dataset.
    """
    
    def __init__(self):
        self.mock = settings.MOCK_MODELS
        self._model = None
        
        if not self.mock:
            self._load_model()
    
    def _load_model(self):
        """Load trained overconfidence classifier."""
        # TODO: Load trained sklearn/pytorch classifier
        pass
    
    def evaluate(
        self,
        signals: InternalSignals,
        top_confidence: float,
        embedding: any = None
    ) -> tuple[float, float, list[str]]:
        """
        Evaluate overconfidence probability.
        
        Args:
            signals: Internal predictive signals from radiologist
            top_confidence: Top hypothesis confidence
            embedding: Optional MedSigLIP embedding for additional features
            
        Returns:
            Tuple of (overconfidence_prob, calculated_uncertainty, concern_signals)
        """
        # Normalize signals to uncertainty contributions
        # Low margin = high uncertainty
        margin_uncertainty = 1.0 - min(signals.logit_margin / 5.0, 1.0)
        
        # High entropy = high uncertainty
        entropy_uncertainty = min(signals.predictive_entropy / 2.0, 1.0)
        
        # High dispersion = uncertain focus
        dispersion_uncertainty = 1.0 - signals.attention_dispersion
        
        # Low stability = high uncertainty
        stability_uncertainty = 1.0 - signals.prediction_stability
        
        # Weighted combination (architecture spec weights)
        base_uncertainty = (
            0.35 * entropy_uncertainty +
            0.25 * margin_uncertainty +
            0.20 * dispersion_uncertainty +
            0.20 * stability_uncertainty
        )
        
        # Detect overconfidence pattern:
        # Model claims high confidence but signals indicate uncertainty
        overconfidence_prob = 0.0
        concern_signals = []
        
        if top_confidence > 0.75 and base_uncertainty > 0.4:
            overconfidence_prob = 0.8
            concern_signals.append("High confidence despite uncertain internal signals")
        elif top_confidence > 0.60 and base_uncertainty > 0.5:
            overconfidence_prob = 0.5
            concern_signals.append("Moderate confidence with elevated uncertainty signals")
        
        if signals.logit_margin < 1.5:
            concern_signals.append(f"Low logit margin ({signals.logit_margin:.2f})")
        
        if signals.attention_dispersion < 0.4:
            concern_signals.append(f"Scattered attention (dispersion={signals.attention_dispersion:.2f})")
        
        if signals.predictive_entropy > 0.7:
            concern_signals.append(f"High predictive entropy ({signals.predictive_entropy:.2f})")
        
        # Final uncertainty incorporates overconfidence signal
        calculated_uncertainty = (
            0.70 * base_uncertainty + 
            0.30 * overconfidence_prob
        )
        
        return overconfidence_prob, calculated_uncertainty, concern_signals


# Singleton instance
critic_model = CriticModel()
