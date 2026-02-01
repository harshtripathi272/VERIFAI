"""
Critic Agent Node

Detects overconfidence in radiologist output using PCam-trained classifier.
"""

from graph.state import VerifaiState, CriticOutput
from .model import critic_model


def critic_node(state: VerifaiState) -> dict:
    """
    Critic Agent: Evaluate radiologist output for overconfidence.
    
    Consumes:
    - Radiologist internal signals (logits, entropy, attention)
    - MedSigLIP embeddings (if available)
    
    Produces:
    - Overconfidence probability
    - Counter-hypotheses to consider
    - Specific signals that triggered concern
    - Calculated uncertainty score (used for routing)
    """
    rad_output = state.get("radiologist_output")
    
    if not rad_output:
        return {
            "critic_output": CriticOutput(
                overconfidence_probability=0.5,
                counter_hypotheses=[],
                concern_signals=["No radiologist output to evaluate"],
                calculated_uncertainty=0.5
            ),
            "current_uncertainty": 0.5,
            "trace": ["CRITIC: ERROR - No radiologist output"]
        }
    
    # Get top confidence from radiologist
    top_confidence = 0.0
    if rad_output.hypotheses:
        top_confidence = rad_output.hypotheses[0].confidence
    
    # Run critic evaluation
    overconf_prob, uncertainty, concern_signals = critic_model.evaluate(
        signals=rad_output.internal_signals,
        top_confidence=top_confidence,
        embedding=None  # Would pass embedding if available
    )
    
    # Generate counter-hypotheses if overconfident
    counter_hypotheses = []
    if overconf_prob > 0.3 and len(rad_output.hypotheses) > 1:
        # Suggest reviewing lower-ranked hypotheses
        for h in rad_output.hypotheses[1:3]:
            counter_hypotheses.append(
                f"Consider {h.diagnosis} (originally {h.confidence:.0%})"
            )
    
    # Adjust uncertainty based on prior context (if re-evaluating)
    if state.get("historian_output"):
        uncertainty *= 0.85  # Context reduces uncertainty
    if state.get("literature_output"):
        uncertainty *= 0.85  # Evidence reduces uncertainty
    
    output = CriticOutput(
        overconfidence_probability=round(overconf_prob, 3),
        counter_hypotheses=counter_hypotheses,
        concern_signals=concern_signals,
        calculated_uncertainty=round(uncertainty, 3)
    )
    
    trace_entry = f"CRITIC: U={uncertainty:.2%}, Overconf={overconf_prob:.2%}, Concerns={len(concern_signals)}"
    
    return {
        "critic_output": output,
        "current_uncertainty": uncertainty,
        "trace": [trace_entry]
    }
