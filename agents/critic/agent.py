"""
Critic Agent Node

Evaluates consistency between linguistic certainty and epistemic uncertainty.
"""

from graph.state import VerifaiState, CriticOutput
from .model import critic_model


def critic_node(state: VerifaiState) -> dict:
    """
    Critic Agent: Evaluate radiologist output for overconfidence.
    
    Consumes:
    - Radiologist FINDINGS and IMPRESSION text
    - KLE-based epistemic uncertainty score
    
    Produces:
    - Boolean overconfidence flag
    - Specific concern flags
    - Recommended hedging language (if needed)
    - Safety score for routing
    """
    rad_output = state.get("radiologist_output")
    kle_uncertainty = state.get("radiologist_kle_uncertainty", 0.5)
    
    if not rad_output:
        return {
            "critic_output": CriticOutput(
                is_overconfident=True,
                concern_flags=["No radiologist output to evaluate"],
                recommended_hedging=None,
                safety_score=0.3
            ),
            "current_uncertainty": 0.8,
            "trace": ["CRITIC: ERROR - No radiologist output"]
        }
    
    # Extract text
    findings = rad_output.findings
    impression = rad_output.impression
    
    # Run critic evaluation
    is_overconfident, concern_flags, recommended_hedging, safety_score = critic_model.evaluate(
        findings=findings,
        impression=impression,
        kle_uncertainty=kle_uncertainty
    )
    
    output = CriticOutput(
        is_overconfident=is_overconfident,
        concern_flags=concern_flags,
        recommended_hedging=recommended_hedging,
        safety_score=round(safety_score, 3)
    )
    
    # Map safety score to uncertainty for routing
    # Lower safety = higher uncertainty
    uncertainty = 1.0 - safety_score
    
    # Adjust based on context (if re-evaluating with historian/literature)
    if state.get("historian_output"):
        uncertainty *= 0.9  # Context reduces uncertainty slightly
    if state.get("literature_output"):
        uncertainty *= 0.9  # Evidence reduces uncertainty slightly
    
    trace_entry = (
        f"CRITIC: Safety={safety_score:.2%}, Overconfident={'YES' if is_overconfident else 'NO'}, "
        f"KLE={kle_uncertainty:.3f}, Concerns={len(concern_flags)}"
    )
    
    return {
        "critic_output": output,
        "current_uncertainty": round(uncertainty, 3),
        "trace": [trace_entry]
    }
