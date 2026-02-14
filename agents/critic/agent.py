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
    - Historian FHIR clinical context (NEW)
    - Literature evidence (NEW)
    
    Produces:
    - Boolean overconfidence flag
    - Specific concern flags (including contextual concerns)
    - Recommended hedging language (if needed)
    - Safety score for routing (adjusted for context)
    """
    rad_output = state.get("radiologist_output")
    kle_uncertainty = state.get("radiologist_kle_uncertainty", 0.5)
    
    # NEW: Get enriched context
    hist_output = state.get("historian_output")
    lit_output = state.get("literature_output")
    
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
    
    # NEW: Run critic evaluation with enriched context
    is_overconfident, concern_flags, recommended_hedging, safety_score = critic_model.evaluate(
        findings=findings,
        impression=impression,
        kle_uncertainty=kle_uncertainty,
        historian_output=hist_output,  # NEW
        literature_output=lit_output    # NEW
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
    
    # No additional adjustment needed - context is already factored into safety_score
    
    trace_entry = (
        f"CRITIC: Safety={safety_score:.2%}, Overconfident={'YES' if is_overconfident else 'NO'}, "
        f"KLE={kle_uncertainty:.3f}, Concerns={len(concern_flags)}"
    )
    
    # NEW: Add context trace if available
    if hist_output or lit_output:
        context_info = []
        if hist_output:
            context_info.append("FHIR")
        if lit_output:
            context_info.append("Literature")
        trace_entry += f" [Context: {'+'.join(context_info)}]"
    
    return {
        "critic_output": output,
        "current_uncertainty": round(uncertainty, 3),
        "trace": [trace_entry]
    }
