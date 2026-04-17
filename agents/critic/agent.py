"""
Critic Agent Node

Evaluates consistency between linguistic certainty and epistemic uncertainty.
"""

from graph.state import VerifaiState, CriticOutput
from .model import critic_model
from uncertainty.muc import compute_ig, compute_critic_uncertainty, compute_critic_alignment


def critic_node(state: VerifaiState) -> dict:
    """
    Critic Agent: Evaluate radiologist output for overconfidence.
    
    Consumes:
    - Radiologist FINDINGS and IMPRESSION text
    - MUC system uncertainty score
    - Historian FHIR clinical context
    - Literature evidence
    - Doctor feedback (NEW: if is_feedback_iteration=True)
    
    Produces:
    - Boolean overconfidence flag
    - Specific concern flags (including contextual concerns and doctor feedback)
    - Recommended hedging language (if needed)
    - Safety score for routing (adjusted for context and feedback)
    """
    rad_output = state.get("radiologist_output")
    current_uncertainty = state.get("current_uncertainty", 0.5)
    
    # Get enriched context
    hist_output = state.get("historian_output")
    lit_output = state.get("literature_output")
    chexbert_output = state.get("chexbert_output")
    
    # NEW: Get doctor feedback if this is a reprocessing iteration
    doctor_feedback = state.get("doctor_feedback")
    is_feedback_iteration = state.get("is_feedback_iteration", False)
    
    if not rad_output:
        # Build appropriate error message
        error_msg = "No radiologist output to evaluate"
        if is_feedback_iteration and doctor_feedback:
            error_msg += f" (feedback iteration for session {doctor_feedback.original_session_id})"
        
        return {
            "critic_output": CriticOutput(
                is_overconfident=True,
                concern_flags=[error_msg],
                recommended_hedging=None,
                safety_score=0.3
            ),
            "current_uncertainty": 0.8,
            "trace": [f"CRITIC: ERROR - {error_msg}"]
        }
    
    # Extract text
    findings = rad_output.findings
    impression = rad_output.impression
    
    # Run critic evaluation with enriched context
    result = critic_model.evaluate(
        findings=findings,
        impression=impression,
        current_uncertainty=current_uncertainty,
        chexbert_output=chexbert_output,
        historian_output=hist_output,
        literature_output=lit_output,
        doctor_feedback=doctor_feedback,
        uncertainty_history=state.get("uncertainty_history", []),
    )
    
    # Unpack result
    if len(result) == 7:
        is_overconfident, concern_flags, recommended_hedging, safety_score, similar_mistakes_count, historical_risk_level, historical_context = result
    elif len(result) == 6:
        is_overconfident, concern_flags, recommended_hedging, safety_score, similar_mistakes_count, historical_risk_level = result
        historical_context = []
    else:
        is_overconfident, concern_flags, recommended_hedging, safety_score = result
        similar_mistakes_count = 0
        historical_risk_level = "none"
        historical_context = []
    
    # NEW: Inject doctor feedback concerns if present
    if is_feedback_iteration and doctor_feedback:
        concern_flags = list(concern_flags)  # Make mutable copy
        
        # Add doctor feedback as a high-priority concern
        feedback_concern = f"DOCTOR FEEDBACK: {doctor_feedback.doctor_notes[:150]}"
        concern_flags.insert(0, feedback_concern)
        
        # If doctor provided correct diagnosis, add it
        if doctor_feedback.correct_diagnosis:
            concern_flags.insert(1, f"Doctor's correct diagnosis: {doctor_feedback.correct_diagnosis}")
        
        # Adjust safety score down since doctor rejected original
        safety_score = max(0.1, safety_score - 0.3)
        is_overconfident = True  # Force reprocessing with higher scrutiny
    
    output = CriticOutput(
        is_overconfident=is_overconfident,
        concern_flags=concern_flags,
        recommended_hedging=recommended_hedging,
        safety_score=round(safety_score, 3),
        similar_mistakes_count=similar_mistakes_count,
        historical_risk_level=historical_risk_level,
        historical_context=historical_context,
    )
    
    # === MUC: Compute Information Gain ===
    critic_unc = compute_critic_uncertainty(safety_score)
    critic_align = compute_critic_alignment(
        safety_score=safety_score,
        is_overconfident=is_overconfident,
        concern_flag_count=len(concern_flags),
    )
    ig_result = compute_ig(
        agent_name="critic",
        agent_uncertainty=critic_unc,
        alignment_score=critic_align,
        system_uncertainty=current_uncertainty,
    )
    
    trace_entry = (
        f"CRITIC: Safety={safety_score:.2%}, Overconfident={'YES' if is_overconfident else 'NO'}, "
        f"Concerns={len(concern_flags)}"
    )
    trace_muc = (
        f"CRITIC MUC: uncertainty={critic_unc:.3f}, alignment={critic_align:.3f}, "
        f"IG={ig_result.information_gain:.4f}"
    )
    
    # Add historical risk indicator if present
    if historical_risk_level != "none":
        trace_entry += f", HistRisk={historical_risk_level.upper()}"
    
    # Add context trace if available
    if hist_output or lit_output:
        context_info = []
        if hist_output:
            context_info.append("FHIR")
        if lit_output:
            context_info.append("Literature")
        trace_entry += f" [Context: {'+'.join(context_info)}]"
    
    # Add feedback trace if this is a reprocessing iteration
    if is_feedback_iteration and doctor_feedback:
        trace_entry += f" [FEEDBACK ITERATION - Original: {doctor_feedback.original_session_id}]"

    # Propagate uncertainty_history
    uncertainty_history = list(state.get("uncertainty_history", []))
    uncertainty_history.append({
        "agent": "critic",
        "system_uncertainty": ig_result.system_uncertainty_after,
    })

    return {
        "critic_output": output,
        "current_uncertainty": ig_result.system_uncertainty_after,
        "uncertainty_history": uncertainty_history,
        "trace": [trace_entry, trace_muc]
    }
