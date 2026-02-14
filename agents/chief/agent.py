"""
Chief Orchestrator Agent Node

MedGemma-27B based final arbitration for high-uncertainty cases.
"""

from graph.state import VerifaiState, FinalDiagnosis
from app.config import settings


CHIEF_SYSTEM_PROMPT = """You are the Chief Radiologist AI, a senior expert responsible for final diagnostic arbitration.
You receive structured outputs from multiple AI agents and must:
1. Perform safety checks on all evidence
2. Aggregate opinions with calibrated confidence
3. Either provide a final diagnosis with explanation, OR defer with explicit reasons

Your decisions must be defensible and evidence-based."""

CHIEF_USER_PROMPT = """Given the following structured agent outputs, perform safety checks, aggregate opinions, and provide a final determination.

## Radiologist Findings
{radiologist_summary}

## Patient History (Historian)
{historian_summary}

## Literature Evidence
{literature_summary}

## Critic Flags
{critic_summary}

## Current Uncertainty
{uncertainty:.2%}

Provide either:
(A) Final diagnosis with calibrated confidence and explanation
(B) Explicit deferral with reason and recommended next steps for human review

Output in structured JSON format."""


def chief_node(state: VerifaiState) -> dict:
    """
    Chief Orchestrator: Final arbitration for high-uncertainty cases.
    
    Invoked when:
    - Uncertainty >= 0.50 after all agents
    - Agent disagreement detected
    - Safety-critical decision required
    
    Uses MedGemma-27B (cloud deployment) for sophisticated reasoning.
    """
    rad = state.get("radiologist_output")
    hist = state.get("historian_output")
    lit = state.get("literature_output")
    critic = state.get("critic_output")
    debate = state.get("debate_output")
    uncertainty = state["current_uncertainty"]
    kle_uncertainty = state.get("radiologist_kle_uncertainty", 0.5)
    
    # In production: format prompt and call MedGemma-27B
    # For now: heuristic-based decision
    
    # Gather evidence summary
    evidence_strength = 0
    explanation_parts = []
    
    # RadiologistOutput is now plain text (findings + impression)
    if rad and rad.impression:
        # Base confidence derived from KLE uncertainty
        base_confidence = max(0.1, 1.0 - kle_uncertainty)
        impression_preview = rad.impression[:200] if len(rad.impression) > 200 else rad.impression
        explanation_parts.append(f"Visual: {impression_preview}")
    else:
        base_confidence = 0.0
        impression_preview = None
    
    if hist:
        evidence_strength += len(hist.supporting_facts)
        base_confidence += hist.confidence_adjustment
        if hist.supporting_facts:
            explanation_parts.append(f"History: {len(hist.supporting_facts)} supporting clinical facts")
    
    if lit:
        if lit.overall_evidence_strength == "high":
            evidence_strength += 3
            base_confidence += 0.10
        elif lit.overall_evidence_strength == "medium":
            evidence_strength += 2
            base_confidence += 0.05
        explanation_parts.append(f"Literature: {len(lit.citations)} citations ({lit.overall_evidence_strength} strength)")
    
    if critic:
        if critic.is_overconfident:
            explanation_parts.append("CAUTION: Critic detected potential overconfidence")
    
    # Use debate consensus diagnosis if available
    diagnosis_text = None
    if debate and debate.consensus_diagnosis:
        diagnosis_text = debate.consensus_diagnosis
        base_confidence += debate.total_confidence_adjustment
    elif impression_preview:
        diagnosis_text = impression_preview
    
    # Decision logic
    if uncertainty >= 0.60 or evidence_strength < 2:
        # Defer to human
        final = FinalDiagnosis(
            diagnosis=diagnosis_text,
            calibrated_confidence=max(0.0, min(base_confidence, 0.50)),
            deferred=True,
            deferral_reason=(
                f"Uncertainty ({uncertainty:.0%}) exceeds safety threshold. "
                f"Evidence strength insufficient for automated diagnosis."
            ),
            recommended_next_steps=[
                "Human radiologist review recommended",
                "Consider additional imaging (CT chest) if clinically indicated",
                "Correlate with clinical symptoms and exam findings"
            ],
            explanation=" | ".join(explanation_parts)
        )
        trace_entry = f"CHIEF: DEFERRED (U={uncertainty:.0%}, evidence={evidence_strength})"
    else:
        # Provide calibrated diagnosis
        calibrated_conf = max(0.0, min(base_confidence, 0.95))
        final = FinalDiagnosis(
            diagnosis=diagnosis_text or "Indeterminate",
            calibrated_confidence=calibrated_conf,
            deferred=False,
            deferral_reason=None,
            recommended_next_steps=[
                "Correlate with clinical findings",
                "Consider follow-up imaging in 4-6 weeks if symptoms persist"
            ],
            explanation=" | ".join(explanation_parts)
        )
        trace_entry = f"CHIEF: DIAGNOSED {final.diagnosis[:80] if final.diagnosis else 'N/A'} (confidence={calibrated_conf:.0%})"
    
    return {
        "final_diagnosis": final,
        "trace": [trace_entry]
    }
