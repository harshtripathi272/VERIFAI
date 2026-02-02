"""
Evidence Packet Compiler

Assembles all agent outputs into a verifiable evidence packet.
"""

from typing import Any
from graph.state import VerifaiState


def compile_evidence_packet(state: VerifaiState) -> dict[str, Any]:
    """
    Compile all evidence into a structured, verifiable packet.
    
    Includes:
    - Visual evidence (findings, saliency)
    - Clinical context (FHIR resources)
    - Literature citations (with provenance)
    - Critic assessment
    - Complete audit trail
    """
    packet = {
        "version": "2.0",
        "visual_evidence": None,
        "clinical_context": None,
        "literature_support": None,
        "uncertainty_assessment": None,
        "audit_trail": state.get("trace", [])
    }
    
    # Visual evidence
    rad = state.get("radiologist_output")
    if rad:
        packet["visual_evidence"] = {
            "findings": [f.model_dump() for f in rad.findings],
            "hypotheses": [h.model_dump() for h in rad.hypotheses],
            "reasoning": rad.reasoning,
            "internal_signals": rad.internal_signals.model_dump(),
            # Placeholder for actual saliency map
            "saliency_map_path": None
        }
    
    # Clinical context
    hist = state.get("historian_output")
    if hist:
        packet["clinical_context"] = {
            "supporting_facts": [f.model_dump() for f in hist.supporting_facts],
            "contradicting_facts": [f.model_dump() for f in hist.contradicting_facts],
            "confidence_adjustment": hist.confidence_adjustment,
            "summary": hist.clinical_summary
        }
    
    # Literature
    lit = state.get("literature_output")
    if lit:
        packet["literature_support"] = {
            "citations": [c.model_dump() for c in lit.citations],
            "overall_strength": lit.overall_evidence_strength
        }
    
    # Uncertainty
    critic = state.get("critic_output")
    if critic:
        packet["uncertainty_assessment"] = {
            "calculated_uncertainty": critic.calculated_uncertainty,
            "overconfidence_probability": critic.overconfidence_probability,
            "concern_signals": critic.concern_signals,
            "counter_hypotheses": critic.counter_hypotheses
        }
    
    return packet
