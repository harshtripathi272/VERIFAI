"""
Historian Agent Node

Retrieves temporally-aligned patient context via FHIR.
"""

from graph.state import VerifaiState, HistorianOutput, HistorianFact
from .fhir_client import get_patient_context


def historian_node(state: VerifaiState) -> dict:
    """
    Historian Agent: Retrieve patient clinical context.
    
    Given patient_id and hypotheses from Radiologist, queries FHIR for:
    - Active conditions
    - Recent laboratory observations
    - Current medications
    - Prior imaging for comparison
    
    Returns supporting/contradicting facts and confidence adjustments.
    """
    patient_id = state.get("patient_id")
    rad_output = state.get("radiologist_output")
    
    if not patient_id:
        return {
            "historian_output": None,
            "trace": ["HISTORIAN: No patient ID provided, skipping context retrieval"]
        }
    
    # Extract hypotheses for targeted context retrieval
    hypotheses = []
    if rad_output and rad_output.hypotheses:
        hypotheses = [h.diagnosis for h in rad_output.hypotheses[:3]]
    
    # Query FHIR
    context = get_patient_context(patient_id, hypotheses)
    
    # Build output
    output = HistorianOutput(
        supporting_facts=context.get("supporting_facts", []),
        contradicting_facts=context.get("contradicting_facts", []),
        confidence_adjustment=context.get("confidence_adjustment", 0.0),
        clinical_summary=context.get("clinical_summary", "")
    )
    
    n_support = len(output.supporting_facts)
    n_contradict = len(output.contradicting_facts)
    
    trace_entry = (
        f"HISTORIAN: Retrieved context for {patient_id}. "
        f"Supporting={n_support}, Contradicting={n_contradict}, "
        f"Adjustment={output.confidence_adjustment:+.0%}"
    )
    
    return {
        "historian_output": output,
        "trace": [trace_entry]
    }
