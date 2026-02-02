# agent.py

from graph.state import VerifaiState, HistorianOutput, HistorianFact
from .fhir_client import fhir_client
from .reasoner import reason_over_fhir


def historian_node(state: VerifaiState) -> dict:
    """
    Historian Agent (FINAL)

    For each radiology hypothesis:
    1. Fetch hypothesis-specific FHIR evidence
    2. Reason over evidence using MedGemma-4B
    3. Extract supporting facts, contradicting facts, confidence deltas,
       and FHIR resource IDs

    Returns a fully populated HistorianOutput.
    """

    patient_id = state.get("patient_id")
    rad_output = state.get("radiologist_output")

    if not patient_id or not rad_output or not rad_output.hypotheses:
        return {
            "historian_output": None,
            "trace": ["HISTORIAN: Missing patient_id or radiologist hypotheses"]
        }

    all_supporting: list[HistorianFact] = []
    all_contradicting: list[HistorianFact] = []
    net_confidence_adjustment = 0.0
    trace = []

    for hyp in rad_output.hypotheses:
        hypothesis_name = hyp.diagnosis

        # 1. Fetch FHIR evidence
        evidence = fhir_client.fetch_evidence_for_hypothesis(
            patient_id=patient_id,
            hypothesis=hypothesis_name
        )
        # 2. Reason with MedGemma-4B
        
        reasoning = reason_over_fhir(
            hypothesis=hypothesis_name,
            evidence=evidence
        )

        """
        Expected reasoning structure (parsed upstream or directly returned):
        {
          "supporting_facts": [
            {"description": "...", "resource_type": "...", "resource_id": "..."}
          ],
          "contradicting_facts": [...],
          "confidence_adjustment": +0.12
        }
        """
        # 3. Convert reasoning → HistorianFact
        
        for fact in reasoning.get("supporting_facts", []):
            all_supporting.append(
                HistorianFact(
                    fact_type="supporting",
                    description=f"[{hypothesis_name}] {fact['description']}",
                    fhir_resource_id=fact.get("resource_id"),
                    fhir_resource_type=fact.get("resource_type")
                )
            )

        for fact in reasoning.get("contradicting_facts", []):
            all_contradicting.append(
                HistorianFact(
                    fact_type="contradicting",
                    description=f"[{hypothesis_name}] {fact['description']}",
                    fhir_resource_id=fact.get("resource_id"),
                    fhir_resource_type=fact.get("resource_type")
                )
            )

        # 4. Accumulate confidence delta
        delta = reasoning.get("confidence_adjustment", 0.0)
        net_confidence_adjustment += delta

        trace.append(
            f"HISTORIAN: {hypothesis_name} Δconfidence={delta:+.2f}"
        )

    # 5. Build final HistorianOutput

    output = HistorianOutput(
        supporting_facts=all_supporting,
        contradicting_facts=all_contradicting,
        confidence_adjustment=round(net_confidence_adjustment, 3),
        clinical_summary=(
            f"Evaluated {len(rad_output.hypotheses)} hypotheses using "
            f"FHIR-grounded historical evidence."
        )
    )

    trace.append(
        f"HISTORIAN: Total Δconfidence={output.confidence_adjustment:+.2f}"
    )

    return {
        "historian_output": output,
        "trace": trace
    }
