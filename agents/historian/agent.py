# agent.py

import re
from graph.state import VerifaiState, HistorianOutput, HistorianFact
from .fhir_client import fhir_client
from .reasoner import reason_over_fhir


def extract_diagnostic_concepts(impression: str) -> list[str]:
    """
    Extract diagnostic hypotheses from plain-text radiologist impression.
    
    Uses regex patterns to identify diagnostic concepts from common
    radiology phrasing patterns.
    
    Args:
        impression: Plain-text impression field from RadiologistOutput
    
    Returns:
        List of extracted diagnostic concepts (max 3)
    """
    if not impression:
        return []
    
    concepts = []
    
    # Pattern 1: "consistent with X", "suggestive of X"
    patterns = [
        r'consistent with ([^.,;]+)',
        r'suggestive of ([^.,;]+)',
        r'findings (?:concerning for|raise concern for) ([^.,;]+)',
        r'(?:possible|probable|likely) ([^.,;]+)',
        r'(?:differential includes?|consider) ([^.,;]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, impression, re.IGNORECASE)
        concepts.extend([m.strip() for m in matches if m.strip()])
    
    # Fallback: Use first sentence if no patterns match
    if not concepts:
        first_sentence = impression.split('.')[0].strip()
        if first_sentence and len(first_sentence) < 200:
            concepts.append(first_sentence)
    
    # Deduplicate and limit to top 3
    seen = set()
    unique_concepts = []
    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower not in seen:
            seen.add(concept_lower)
            unique_concepts.append(concept)
        if len(unique_concepts) >= 3:
            break
    
    return unique_concepts


def historian_node(state: VerifaiState) -> dict:
    """
    Historian Agent

    Extracts diagnostic concepts from radiologist's plain-text impression,
    then for each concept:
    1. Fetch hypothesis-specific FHIR evidence
    2. Reason over evidence using MedGemma-4B
    3. Extract supporting facts, contradicting facts, confidence deltas,
       and FHIR resource IDs

    Returns a fully populated HistorianOutput.
    """

    patient_id = state.get("patient_id")
    rad_output = state.get("radiologist_output")

    # NEW: Check for impression text instead of hypotheses
    if not patient_id or not rad_output or not rad_output.impression:
        return {
            "historian_output": None,
            "trace": ["HISTORIAN: Missing patient_id or radiologist impression"]
        }

    # NEW: Extract diagnostic concepts from impression
    hypotheses = extract_diagnostic_concepts(rad_output.impression)
    
    if not hypotheses:
        return {
            "historian_output": None,
            "trace": ["HISTORIAN: Could not extract diagnostic concepts from impression"]
        }
    
    all_supporting: list[HistorianFact] = []
    all_contradicting: list[HistorianFact] = []
    net_confidence_adjustment = 0.0
    trace = [f"HISTORIAN: Extracted {len(hypotheses)} concepts from impression"]

    for hypothesis_name in hypotheses:

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
            f"Evaluated {len(hypotheses)} diagnostic concepts using "
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
