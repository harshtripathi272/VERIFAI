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

    Uses BOTH radiologist findings and CheXbert labels to gather FHIR evidence.
    
    Inputs:
    - radiologist_output.findings: Detailed observations
    - radiologist_output.impression: Diagnostic conclusion
    - chexbert_output.labels: Structured pathology labels (present/uncertain)
    
    For each condition (from impression + CheXbert):
    1. Fetch hypothesis-specific FHIR evidence
    2. Reason over evidence using MedGemma-4B
    3. Extract supporting facts, contradicting facts, confidence deltas

    Returns a fully populated HistorianOutput.
    """

    patient_id = state.get("patient_id")
    rad_output = state.get("radiologist_output")
    chexbert_output = state.get("chexbert_output")

    # Validate inputs: Require patient_id AND radiologist output with BOTH findings/impression
    if not patient_id:
        return {
            "historian_output": None,
            "trace": ["HISTORIAN: Missing patient_id"]
        }
        
    if not rad_output:
        return {
            "historian_output": None,
            "trace": ["HISTORIAN: No radiologist output available"]
        }
        
    if not rad_output.impression or not rad_output.findings:
        return {
            "historian_output": None,
            "trace": ["HISTORIAN: Missing findings or impression in radiologist report"]
        }

    # Build hypothesis list from BOTH sources
    hypotheses = []
    
    # 1. Extract concepts from impression text
    text_concepts = extract_diagnostic_concepts(rad_output.impression)
    hypotheses.extend(text_concepts)
    
    # 2. Add CheXbert labels (present and uncertain conditions)
    if chexbert_output and chexbert_output.labels:
        for condition in chexbert_output.labels.keys():
            hypotheses.append(condition)
    
    # Deduplicate while preserving order
    seen = set()
    unique_hypotheses = []
    for h in hypotheses:
        h_lower = h.lower()
        if h_lower not in seen:
            seen.add(h_lower)
            unique_hypotheses.append(h)
    
    hypotheses = unique_hypotheses
    
    if not hypotheses:
        return {
            "historian_output": None,
            "trace": ["HISTORIAN: Could not extract diagnostic concepts"]
        }
    
    all_supporting: list[HistorianFact] = []
    all_contradicting: list[HistorianFact] = []
    net_confidence_adjustment = 0.0
    trace = [
        f"HISTORIAN: Analyzing {len(hypotheses)} conditions",
        f"HISTORIAN: Sources - Impression text + CheXbert labels"
    ]

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
