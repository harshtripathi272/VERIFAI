# agent.py

import re
from graph.state import VerifaiState, HistorianOutput, HistorianFact
from .fhir_client import fhir_client
from .reasoner import reason_over_fhir
from uncertainty.muc import (
    compute_ig,
    compute_historian_uncertainty,
    compute_historian_alignment,
)
import logging

logger = logging.getLogger("historian")

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(name)s] %(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False

def log_retrieved_evidence(hypothesis: str, evidence: dict, top_k: int = 3):
    """
    Log top retrieved evidence for debugging hybrid retrieval.
    """

    logger.info(f"\n=== Retrieved Evidence for '{hypothesis}' ===")

    for category in ["conditions", "observations", "documents"]:
        items = evidence.get(category, [])
        logger.debug(f"{category.upper()} count: {len(items)}")

        for i, item in enumerate(items[:top_k]):
            score = item.get("_relevance_score", "N/A")
            summary = item.get("_summary", "")

            if isinstance(summary, str):
                summary = summary[:150].replace("\n", " ")

            logger.debug(
                f"  {i+1}. score={score} | {summary}"
            )

    logger.debug("==========================================")




def extract_diagnostic_concepts(impression: str) -> list[str]:
    """
    Extract diagnostic hypotheses from plain-text radiologist impression.
    Uses regex patterns to identify diagnostic concepts.
    """

    if not impression:
        return []

    concepts = []

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

    # Fallback: use first sentence
    if not concepts:
        first_sentence = impression.split('.')[0].strip()
        if first_sentence and len(first_sentence) < 200:
            concepts.append(first_sentence)

    # Deduplicate (max 3)
    seen = set()
    unique = []
    for c in concepts:
        c_lower = c.lower()
        if c_lower not in seen:
            seen.add(c_lower)
            unique.append(c)
        if len(unique) >= 3:
            break

    return unique

# Historian Agent Node
def historian_node(state: VerifaiState) -> dict:
    """
    Historian Agent

    Uses:
    - Radiologist impression (text extraction)
    - CheXbert structured labels

    For each hypothesis:
        1. Fetch hypothesis-specific hybrid FHIR evidence
        2. Run MedGemma reasoning
        3. Accumulate supporting/contradicting facts
        4. Aggregate confidence adjustments
    """

    patient_id = state.get("patient_id")
    rad_output = state.get("radiologist_output")
    chexbert_output = state.get("chexbert_output")
    current_fhir = state.get("current_fhir")

    # -------------------------------------------------------
    # Validate Inputs
    # -------------------------------------------------------

    if (not patient_id or patient_id == "N/A") and not current_fhir:
        return {
            "historian_output": None,
            "trace": ["HISTORIAN: Missing patient_id and no current FHIR report provided"]
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

    hypotheses = []

    # 1. Extract from impression
    text_concepts = extract_diagnostic_concepts(rad_output.impression)
    hypotheses.extend(text_concepts)

    # 2. Add CheXbert labels
    if chexbert_output and chexbert_output.labels:
        for label in chexbert_output.labels.keys():
            hypotheses.append(label)

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

    
    # Prepare Aggregation
    all_supporting: list[HistorianFact] = []
    all_contradicting: list[HistorianFact] = []
    net_confidence_adjustment = 0.0

    trace = [
        f"HISTORIAN: Analyzing {len(hypotheses)} conditions",
        "HISTORIAN: Using hypothesis-specific hybrid retrieval"
    ]

    # -------------------------------------------------------
    # Process Each Hypothesis (Correct Hybrid Flow)
    # -------------------------------------------------------

    for hypothesis_name in hypotheses:

        try:
            # 🔥 Hypothesis-specific hybrid retrieval
            evidence = fhir_client.fetch_evidence_hybrid(
                patient_id or "global_pattern",
                hypothesis_name
            )
        except Exception as e:
            trace.append(
                f"HISTORIAN: FHIR retrieval error for '{hypothesis_name}': {str(e)}"
            )
            continue
        # 🔍 DEBUG LOGGING
        log_retrieved_evidence(hypothesis_name, evidence)

        # 🔥 Single MedGemma reasoning call
        reasoning_output = reason_over_fhir(
            hypothesis=hypothesis_name,
            evidence=evidence,
            current_fhir=current_fhir,
            uncertainty_history=state.get("uncertainty_history", []),
        )

        # ---------------------------------------------------
        # Accumulate Supporting Facts
        # ---------------------------------------------------

        for fact in reasoning_output.supporting_facts:
            fact.description = f"[{hypothesis_name}] {fact.description}"
            all_supporting.append(fact)

        # ---------------------------------------------------
        # Accumulate Contradicting Facts
        # ---------------------------------------------------

        for fact in reasoning_output.contradicting_facts:
            fact.description = f"[{hypothesis_name}] {fact.description}"
            all_contradicting.append(fact)

        # ---------------------------------------------------
        # Accumulate Confidence
        # ---------------------------------------------------

        net_confidence_adjustment += reasoning_output.confidence_adjustment

        trace.append(
            f"HISTORIAN: {hypothesis_name} Δconfidence="
            f"{reasoning_output.confidence_adjustment:+.2f}"
        )

    # -------------------------------------------------------
    # Final Structured Output
    # -------------------------------------------------------

    output = HistorianOutput(
        supporting_facts=all_supporting,
        contradicting_facts=all_contradicting,
        confidence_adjustment=round(net_confidence_adjustment, 3),
        clinical_summary=(
            f"Evaluated {len(hypotheses)} diagnostic concepts using "
            f"hypothesis-specific FHIR-grounded historical evidence."
        )
    )

    trace.append(
        f"HISTORIAN: Total Δconfidence={output.confidence_adjustment:+.2f}"
    )

    # === MUC: Compute Information Gain for Historian ===
    system_uncertainty = state.get("current_uncertainty", 0.5)
    h_unc = compute_historian_uncertainty(
        supporting_count=len(all_supporting),
        contradicting_count=len(all_contradicting),
    )
    h_align = compute_historian_alignment(
        supporting_count=len(all_supporting),
        contradicting_count=len(all_contradicting),
        confidence_adjustment=net_confidence_adjustment,
    )
    ig_result = compute_ig(
        agent_name="historian",
        agent_uncertainty=h_unc,
        alignment_score=h_align,
        system_uncertainty=system_uncertainty,
    )
    trace.append(
        f"HISTORIAN MUC: unc={h_unc:.3f}, align={h_align:.3f}, "
        f"IG={ig_result.information_gain:.4f}, "
        f"U: {system_uncertainty:.4f} -> {ig_result.system_uncertainty_after:.4f}"
    )

    # Append to uncertainty_history
    uncertainty_history = list(state.get("uncertainty_history", []))
    uncertainty_history.append({
        "agent": "historian",
        "system_uncertainty": ig_result.system_uncertainty_after,
    })

    return {
        "historian_output": output,
        "current_uncertainty": ig_result.system_uncertainty_after,
        "uncertainty_history": uncertainty_history,
        "trace": trace
    }
