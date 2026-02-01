"""
VERIFAI Agent Nodes

Each function is a LangGraph node that takes the state, performs its logic,
and returns a partial state update. These are mock implementations that
simulate HAI-DEF model behavior.
"""

import random
from datetime import datetime, timezone
from app.state import (
    VerifaiState,
    RadiologistOutput,
    CriticOutput,
    HistorianOutput,
    LiteratureOutput,
    FinalDiagnosis,
    Finding,
    DiagnosisCandidate,
    RawUncertaintySignals,
    Citation,
)


# --- Uncertainty Thresholds ---
THRESHOLD_HISTORIAN = 0.30
THRESHOLD_LITERATURE = 0.40
THRESHOLD_CHIEF = 0.50
MAX_STEPS = 5  # Safety limit for loop prevention


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================
# RADIOLOGIST NODE
# =============================================================================
def radiologist_node(state: VerifaiState) -> dict:
    """
    Simulates MedGemma-4B + MedSigLIP processing the chest X-ray.

    In production, this would:
    1. Load DICOM via MedSigLIP encoder (frozen)
    2. Run MedGemma-4B with LoRA adapter
    3. Extract attention maps, logits, and generate findings

    For this mock, we simulate plausible outputs.
    """
    # Simulate finding detection (in real system: vision model inference)
    findings = [
        Finding(location="RLL", observation="opacity", severity=0.7),
        Finding(location="Left hilum", observation="mild prominence", severity=0.3),
    ]

    differential = [
        DiagnosisCandidate(diagnosis="Community-Acquired Pneumonia", probability=0.65),
        DiagnosisCandidate(diagnosis="Atelectasis", probability=0.20),
        DiagnosisCandidate(diagnosis="Pulmonary Edema", probability=0.10),
    ]

    # Simulate raw uncertainty signals
    raw_signals = RawUncertaintySignals(
        logit_margin=2.1,  # Moderate margin
        entropy=0.45,  # Moderate entropy
        attention_dispersion=0.55,  # Somewhat focused
        prediction_stability=0.12,  # Relatively stable
    )

    output = RadiologistOutput(
        findings=findings,
        differential=differential,
        raw_signals=raw_signals,
        reasoning=(
            "Right lower lobe opacity with air bronchograms suggestive of consolidation. "
            "Mild left hilar prominence noted, possibly reactive lymphadenopathy."
        ),
    )

    trace_entry = f"[{_timestamp()}] RADIOLOGIST: Identified {len(findings)} findings, top Dx: {differential[0].diagnosis} ({differential[0].probability:.0%})"

    return {
        "radiologist_output": output,
        "trace": [trace_entry],
    }


# =============================================================================
# CRITIC NODE
# =============================================================================
def critic_node(state: VerifaiState) -> dict:
    """
    Simulates the PCam-trained Critic Head that detects overconfidence.

    Takes raw signals from radiologist and computes combined uncertainty.
    This is the "novel task" - trained to recognize patterns of
    miscalibration that transfer across medical imaging domains.
    """
    rad_output = state["radiologist_output"]
    if rad_output is None:
        # Should not happen in normal flow
        return {
            "critic_output": CriticOutput(
                overconfidence_score=0.5,
                critiques=["No radiologist output to critique"],
                calculated_uncertainty=0.5,
            ),
            "current_uncertainty": 0.5,
            "trace": [f"[{_timestamp()}] CRITIC: ERROR - No radiologist output"],
        }

    signals = rad_output.raw_signals

    # Normalize signals to [0, 1] range for uncertainty calculation
    logit_margin_norm = 1 - min(signals.logit_margin / 5.0, 1.0)  # Low margin = high uncertainty
    entropy_norm = min(signals.entropy / 1.0, 1.0)  # High entropy = high uncertainty
    dispersion_norm = 1 - signals.attention_dispersion  # Scattered attention = uncertain
    stability_norm = min(signals.prediction_stability / 0.3, 1.0)  # High instability = uncertain

    # Weighted combination (as per architecture doc)
    base_uncertainty = (
        0.35 * entropy_norm
        + 0.20 * logit_margin_norm
        + 0.10 * dispersion_norm
        + 0.10 * stability_norm
    )

    # Simulate critic's overconfidence detection (in production: trained classifier)
    # Higher score if top probability is very high but signals are mixed
    top_prob = rad_output.differential[0].probability if rad_output.differential else 0.5
    overconfidence_signal = max(0, top_prob - (1 - base_uncertainty) - 0.1)
    overconfidence_score = min(overconfidence_signal + random.uniform(0.05, 0.15), 1.0)

    # Incorporate critic's assessment into final uncertainty
    calculated_uncertainty = (
        0.65 * base_uncertainty + 0.35 * overconfidence_score
    )

    # Generate critiques
    critiques = []
    if signals.attention_dispersion < 0.5:
        critiques.append("Attention is relatively scattered - model may be uncertain about focal region")
    if signals.logit_margin < 2.0:
        critiques.append("Low logit margin between top diagnoses - consider differential more carefully")
    if overconfidence_score > 0.25:
        critiques.append("Overconfidence pattern detected - recommend additional context")

    output = CriticOutput(
        overconfidence_score=round(overconfidence_score, 3),
        critiques=critiques,
        calculated_uncertainty=round(calculated_uncertainty, 3),
    )

    trace_entry = f"[{_timestamp()}] CRITIC: Uncertainty={calculated_uncertainty:.2%}, Overconfidence={overconfidence_score:.2%}"

    return {
        "critic_output": output,
        "current_uncertainty": calculated_uncertainty,
        "trace": [trace_entry],
    }


# =============================================================================
# HISTORIAN NODE
# =============================================================================
def historian_node(state: VerifaiState) -> dict:
    """
    Simulates MedGemma-4B + FHIR MCP tools retrieving patient context.

    In production, this would:
    1. Call FHIR MCP server to get conditions, labs, medications
    2. Use MedGemma to synthesize relevant clinical context
    3. Adjust probability based on risk factors
    """
    patient_id = state.get("patient_id", "UNKNOWN")

    # Simulated FHIR retrieval
    output = HistorianOutput(
        relevant_conditions=["Type 2 Diabetes Mellitus (E11.9)", "Hypertension (I10)"],
        risk_factors=["Immunocompromised risk due to diabetes", "Recent hospitalization (2 weeks ago)"],
        relevant_labs={
            "WBC": 14500.0,  # Elevated
            "CRP": 85.0,  # Elevated
            "Procalcitonin": 0.8,  # Borderline
        },
        prior_imaging_comparison="New opacity compared to baseline from 3 months ago",
        clinical_summary=(
            f"Patient {patient_id} is a diabetic with elevated inflammatory markers (WBC, CRP) "
            "and new pulmonary opacity not present on prior imaging. High risk for bacterial infection."
        ),
        probability_adjustment=0.12,  # Increases pneumonia likelihood
    )

    # Context typically reduces uncertainty
    uncertainty_reduction = 0.08

    trace_entry = f"[{_timestamp()}] HISTORIAN: Retrieved context for patient {patient_id}. Risk factors: {len(output.risk_factors)}, Labs retrieved: {len(output.relevant_labs)}"

    return {
        "historian_output": output,
        "current_uncertainty": max(0, state["current_uncertainty"] - uncertainty_reduction),
        "trace": [trace_entry],
    }


# =============================================================================
# LITERATURE NODE
# =============================================================================
def literature_node(state: VerifaiState) -> dict:
    """
    Simulates MedGemma-4B + RAG over PubMed retrieving evidence.

    In production, this would:
    1. Call PubMed MCP server with relevant queries
    2. Retrieve and rank abstracts by relevance
    3. Synthesize supporting/contradicting evidence
    """
    rad_output = state.get("radiologist_output")
    top_diagnosis = "pneumonia"
    if rad_output and rad_output.differential:
        top_diagnosis = rad_output.differential[0].diagnosis.lower()

    # Simulated literature retrieval
    supporting = [
        Citation(
            pmid="38472615",
            title="Radiographic Patterns in Community-Acquired Pneumonia: A Systematic Review",
            relevance=0.92,
            excerpt="Lobar consolidation with air bronchograms is highly specific for bacterial pneumonia (specificity 94%, CI 91-96%)",
        ),
        Citation(
            pmid="39182734",
            title="Outcomes of Pneumonia in Diabetic Patients: Meta-Analysis",
            relevance=0.78,
            excerpt="Diabetic patients show 2.3x mortality risk in CAP; early aggressive treatment recommended",
        ),
    ]

    contradicting = []
    if "atelectasis" in state.get("radiologist_output", RadiologistOutput(raw_signals=RawUncertaintySignals(logit_margin=0, entropy=0, attention_dispersion=0, prediction_stability=0), findings=[], differential=[], reasoning="")).reasoning.lower():
        contradicting.append(
            Citation(
                pmid="37891234",
                title="Differentiating Atelectasis from Consolidation on Chest Radiographs",
                relevance=0.65,
                excerpt="Volume loss and shift toward opacity favors atelectasis over pneumonia",
            )
        )

    output = LiteratureOutput(
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        evidence_strength="moderate" if len(supporting) >= 2 else "weak",
    )

    # Strong evidence reduces uncertainty
    uncertainty_reduction = 0.10 if output.evidence_strength == "moderate" else 0.05

    trace_entry = f"[{_timestamp()}] LITERATURE: Found {len(supporting)} supporting, {len(contradicting)} contradicting citations. Strength: {output.evidence_strength}"

    return {
        "literature_output": output,
        "current_uncertainty": max(0, state["current_uncertainty"] - uncertainty_reduction),
        "trace": [trace_entry],
    }


# =============================================================================
# CHIEF ORCHESTRATOR NODE
# =============================================================================
def chief_node(state: VerifaiState) -> dict:
    """
    Simulates MedGemma-27B performing conflict resolution and final arbitration.

    Only invoked when:
    - Uncertainty remains >= 0.50 after all agents
    - Agent disagreement detected
    - Safety-critical decision required

    In production, this runs on cloud (larger model).
    """
    rad_output = state.get("radiologist_output")
    historian_output = state.get("historian_output")
    literature_output = state.get("literature_output")
    current_uncertainty = state["current_uncertainty"]

    # Attempt to reach a diagnosis or explicitly defer
    if current_uncertainty >= 0.60:
        # Too uncertain - explicit deferral
        diagnosis = FinalDiagnosis(
            diagnosis=None,
            confidence=1 - current_uncertainty,
            deferred=True,
            deferral_reason=(
                f"Uncertainty ({current_uncertainty:.0%}) exceeds safety threshold. "
                "Recommend human radiologist review with provided evidence packet."
            ),
        )
        trace_entry = f"[{_timestamp()}] CHIEF: DEFERRED to human review. Uncertainty={current_uncertainty:.2%}"
    else:
        # Synthesize final diagnosis with calibrated confidence
        top_dx = "Community-Acquired Pneumonia"
        if rad_output and rad_output.differential:
            top_dx = rad_output.differential[0].diagnosis

        # Adjust confidence based on all inputs
        base_confidence = 1 - current_uncertainty
        if historian_output and historian_output.probability_adjustment > 0:
            base_confidence = min(0.95, base_confidence + 0.05)
        if literature_output and literature_output.evidence_strength == "moderate":
            base_confidence = min(0.95, base_confidence + 0.03)

        diagnosis = FinalDiagnosis(
            diagnosis=top_dx,
            confidence=round(base_confidence, 2),
            deferred=False,
            deferral_reason=None,
        )
        trace_entry = f"[{_timestamp()}] CHIEF: Final Dx={top_dx}, Confidence={base_confidence:.0%}"

    return {
        "final_diagnosis": diagnosis,
        "trace": [trace_entry],
    }


# =============================================================================
# ROUTING NODE
# =============================================================================
def router_node(state: VerifaiState) -> dict:
    """
    Determines next step based on current uncertainty and steps taken.

    Returns routing_decision which is used by conditional edges.
    """
    uncertainty = state["current_uncertainty"]
    steps = state.get("steps_taken", 0)

    # Safety: prevent infinite loops
    if steps >= MAX_STEPS:
        decision = "chief"
        trace_entry = f"[{_timestamp()}] ROUTER: Max steps reached ({steps}), escalating to CHIEF"
    elif uncertainty < THRESHOLD_HISTORIAN:
        decision = "diagnose"
        trace_entry = f"[{_timestamp()}] ROUTER: U={uncertainty:.2%} < {THRESHOLD_HISTORIAN:.0%}, proceeding to DIAGNOSIS"
    elif uncertainty < THRESHOLD_LITERATURE:
        # Check if we already have historian output
        if state.get("historian_output") is None:
            decision = "historian"
            trace_entry = f"[{_timestamp()}] ROUTER: U={uncertainty:.2%}, invoking HISTORIAN"
        else:
            decision = "diagnose"
            trace_entry = f"[{_timestamp()}] ROUTER: Already have context, proceeding to DIAGNOSIS"
    elif uncertainty < THRESHOLD_CHIEF:
        # Check if we already have literature output
        if state.get("literature_output") is None:
            decision = "literature"
            trace_entry = f"[{_timestamp()}] ROUTER: U={uncertainty:.2%}, invoking LITERATURE"
        else:
            decision = "diagnose"
            trace_entry = f"[{_timestamp()}] ROUTER: Already have evidence, proceeding to DIAGNOSIS"
    else:
        decision = "chief"
        trace_entry = f"[{_timestamp()}] ROUTER: U={uncertainty:.2%} >= {THRESHOLD_CHIEF:.0%}, escalating to CHIEF"

    return {
        "routing_decision": decision,
        "steps_taken": steps + 1,
        "trace": [trace_entry],
    }


# =============================================================================
# FINALIZE NODE
# =============================================================================
def finalize_node(state: VerifaiState) -> dict:
    """
    Creates final diagnosis from radiologist output when no escalation needed.
    """
    rad_output = state.get("radiologist_output")
    historian_output = state.get("historian_output")
    literature_output = state.get("literature_output")
    uncertainty = state["current_uncertainty"]

    if rad_output and rad_output.differential:
        top_dx = rad_output.differential[0]
        base_confidence = top_dx.probability

        # Adjust based on gathered evidence
        if historian_output:
            base_confidence = min(0.95, base_confidence + historian_output.probability_adjustment * 0.5)
        if literature_output and literature_output.evidence_strength == "moderate":
            base_confidence = min(0.95, base_confidence + 0.05)

        diagnosis = FinalDiagnosis(
            diagnosis=top_dx.diagnosis,
            confidence=round(base_confidence, 2),
            deferred=False,
        )
    else:
        diagnosis = FinalDiagnosis(
            diagnosis=None,
            confidence=0.0,
            deferred=True,
            deferral_reason="No diagnostic findings available",
        )

    trace_entry = f"[{_timestamp()}] FINALIZE: Dx={diagnosis.diagnosis}, Confidence={diagnosis.confidence:.0%}"

    return {
        "final_diagnosis": diagnosis,
        "trace": [trace_entry],
    }
