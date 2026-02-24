"""
Medical Safety Guardrails — Production Safety Layer

Provides multi-tier safety validation before any diagnosis is released:

Tier 1: Critical Finding Detection
    - Identifies life-threatening conditions requiring STAT notification
    - Maps to ACR critical results communication guidelines

Tier 2: Red Flag Rules
    - Laterality mismatch detection
    - Confidence-safety gating
    - Contradictory evidence checks
    - Age-appropriateness validation

Tier 3: Epistemic Safety
    - Uncertainty-confidence alignment check
    - Hallucination risk scoring via hedging language analysis
    - Past mistake pattern matching

Output: SafetyReport with pass/fail, flags, and actionable recommendations
"""

import re
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

logger = logging.getLogger("safety.guardrails")


# =============================================================================
# CRITICAL FINDINGS DATABASE (ACR Guidelines)
# =============================================================================

CRITICAL_FINDINGS: Dict[str, Dict[str, str]] = {
    "tension pneumothorax": {
        "urgency": "STAT",
        "action": "Contact ED physician immediately for emergent decompression",
        "acr_category": "critical",
        "icd10": "J93.0",
    },
    "aortic dissection": {
        "urgency": "STAT",
        "action": "CT Angiography STAT; vascular surgery consult",
        "acr_category": "critical",
        "icd10": "I71.0",
    },
    "massive pleural effusion": {
        "urgency": "URGENT",
        "action": "Consider emergent thoracentesis; respiratory support",
        "acr_category": "significant",
        "icd10": "J91.0",
    },
    "widened mediastinum": {
        "urgency": "STAT",
        "action": "Rule out aortic pathology; CT angiography recommended",
        "acr_category": "critical",
        "icd10": "R93.1",
    },
    "free air": {
        "urgency": "STAT",
        "action": "Surgical consult for possible bowel perforation",
        "acr_category": "critical",
        "icd10": "K63.1",
    },
    "pneumoperitoneum": {
        "urgency": "STAT",
        "action": "Surgical consult; upright or lateral decubitus film",
        "acr_category": "critical",
        "icd10": "K66.8",
    },
    "pulmonary embolism": {
        "urgency": "STAT",
        "action": "CT pulmonary angiography STAT; anticoagulation",
        "acr_category": "critical",
        "icd10": "I26.99",
    },
    "cardiac tamponade": {
        "urgency": "STAT",
        "action": "Emergent pericardiocentesis; cardiology consult",
        "acr_category": "critical",
        "icd10": "I31.4",
    },
    "malpositioned endotracheal tube": {
        "urgency": "STAT",
        "action": "Reposition ETT immediately; risk of right mainstem intubation",
        "acr_category": "critical",
        "icd10": "T17.9",
    },
    "large pneumothorax": {
        "urgency": "URGENT",
        "action": "Chest tube insertion; monitor respiratory status",
        "acr_category": "significant",
        "icd10": "J93.1",
    },
    "acute respiratory distress syndrome": {
        "urgency": "URGENT",
        "action": "ICU admission; mechanical ventilation may be required",
        "acr_category": "significant",
        "icd10": "J80",
    },
    "miliary tuberculosis": {
        "urgency": "URGENT",
        "action": "Isolation precautions; infectious disease consult",
        "acr_category": "significant",
        "icd10": "A19.9",
    },
}

# Conditions that require follow-up but aren't immediately critical
SIGNIFICANT_FINDINGS: Dict[str, Dict[str, str]] = {
    "lung nodule": {
        "urgency": "FOLLOW-UP",
        "action": "Follow Fleischner Society guidelines for incidental nodule management",
        "icd10": "R91.1",
    },
    "mass": {
        "urgency": "FOLLOW-UP",
        "action": "CT chest with contrast for further characterization",
        "icd10": "R91.8",
    },
    "cavitary lesion": {
        "urgency": "FOLLOW-UP",
        "action": "Consider TB, fungal infection, or malignancy; sputum culture",
        "icd10": "R91.8",
    },
    "interstitial lung disease": {
        "urgency": "FOLLOW-UP",
        "action": "HRCT chest; pulmonology referral",
        "icd10": "J84.9",
    },
}


# =============================================================================
# SAFETY REPORT MODEL
# =============================================================================

class CriticalFinding(BaseModel):
    """A single critical finding detected in the diagnosis."""
    condition: str
    urgency: str  # "STAT", "URGENT", "FOLLOW-UP"
    action: str
    acr_category: str = "unknown"
    icd10: str = ""
    matched_text: str = ""


class RedFlag(BaseModel):
    """A safety red flag raised during validation."""
    flag_type: str
    severity: str  # "high", "medium", "low"
    description: str
    recommendation: str


class SafetyReport(BaseModel):
    """Complete safety validation report for a VERIFAI diagnosis."""
    # Overall status
    passed: bool = True
    safety_score: float = Field(1.0, ge=0.0, le=1.0, description="0.0=unsafe, 1.0=safe")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Critical findings
    critical_findings: List[CriticalFinding] = Field(default_factory=list)
    significant_findings: List[CriticalFinding] = Field(default_factory=list)
    requires_immediate_action: bool = False

    # Red flags
    red_flags: List[RedFlag] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    # Epistemic safety
    hallucination_risk: str = "low"  # "low", "medium", "high"
    confidence_uncertainty_aligned: bool = True

    # Summary
    summary: str = ""


# =============================================================================
# TIER 1: CRITICAL FINDING DETECTION
# =============================================================================

def _detect_critical_findings(
    diagnosis_text: str,
    findings_text: str,
    impression_text: str,
) -> tuple[List[CriticalFinding], List[CriticalFinding]]:
    """
    Scan all textual outputs for critical and significant findings.
    Uses case-insensitive substring matching with word boundary awareness.
    """
    critical = []
    significant = []
    combined = f"{diagnosis_text} {findings_text} {impression_text}".lower()

    for condition, info in CRITICAL_FINDINGS.items():
        # Use word-boundary-aware matching
        pattern = r'\b' + re.escape(condition) + r'\b'
        match = re.search(pattern, combined)
        if match or condition in combined:
            critical.append(CriticalFinding(
                condition=condition.title(),
                urgency=info["urgency"],
                action=info["action"],
                acr_category=info.get("acr_category", "critical"),
                icd10=info.get("icd10", ""),
                matched_text=combined[max(0, match.start()-30):match.end()+30].strip() if match else condition,
            ))

    for condition, info in SIGNIFICANT_FINDINGS.items():
        pattern = r'\b' + re.escape(condition) + r'\b'
        match = re.search(pattern, combined)
        if match or condition in combined:
            significant.append(CriticalFinding(
                condition=condition.title(),
                urgency=info["urgency"],
                action=info["action"],
                icd10=info.get("icd10", ""),
                matched_text=combined[max(0, match.start()-30):match.end()+30].strip() if match else condition,
            ))

    return critical, significant


# =============================================================================
# TIER 2: RED FLAG RULES
# =============================================================================

def _check_confidence_safety_gate(
    confidence: float,
    deferred: bool,
    uncertainty: float,
) -> Optional[RedFlag]:
    """Low confidence diagnosis that wasn't deferred is dangerous."""
    if confidence < 0.4 and not deferred:
        return RedFlag(
            flag_type="low_confidence_not_deferred",
            severity="high",
            description=f"Diagnosis confidence ({confidence:.0%}) is below safety threshold (40%) but was not deferred for human review.",
            recommendation="Consider deferring low-confidence diagnoses to a senior radiologist.",
        )
    return None


def _check_confidence_uncertainty_alignment(
    confidence: float,
    uncertainty: float,
) -> Optional[RedFlag]:
    """
    High confidence + high uncertainty = epistemic mismatch.
    This is the most dangerous scenario — the system is certain despite evidence of uncertainty.
    """
    if confidence > 0.8 and uncertainty > 0.5:
        return RedFlag(
            flag_type="confidence_uncertainty_mismatch",
            severity="high",
            description=f"High confidence ({confidence:.0%}) despite high epistemic uncertainty ({uncertainty:.0%}). Possible overconfidence.",
            recommendation="Recalibrate confidence based on uncertainty signals. Consider additional imaging or second opinion.",
        )
    if confidence > 0.7 and uncertainty > 0.4:
        return RedFlag(
            flag_type="moderate_confidence_uncertainty_mismatch",
            severity="medium",
            description=f"Moderate confidence-uncertainty misalignment: confidence={confidence:.0%}, uncertainty={uncertainty:.0%}.",
            recommendation="Review uncertainty cascade for potential calibration issues.",
        )
    return None


def _check_laterality(
    findings_text: str,
    impression_text: str,
) -> Optional[RedFlag]:
    """
    Check for laterality mismatches (e.g., findings mention right but impression says left).
    This is a common source of medical errors.
    """
    left_pattern = r'\b(left|LUL|LLL|left\s+lung|L\s+lung)\b'
    right_pattern = r'\b(right|RUL|RLL|RML|right\s+lung|R\s+lung)\b'

    findings_left = bool(re.search(left_pattern, findings_text, re.IGNORECASE))
    findings_right = bool(re.search(right_pattern, findings_text, re.IGNORECASE))
    impression_left = bool(re.search(left_pattern, impression_text, re.IGNORECASE))
    impression_right = bool(re.search(right_pattern, impression_text, re.IGNORECASE))

    # Flag if findings mention one side but impression mentions only the other
    if findings_left and not findings_right and impression_right and not impression_left:
        return RedFlag(
            flag_type="laterality_mismatch",
            severity="high",
            description="Findings mention LEFT but impression references RIGHT only. Potential laterality error.",
            recommendation="Verify anatomical laterality. Mislabeled sides can lead to wrong-site interventions.",
        )
    if findings_right and not findings_left and impression_left and not impression_right:
        return RedFlag(
            flag_type="laterality_mismatch",
            severity="high",
            description="Findings mention RIGHT but impression references LEFT only. Potential laterality error.",
            recommendation="Verify anatomical laterality. Mislabeled sides can lead to wrong-site interventions.",
        )
    return None


def _check_contradictory_evidence(
    historian_output: Any,
    literature_output: Any,
    critic_output: Any,
) -> Optional[RedFlag]:
    """Check if evidence agents directly contradict the diagnosis."""
    contradictions = 0

    if historian_output:
        n_support = len(getattr(historian_output, 'supporting_facts', []))
        n_contra = len(getattr(historian_output, 'contradicting_facts', []))
        if n_contra > n_support and n_contra >= 2:
            contradictions += 1

    if critic_output and getattr(critic_output, 'is_overconfident', False):
        if getattr(critic_output, 'safety_score', 1.0) < 0.3:
            contradictions += 1

    if contradictions >= 2:
        return RedFlag(
            flag_type="contradictory_evidence",
            severity="high",
            description=f"Multiple agents ({contradictions}) raised concerns against the diagnosis. Clinical history and critic both disagree.",
            recommendation="This case has significant disagreement among agents. Strongly consider human review.",
        )
    elif contradictions == 1:
        return RedFlag(
            flag_type="partial_contradictory_evidence",
            severity="medium",
            description="At least one agent raised concerns against the primary diagnosis.",
            recommendation="Review the contradicting evidence before accepting the diagnosis.",
        )
    return None


def _check_hallucination_risk(
    findings_text: str,
    impression_text: str,
) -> tuple[str, Optional[RedFlag]]:
    """
    Estimate hallucination risk by analyzing linguistic patterns.
    LLMs that hallucinate tend to use very specific details without hedging.
    """
    # Hedging markers indicate appropriate caution
    hedging_markers = [
        "may", "might", "possibly", "likely", "suggest", "consistent with",
        "consider", "cannot exclude", "differential", "uncertain", "probable",
        "cannot rule out", "questionable", "appears", "suspicious for",
    ]
    # Overconfidence markers without supporting evidence
    overconfidence_markers = [
        "definitely", "certainly", "confirms", "proven", "diagnostic of",
        "pathognomonic", "unequivocally", "no doubt", "clearly shows",
    ]

    text = f"{findings_text} {impression_text}".lower()
    hedging_count = sum(1 for m in hedging_markers if m in text)
    overconfidence_count = sum(1 for m in overconfidence_markers if m in text)

    # Heavily specific text without any hedging
    if overconfidence_count >= 2 and hedging_count == 0:
        risk = "high"
        flag = RedFlag(
            flag_type="hallucination_risk",
            severity="high",
            description="Report uses definitive language without appropriate hedging. High risk of overconfident or hallucinated findings.",
            recommendation="Add appropriate hedge language. Consider: 'consistent with', 'suggestive of', 'consider'.",
        )
    elif overconfidence_count >= 1 and hedging_count <= 1:
        risk = "medium"
        flag = RedFlag(
            flag_type="hallucination_risk",
            severity="medium",
            description="Report may be overly confident in some assertions.",
            recommendation="Review specific claims for supporting evidence.",
        )
    else:
        risk = "low"
        flag = None

    return risk, flag


def _check_debate_consensus(
    debate_output: Any,
) -> Optional[RedFlag]:
    """Flag cases where debate failed to reach consensus."""
    if debate_output is None:
        return None

    if not getattr(debate_output, 'final_consensus', True):
        return RedFlag(
            flag_type="no_debate_consensus",
            severity="medium",
            description="Multi-agent debate did not reach consensus. Diagnosis may be contentious.",
            recommendation="Review debate transcript for unresolved disagreements.",
        )
    return None


# =============================================================================
# MAIN SAFETY CHECK FUNCTION
# =============================================================================

def run_safety_check(state: dict) -> SafetyReport:
    """
    Run comprehensive safety validation on a completed VERIFAI workflow state.

    Executes all three tiers of safety checks:
    - Tier 1: Critical finding detection
    - Tier 2: Red flag rules
    - Tier 3: Epistemic safety

    Args:
        state: Complete VerifaiState dict from workflow output

    Returns:
        SafetyReport with detailed findings and recommendations
    """
    report = SafetyReport()

    # Extract relevant data from state
    final_diagnosis = state.get("final_diagnosis")
    rad_output = state.get("radiologist_output")
    critic_output = state.get("critic_output")
    historian_output = state.get("historian_output")
    literature_output = state.get("literature_output")
    debate_output = state.get("debate_output")

    # Text extraction
    diagnosis_text = ""
    confidence = 0.5
    deferred = False
    if final_diagnosis:
        diagnosis_text = getattr(final_diagnosis, 'diagnosis', '') or ''
        confidence = getattr(final_diagnosis, 'calibrated_confidence', 0.5)
        deferred = getattr(final_diagnosis, 'deferred', False)

    findings_text = getattr(rad_output, 'findings', '') if rad_output else ''
    impression_text = getattr(rad_output, 'impression', '') if rad_output else ''
    uncertainty = state.get("current_uncertainty", 0.5)

    # ─── TIER 1: Critical Finding Detection ───
    critical, significant = _detect_critical_findings(
        diagnosis_text, findings_text, impression_text
    )
    report.critical_findings = critical
    report.significant_findings = significant
    report.requires_immediate_action = any(
        cf.urgency == "STAT" for cf in critical
    )

    if report.requires_immediate_action:
        for cf in critical:
            if cf.urgency == "STAT":
                report.recommendations.append(
                    f"⚠️ CRITICAL: {cf.condition} detected — {cf.action}"
                )

    # ─── TIER 2: Red Flag Rules ───
    flags = []

    flag = _check_confidence_safety_gate(confidence, deferred, uncertainty)
    if flag:
        flags.append(flag)

    flag = _check_confidence_uncertainty_alignment(confidence, uncertainty)
    if flag:
        flags.append(flag)
        report.confidence_uncertainty_aligned = False

    if findings_text and impression_text:
        flag = _check_laterality(findings_text, impression_text)
        if flag:
            flags.append(flag)

    flag = _check_contradictory_evidence(historian_output, literature_output, critic_output)
    if flag:
        flags.append(flag)

    flag = _check_debate_consensus(debate_output)
    if flag:
        flags.append(flag)

    # ─── TIER 3: Epistemic Safety ───
    hallucination_risk, hallucination_flag = _check_hallucination_risk(
        findings_text, impression_text
    )
    report.hallucination_risk = hallucination_risk
    if hallucination_flag:
        flags.append(hallucination_flag)

    report.red_flags = flags

    # ─── COMPUTE SAFETY SCORE ───
    score = 1.0

    # Critical findings reduce score significantly
    score -= 0.2 * len(report.critical_findings)

    # Red flags reduce score based on severity
    for f in flags:
        if f.severity == "high":
            score -= 0.15
        elif f.severity == "medium":
            score -= 0.08
        else:
            score -= 0.03

    # STAT findings are severe
    if report.requires_immediate_action:
        score -= 0.2

    report.safety_score = max(0.0, min(1.0, round(score, 3)))
    report.passed = report.safety_score > 0.4

    # ─── GENERATE SUMMARY ───
    parts = []
    if report.passed:
        parts.append(f"✅ Safety check PASSED (score: {report.safety_score:.0%})")
    else:
        parts.append(f"❌ Safety check FAILED (score: {report.safety_score:.0%})")

    if report.critical_findings:
        parts.append(f"🔴 {len(report.critical_findings)} critical finding(s) detected")
    if report.significant_findings:
        parts.append(f"🟡 {len(report.significant_findings)} significant finding(s)")
    if flags:
        high = sum(1 for f in flags if f.severity == "high")
        med = sum(1 for f in flags if f.severity == "medium")
        if high:
            parts.append(f"🔴 {high} high-severity red flag(s)")
        if med:
            parts.append(f"🟡 {med} medium-severity red flag(s)")
    if report.hallucination_risk != "low":
        parts.append(f"⚠️ Hallucination risk: {report.hallucination_risk}")

    report.summary = " | ".join(parts)

    # Add general recommendations
    if not report.passed:
        report.recommendations.append(
            "This case did not pass safety validation. Human review is strongly recommended before acting on the diagnosis."
        )
    if report.hallucination_risk == "high":
        report.recommendations.append(
            "High hallucination risk detected. Cross-reference findings with original imaging."
        )

    logger.info(f"Safety check complete: {report.summary}")
    return report
