"""
Automatic Mistake Detection

Compares initial diagnosis vs. final validated diagnosis to automatically detect errors.
Generates mistake entries for review and insertion into the past mistakes database.
"""

import logging
from typing import Optional, Dict, Tuple
from datetime import datetime

from graph.state import VerifaiState, RadiologistOutput, CriticOutput, FinalDiagnosis
from uncertainty.case_embedding import generate_case_embedding_from_fields

logger = logging.getLogger(__name__)


# =============================================================================
# SEVERITY SCORING
# =============================================================================

def calculate_severity_score(
    initial_diagnosis: str,
    final_diagnosis: str,
    kle_uncertainty: float,
    critic_safety_score: float,
    clinical_outcome: Optional[str] = None
) -> int:
    """
    Calculate error severity (1-5) based on diagnostic discrepancy and metrics.
    
    Severity levels:
    - 1: Minor misinterpretation, no clinical impact
    - 2: Small diagnostic error, minimal impact
    - 3: Moderate error, could affect treatment
    - 4: Significant error, likely clinical impact
    - 5: Critical error, severe consequences
    
    Args:
        initial_diagnosis: Original radiologist diagnosis
        final_diagnosis: Validated correct diagnosis
        kle_uncertainty: KLE epistemic uncertainty at time of diagnosis
        critic_safety_score: Critic safety score
        clinical_outcome: Optional clinical outcome description
    
    Returns:
        Severity score 1-5
    """
    severity = 1
    
    # High uncertainty + low safety = higher severity
    if kle_uncertainty > 0.5 and critic_safety_score < 0.4:
        severity = max(severity, 4)
    elif kle_uncertainty > 0.4 and critic_safety_score < 0.5:
        severity = max(severity, 3)
    
    # Check for critical keywords in diagnoses
    critical_keywords = ['pneumothorax', 'hemothorax', 'aortic', 'mass', 'malignancy', 'urgent', 'emergent']
    for keyword in critical_keywords:
        if keyword in final_diagnosis.lower() and keyword not in initial_diagnosis.lower():
            severity = max(severity, 4)  # Missed critical finding
    
    # Clinical outcome indicators
    if clinical_outcome:
        outcome_lower = clinical_outcome.lower()
        if any(word in outcome_lower for word in ['severe', 'critical', 'death', 'complication']):
            severity = 5
        elif any(word in outcome_lower for word in ['adverse', 'harm', 'delayed']):
            severity = max(severity, 4)
        elif 'improved' in outcome_lower or 'resolved' in outcome_lower:
            severity = min(severity, 2)
    
    # Cap at 5
    return min(severity, 5)


def classify_error_type(
    initial_diagnosis: str,
    final_diagnosis: str,
    kle_uncertainty: float,
    critic_was_overconfident: bool,
    linguistic_certainty: Optional[str] = None
) -> str:
    """
    Classify the type of diagnostic error.
    
    Error types:
    - overconfidence: High certainty language despite uncertainty
    - misdiagnosis: Wrong primary pathology identified
    - missed_differential: Failed to consider correct diagnosis
    - calibration_error: Uncertainty estimate was wrong
    
    Args:
        initial_diagnosis: Original diagnosis
        final_diagnosis: Validated diagnosis
        kle_uncertainty: Epistemic uncertainty
        critic_was_overconfident: Whether critic flagged overconfidence
        linguistic_certainty: Optional extracted certainty phrases
    
    Returns:
        Error type string
    """
    initial_lower = initial_diagnosis.lower()
    final_lower = final_diagnosis.lower()
    
    # Overconfidence: certain language despite high uncertainty
    if critic_was_overconfident and kle_uncertainty > 0.3:
        return "overconfidence"
    
    # Misdiagnosis: completely different pathology
    # Check if initial says "normal" but final has pathology
    if 'normal' in initial_lower and any(word in final_lower for word in [
        'pneumonia', 'effusion', 'edema', 'mass', 'fracture', 'consolidation'
    ]):
        return "misdiagnosis"
    
    # Missed differential: diagnosis was in the realm but not primary
    if 'consider' in initial_lower or 'possible' in initial_lower or 'differential' in initial_lower:
        return "missed_differential"
    
    # Calibration error: uncertainty score didn't reflect actual difficulty
    if abs(kle_uncertainty - 0.5) > 0.3:  # Very low or very high uncertainty but still wrong
        return "calibration_error"
    
    # Default to misdiagnosis
    return "misdiagnosis"


# =============================================================================
# MISTAKE DETECTION
# =============================================================================

def detect_mistake(
    workflow_state: VerifaiState,
    final_validated_diagnosis: str,
    clinical_outcome: Optional[str] = None
) -> Optional[Dict]:
    """
    Detect if a diagnostic mistake occurred by comparing initial vs. final diagnosis.
    
    Args:
        workflow_state: Complete workflow state from VERIFAI
        final_validated_diagnosis: Validated correct diagnosis (from clinician, pathology, etc.)
        clinical_outcome: Optional clinical outcome description
    
    Returns:
        Dict with mistake details if discrepancy detected, None otherwise
    """
    rad_output = workflow_state.get("radiologist_output")
    critic_output = workflow_state.get("critic_output")
    kle_uncertainty = workflow_state.get("current_uncertainty", workflow_state.get("radiologist_kle_uncertainty", 0.5))
    chexbert_output = workflow_state.get("chexbert_output")
    historian_output = workflow_state.get("historian_output")
    debate_output = workflow_state.get("debate_output")
    
    if not rad_output:
        logger.warning("[AUTO-DETECT] No radiologist output available")
        return None
    
    initial_diagnosis = rad_output.impression
    
    # Check if there's a significant discrepancy
    initial_lower = initial_diagnosis.lower()
    final_lower = final_validated_diagnosis.lower()
    
    # Simple discrepancy check (can be enhanced)
    if initial_lower == final_lower:
        logger.info("[AUTO-DETECT] No discrepancy detected (identical diagnoses)")
        return None
    
    # Check for "normal" vs pathology
    if 'normal' in initial_lower and 'normal' not in final_lower:
        discrepancy_detected = True
    # Check for key pathology mismatches
    elif any(word in initial_lower and word not in final_lower 
             for word in ['pneumonia', 'effusion', 'edema', 'atelectasis']):
        discrepancy_detected = True
    elif any(word in final_lower and word not in initial_lower
             for word in ['pneumonia', 'effusion', 'edema', 'atelectasis', 'consolidation']):
        discrepancy_detected = True
    else:
        # Conservative: assume no major discrepancy
        discrepancy_detected = False
    
    if not discrepancy_detected:
        logger.info("[AUTO-DETECT] No significant discrepancy detected")
        return None
    
    # Discrepancy detected - generate mistake details
    logger.info(f"[AUTO-DETECT] Discrepancy detected:\n  Initial: {initial_diagnosis}\n  Final: {final_validated_diagnosis}")
    
    # Calculate severity
    safety_score = critic_output.safety_score if critic_output else 0.5
    severity = calculate_severity_score(
        initial_diagnosis=initial_diagnosis,
        final_diagnosis=final_validated_diagnosis,
        kle_uncertainty=kle_uncertainty,
        critic_safety_score=safety_score,
        clinical_outcome=clinical_outcome
    )
    
    # Classify error type
    was_overconfident = critic_output.is_overconfident if critic_output else False
    error_type = classify_error_type(
        initial_diagnosis=initial_diagnosis,
        final_diagnosis=final_validated_diagnosis,
        kle_uncertainty=kle_uncertainty,
        critic_was_overconfident=was_overconfident
    )
    
    # Extract disease type from CheXbert or final diagnosis
    disease_type = "unknown"
    if chexbert_output and chexbert_output.labels:
        # Find first 'present' label
        present_labels = [k.lower() for k, v in chexbert_output.labels.items() if v == 'present']
        if present_labels:
            disease_type = present_labels[0]
    
    if disease_type == "unknown":
        # Extract from final diagnosis
        for disease in ['pneumonia', 'effusion', 'edema', 'atelectasis', 'cardiomegaly', 'pneumothorax']:
            if disease in final_lower:
                disease_type = disease
                break
    
    # Compile clinical context
    clinical_summary = None
    if historian_output:
        facts = historian_output.fhir_facts[:3] if historian_output.fhir_facts else []
        clinical_summary = "\n".join([f"- {fact.fact}" for fact in facts])
    
    # Compile debate summary
    debate_summary = None
    if debate_output:
        rounds = debate_output.rounds[:2] if debate_output.rounds else []
        debate_summary = "\n".join([f"Round {i+1}: {round.summary}" for i, round in enumerate(rounds)])
    
    mistake_details = {
        "session_id": workflow_state.get("trace", ["unknown"])[0] if workflow_state.get("trace") else "unknown",
        "image_path": workflow_state.get("image_path", "unknown"),
        "original_diagnosis": initial_diagnosis,
        "corrected_diagnosis": final_validated_diagnosis,
        "disease_type": disease_type,
        "error_type": error_type,
        "severity_level": severity,
        "kle_uncertainty": kle_uncertainty,
        "safety_score": safety_score,
        "chexbert_labels": chexbert_output.labels if chexbert_output else {},
        "clinical_summary": clinical_summary,
        "debate_summary": debate_summary,
        "clinical_outcome": clinical_outcome,
        "detected_at": datetime.now().isoformat()
    }
    
    return mistake_details


def generate_mistake_summary(mistake_details: Dict) -> str:
    """
    Generate a human-readable summary for clinician review.
    
    Args:
        mistake_details: Dict from detect_mistake()
    
    Returns:
        Formatted summary string
    """
    summary = f"""
**Potential Diagnostic Error Detected**

**Original Diagnosis:** {mistake_details['original_diagnosis']}
**Corrected Diagnosis:** {mistake_details['corrected_diagnosis']}

**Error Classification:**
- Type: {mistake_details['error_type']}
- Severity: {mistake_details['severity_level']}/5
- Disease Category: {mistake_details['disease_type']}

**Metrics at Time of Diagnosis:**
- KLE Uncertainty: {mistake_details['kle_uncertainty']:.2f}
- Safety Score: {mistake_details['safety_score']:.2f}

**Clinical Context:**
{mistake_details.get('clinical_summary', 'N/A')}

**Debate Summary:**
{mistake_details.get('debate_summary', 'N/A')}

---
Please review and confirm if this should be added to the past mistakes database.
"""
    return summary.strip()


# =============================================================================
# INTEGRATION HOOK
# =============================================================================

def on_clinical_validation(
    workflow_state: VerifaiState,
    validated_diagnosis: str,
    clinical_outcome: Optional[str] = None,
    auto_insert: bool = False
) -> Optional[str]:
    """
    Hook to call when clinical validation is received for a case.
    
    This should be integrated into the workflow completion flow or
    called by an external validation system.
    
    Args:
        workflow_state: Complete VERIFAI workflow state
        validated_diagnosis: Final validated diagnosis from clinician/pathology
        clinical_outcome: Optional clinical outcome description
        auto_insert: If True, automatically insert without confirmation (use with caution)
    
    Returns:
        Mistake ID if inserted, None otherwise
    """
    from db.past_mistakes import insert_validated_mistake
    
    # Detect mistake
    mistake = detect_mistake(workflow_state, validated_diagnosis, clinical_outcome)
    
    if not mistake:
        logger.info("[AUTO-DETECT] No mistake detected, skipping insertion")
        return None
    
    # Generate summary for logging
    summary = generate_mistake_summary(mistake)
    logger.info(f"[AUTO-DETECT] Mistake detected:\n{summary}")
    
    if auto_insert:
        # Generate embedding and insert
        embedding = generate_case_embedding_from_fields(
            disease_type=mistake['disease_type'],
            original_diagnosis=mistake['original_diagnosis'],
            corrected_diagnosis=mistake['corrected_diagnosis'],
            error_type=mistake['error_type'],
            kle_uncertainty=mistake['kle_uncertainty'],
            chexbert_labels=mistake['chexbert_labels'],
            clinical_summary=mistake['clinical_summary'],
            debate_summary=mistake['debate_summary']
        )
        
        mistake_id = insert_validated_mistake(
            session_id=mistake['session_id'],
            image_path=mistake['image_path'],
            original_diagnosis=mistake['original_diagnosis'],
            corrected_diagnosis=mistake['corrected_diagnosis'],
            disease_type=mistake['disease_type'],
            error_type=mistake['error_type'],
            severity_level=mistake['severity_level'],
            case_embedding=embedding,
            kle_uncertainty=mistake['kle_uncertainty'],
            safety_score=mistake['safety_score'],
            chexbert_labels=mistake['chexbert_labels'],
            clinical_summary=mistake['clinical_summary'],
            debate_summary=mistake['debate_summary']
        )
        
        logger.info(f"[AUTO-DETECT] Auto-inserted mistake {mistake_id}")
        return mistake_id
    else:
        logger.info("[AUTO-DETECT] Mistake detected but auto_insert=False, awaiting confirmation")
        # In a real system, this would trigger a UI prompt for clinician review
        return None
