"""
Doctor Feedback Agent

Handles doctor rejection/correction of diagnoses and triggers reprocessing.

When a doctor rejects a diagnosis:
1. Capture the feedback (what's wrong, correct diagnosis)
2. Store the full context (all agent outputs at rejection point)
3. Restart workflow from CRITIC with doctor's input injected
4. Link reprocessing result back to original feedback

FLOW:
Doctor Rejects → Feedback Captured → Restart from Critic → 
Evidence + Debate → New Diagnosis → Track Improvement
"""

import uuid
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from graph.state import VerifaiState, CriticOutput, DoctorFeedback
from db.adapter import get_logger


class FeedbackReprocessingInput(BaseModel):
    """Input for reprocessing after doctor feedback."""
    feedback_id: int
    original_session_id: str
    doctor_notes: str
    correct_diagnosis: Optional[str] = None
    rejection_reasons: list[str] = Field(default_factory=list)
    
    # Context from original workflow (preserved for critic)
    radiologist_output: Any = None
    chexbert_output: Any = None
    historian_output: Any = None
    literature_output: Any = None
    uncertainty: float = 0.5


def capture_doctor_feedback(
    session_id: str,
    feedback_type: str,
    doctor_notes: str,
    correct_diagnosis: str = None,
    rejection_reasons: list = None,
    doctor_id: str = None
) -> int:
    """
    Capture doctor feedback for a completed workflow session.
    
    Args:
        session_id: The session that is being reviewed
        feedback_type: 'rejection', 'correction', or 'approval'
        doctor_notes: Doctor's explanation of what's wrong
        correct_diagnosis: What doctor believes is correct (optional)
        rejection_reasons: List of issue categories
        doctor_id: Optional identifier for the doctor
    
    Returns:
        feedback_id for tracking
    """
    from db.adapter import get_logger
    
    # Get full context snapshot from the original session
    context_snapshot = get_logger.get_session_summary(session_id)
    
    # Create logger to record feedback
    logger = get_logger(session_id=session_id)
    
    feedback_id = logger.log_doctor_feedback(
        feedback_type=feedback_type,
        doctor_notes=doctor_notes,
        correct_diagnosis=correct_diagnosis,
        rejection_reasons=rejection_reasons or [],
        context_snapshot=context_snapshot,
        doctor_id=doctor_id
    )
    
    print(f"[Feedback] Captured feedback {feedback_id} for session {session_id}")
    print(f"[Feedback] Type: {feedback_type}")
    print(f"[Feedback] Doctor notes: {doctor_notes[:100]}...")
    
    return feedback_id


def prepare_feedback_for_reprocessing(feedback_id: int) -> FeedbackReprocessingInput:
    """
    Load feedback and prepare context for reprocessing.
    
    Retrieves:
    - Original diagnosis that was rejected
    - Doctor's notes and corrections
    - Full agent outputs from original workflow
    
    Args:
        feedback_id: The feedback record to reprocess
    
    Returns:
        FeedbackReprocessingInput with all necessary context
    """
    from db.adapter import get_logger
    
    # Get feedback with full context
    feedback_data = get_logger.get_feedback_for_reprocessing(feedback_id)
    
    if not feedback_data:
        raise ValueError(f"Feedback {feedback_id} not found")
    
    feedback = feedback_data['feedback']
    context = feedback_data['original_context']
    
    # Extract agent outputs from original session
    radiologist_output = None
    chexbert_output = None
    historian_output = None
    literature_output = None
    uncertainty = 0.5
    
    for invocation in context.get('invocations', []):
        agent = invocation['agent_name']
        output = invocation.get('output_summary')
        
        if agent == 'radiologist':
            radiologist_output = output
            # Get uncertainty score from DB log (stored under kle_uncertainty column)
            uncertainty = invocation.get('kle_uncertainty', 0.5)
        elif agent == 'chexbert':
            chexbert_output = output
        elif agent == 'historian':
            historian_output = output
        elif agent == 'literature':
            literature_output = output
    
    return FeedbackReprocessingInput(
        feedback_id=feedback_id,
        original_session_id=feedback['session_id'],
        doctor_notes=feedback['doctor_notes'],
        correct_diagnosis=feedback.get('correct_diagnosis'),
        rejection_reasons=feedback.get('rejection_reason', []),
        radiologist_output=radiologist_output,
        chexbert_output=chexbert_output,
        historian_output=historian_output,
        literature_output=literature_output,
        uncertainty=uncertainty
    )


def create_feedback_enhanced_state(
    feedback_input: FeedbackReprocessingInput,
    image_path: str,
    patient_id: str = None
) -> VerifaiState:
    """
    Create a new workflow state pre-populated with original context
    plus doctor feedback for reprocessing.
    
    This allows the workflow to restart from CRITIC with:
    - All original agent outputs preserved
    - Doctor's feedback injected as additional context
    - Clear flag that this is a feedback iteration
    
    Args:
        feedback_input: Prepared feedback with context
        image_path: Path to X-ray image
        patient_id: Optional patient ID
    
    Returns:
        VerifaiState ready for reprocessing
    """
    # Create new session ID for reprocessing
    new_session_id = str(uuid.uuid4())
    
    # Build doctor feedback object for state
    doctor_feedback = DoctorFeedback(
        feedback_id=feedback_input.feedback_id,
        original_session_id=feedback_input.original_session_id,
        feedback_type='rejection',
        doctor_notes=feedback_input.doctor_notes,
        correct_diagnosis=feedback_input.correct_diagnosis,
        rejection_reasons=feedback_input.rejection_reasons
    )
    
    # Create state with preserved context
    state = VerifaiState(
        _session_id=new_session_id,
        image_path=image_path,
        patient_id=patient_id,
        
        # Preserved outputs from original workflow
        radiologist_output=feedback_input.radiologist_output,
        chexbert_output=feedback_input.chexbert_output,
        historian_output=feedback_input.historian_output,
        literature_output=feedback_input.literature_output,
        radiologist_kle_uncertainty=feedback_input.uncertainty,
        
        # NEW: Doctor feedback context
        doctor_feedback=doctor_feedback,
        is_feedback_iteration=True,
        
        # Initialize routing
        routing_decision=None,
        current_uncertainty=feedback_input.uncertainty,
        trace=[
            f"FEEDBACK_REPROCESS: Starting reprocessing for feedback {feedback_input.feedback_id}",
            f"FEEDBACK_REPROCESS: Original session: {feedback_input.original_session_id}",
            f"FEEDBACK_REPROCESS: Doctor notes: {feedback_input.doctor_notes[:100]}...",
            f"FEEDBACK_REPROCESS: Restarting from CRITIC with enriched context"
        ]
    )
    
    return state


def link_feedback_reprocessing_result(
    feedback_id: int,
    new_session_id: str,
    final_diagnosis: str,
    final_confidence: float
):
    """
    Link the reprocessing result back to the original feedback.
    
    This creates an audit trail showing:
    - Original diagnosis (rejected)
    - Doctor feedback
    - New diagnosis after reprocessing
    
    Args:
        feedback_id: Original feedback ID
        new_session_id: Session ID of reprocessing workflow
        final_diagnosis: New diagnosis after reprocessing
        final_confidence: Confidence of new diagnosis
    """
    logger = get_logger()
    logger.link_feedback_reprocessing(
        feedback_id=feedback_id,
        new_session_id=new_session_id,
        final_result=final_diagnosis,
        final_confidence=final_confidence
    )
    
    print(f"[Feedback] Linked reprocessing result to feedback {feedback_id}")
    print(f"[Feedback] New session: {new_session_id}")
    print(f"[Feedback] New diagnosis: {final_diagnosis}")
    print(f"[Feedback] Confidence: {final_confidence:.2%}")


def feedback_node(state: VerifaiState) -> dict:
    """
    Node for processing doctor feedback within workflow.
    
    This is used when feedback is provided during the workflow
    (not just at the end). It injects the feedback context into
    the critic evaluation.
    
    Returns updated state with feedback context.
    """
    doctor_feedback = state.get("doctor_feedback")
    
    if not doctor_feedback:
        return {
            "trace": ["FEEDBACK: No doctor feedback present"]
        }
    
    trace_entries = [
        f"FEEDBACK: Processing doctor feedback {doctor_feedback.feedback_id}",
        f"FEEDBACK: Type - {doctor_feedback.feedback_type}",
        f"FEEDBACK: Notes - {doctor_feedback.doctor_notes[:100]}..."
    ]
    
    if doctor_feedback.correct_diagnosis:
        trace_entries.append(
            f"FEEDBACK: Doctor's correct diagnosis - {doctor_feedback.correct_diagnosis}"
        )
    
    # Flag that this is a feedback iteration
    return {
        "is_feedback_iteration": True,
        "trace": trace_entries
    }


# =============================================================================
# STATE UPDATES FOR DOCTOR FEEDBACK SUPPORT
# =============================================================================

# The following additions should be made to graph/state.py:

"""
Add to VerifaiState TypedDict:

class DoctorFeedback(BaseModel):
    '''Doctor feedback for diagnosis review.'''
    feedback_id: int
    original_session_id: str
    feedback_type: str  # 'rejection', 'correction', 'approval'
    doctor_notes: str
    correct_diagnosis: Optional[str] = None
    rejection_reasons: list[str] = Field(default_factory=list)


# Add to VerifaiState:
class VerifaiState(TypedDict, total=False):
    # ... existing fields ...
    
    # NEW: Doctor feedback support
    doctor_feedback: Optional[DoctorFeedback]
    is_feedback_iteration: bool
"""
