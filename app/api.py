"""
VERIFAI API Endpoints

/diagnose - Run diagnostic workflow
/health - Health check
/tools - List available MCP tools
/logs/* - Query agent logs, sessions, debates, and stats
"""

import os
import shutil
import uuid
from typing import Optional, Any

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from app.config import settings
from graph.workflow import app as graph_app
from graph.state import VerifaiState
from tools.registry import registry
from db.logger import AgentLogger
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status, BackgroundTasks
from pydantic import BaseModel, Field

from db.past_mistakes import (
    insert_validated_mistake,
    retrieve_similar_mistakes,
    get_mistake_by_id,
    delete_mistake,
    get_statistics
)
from uncertainty.case_embedding import generate_case_embedding_from_fields
from langgraph.types import Command

router = APIRouter()



# RESPONSE MODELS
class HealthResponse(BaseModel):
    status: str
    version: str
    mock_mode: bool


class DiagnosisResponse(BaseModel):
    diagnosis: str | None
    confidence: float
    deferred: bool
    deferral_reason: str | None
    recommended_next_steps: list[str]
    explanation: str
    uncertainty: float
    trace: list[str]
    evidence_packet: dict[str, Any]


class WorkflowStartResponse(BaseModel):
    session_id: str
    status: str
    message: str


class WorkflowStatusResponse(BaseModel):
    session_id: str
    status: str  # "running", "suspended", "completed", "failed", "not_found"
    pending_review_data: dict[str, Any] | None = None
    final_result: dict[str, Any] | None = None


class HumanReviewRequest(BaseModel):
    action: str  # "approve" or "reject"
    feedback: str = ""
    correct_diagnosis: str | None = None


class ToolsResponse(BaseModel):
    tools: list[dict]
    total: int



# ENDPOINTS
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        mock_mode=settings.MOCK_MODELS
    )


@router.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """List available MCP tools."""
    tools = registry.list_tools()
    return ToolsResponse(tools=tools, total=len(tools))


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    image: UploadFile = File(...),
    patient_id: Optional[str] = None
):
    """
    Run VERIFAI diagnostic workflow on a chest X-ray.
    
    Args:
        image: Chest X-ray image file (PNG, JPEG, or DICOM)
        patient_id: Optional patient ID for FHIR context retrieval
        
    Returns:
        Complete diagnosis with evidence packet and audit trail
    """
    # Validate file
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{image.filename}"
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    
    # Generate session ID for DB logging
    session_id = str(uuid.uuid4())
    
    try:
        # Initialize state
        initial_state: VerifaiState = {
            "_session_id": session_id,
            "image_path": file_path,
            "patient_id": patient_id,
            "dicom_metadata": None,
            "view": None,
            "radiologist_output": None,
            "critic_output": None,
            "historian_output": None,
            "literature_output": None,
            "debate_output": None,
            "current_uncertainty": 1.0,
            "routing_decision": "",
            "steps_taken": 0,
            "radiologist_kle_uncertainty": None,
            "final_diagnosis": None,
            "trace": [f"[INIT] Processing {image.filename}, Patient: {patient_id or 'N/A'}"]
        }
        
        # Run graph
        result = graph_app.invoke(initial_state)
        
        # Build evidence packet
        evidence_packet = _build_evidence_packet(result)
        
        # Extract final diagnosis
        final_dx = result.get("final_diagnosis")
        if not final_dx:
            raise HTTPException(status_code=500, detail="Workflow completed without diagnosis")
        
        return DiagnosisResponse(
            diagnosis=final_dx.diagnosis,
            confidence=final_dx.calibrated_confidence,
            deferred=final_dx.deferred,
            deferral_reason=final_dx.deferral_reason,
            recommended_next_steps=final_dx.recommended_next_steps,
            explanation=final_dx.explanation,
            uncertainty=result.get("current_uncertainty", 0.0),
            trace=result.get("trace", []),
            evidence_packet=evidence_packet
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)


def _build_evidence_packet(state: VerifaiState) -> dict[str, Any]:
    """Build structured evidence packet from workflow state."""
    packet = {
        "visual": None,
        "clinical": None,
        "literature": None,
        "critic": None,
        "debate": None
    }
    
    # Visual evidence (Radiologist)
    rad = state.get("radiologist_output")
    if rad:
        packet["visual"] = {
            "findings": rad.findings,      # Now a plain string
            "impression": rad.impression   # Now a plain string
        }
    
    # Clinical context
    hist = state.get("historian_output")
    if hist:
        packet["clinical"] = {
            "supporting_facts": [f.model_dump() for f in hist.supporting_facts],
            "contradicting_facts": [f.model_dump() for f in hist.contradicting_facts],
            "confidence_adjustment": hist.confidence_adjustment,
            "summary": hist.clinical_summary
        }
    
    # Literature
    lit = state.get("literature_output")
    if lit:
        packet["literature"] = {
            "citations": [c.model_dump() for c in lit.citations],
            "overall_strength": lit.overall_evidence_strength
        }
    
    # Critic assessment
    critic = state.get("critic_output")
    if critic:
        packet["critic"] = {
            "is_overconfident": critic.is_overconfident,
            "concern_flags": critic.concern_flags,
            "recommended_hedging": critic.recommended_hedging,
            "safety_score": critic.safety_score
        }
    
    # Debate history
    debate = state.get("debate_output")
    if debate:
        packet["debate"] = {
            "rounds": [round.model_dump() for round in debate.rounds],
            "final_consensus": debate.final_consensus,
            "debate_summary": debate.debate_summary,
            "total_confidence_adjustment": debate.total_confidence_adjustment
        }
    
    return packet


# =============================================================================
# WORKFLOW EXECUTION ENDPOINTS (ASYNC / BACKGROUND)
# =============================================================================

def _run_workflow_background(file_path: str, patient_id: str, fhir_content: Optional[str], session_id: str):
    """Background task to execute the graph."""
    try:
        initial_state: VerifaiState = {
            "_session_id": session_id,
            "image_path": file_path,
            "patient_id": patient_id,
            "fhir_context": fhir_content,
            "dicom_metadata": None,
            "view": None,
            "radiologist_output": None,
            "critic_output": None,
            "historian_output": None,
            "literature_output": None,
            "debate_output": None,
            "current_uncertainty": 1.0,
            "routing_decision": "",
            "steps_taken": 0,
            "radiologist_kle_uncertainty": None,
            "final_diagnosis": None,
            "trace": [f"[INIT] Processing async, Patient: {patient_id or 'N/A'}"],
            "is_feedback_iteration": False
        }
        
        # Thread config for memory checkpointer
        config = {"configurable": {"thread_id": session_id}}
        
        # This will run until it hits the `interrupt()` in `human_review_node`,
        # at which point it suspends and saves state to MemorySaver
        graph_app.invoke(initial_state, config=config)
        
    except Exception as e:
        print(f"[BACKGROUND] Workflow {session_id} failed: {e}")
    # We deliberately DO NOT delete file_path here because the workflow is suspended.
    # It must be deleted when the graph reaches END.


@router.post("/workflows/start", response_model=WorkflowStartResponse)
async def start_workflow(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    patient_id: Optional[str] = None,
    fhir_report: Optional[UploadFile] = File(None)
):
    """
    Start the diagnostic workflow asynchronously.
    Returns immediately with a session_id you can poll.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{image.filename}"
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
        
    fhir_content = None
    if fhir_report:
        try:
            content_bytes = await fhir_report.read()
            fhir_content = content_bytes.decode("utf-8")
        except Exception as e:
            print(f"[API] Error reading FHIR report: {e}")
            fhir_content = None
    
    session_id = str(uuid.uuid4())
    
    # Launch in background
    background_tasks.add_task(_run_workflow_background, file_path, patient_id, fhir_content, session_id)
    
    return WorkflowStartResponse(
        session_id=session_id,
        status="running",
        message="Workflow initialized and running in background."
    )


@router.get("/workflows/{session_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(session_id: str):
    """
    Poll the LangGraph Checkpointer to see the current status of the workflow thread.
    """
    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = graph_app.get_state(config)
    
    if not state_snapshot or not state_snapshot.created_at:
        return WorkflowStatusResponse(session_id=session_id, status="not_found")
    
    # If the graph is not running and has next tasks, it's either suspended or interrupted
    if not state_snapshot.next:
        # No more nodes to run -> It is COMPLETED
        final_state = state_snapshot.values
        final_dx = final_state.get("final_diagnosis")
        evidence = _build_evidence_packet(final_state)
        
        return WorkflowStatusResponse(
            session_id=session_id,
            status="completed",
            final_result={
                "diagnosis": getattr(final_dx, "diagnosis", None),
                "confidence": getattr(final_dx, "calibrated_confidence", 0.0),
                "evidence_packet": evidence,
                "trace": final_state.get("trace", [])
            }
        )
    
    # It has next nodes. Check if it's currently interrupted by the `human_review_node`
    # In LangGraph 0.2+, `tasks` contains the `interrupts`
    interrupts = []
    if hasattr(state_snapshot, "tasks") and state_snapshot.tasks:
        for task in state_snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                 interrupts.extend(task.interrupts)
                 
    if interrupts:
        # It's waiting on a human!
        # The interrupt payload is inside `interrupt.value`
        pending_data = interrupts[0].value
        return WorkflowStatusResponse(
            session_id=session_id,
            status="suspended",
            pending_review_data=pending_data
        )
        
    # Otherwise, it's just actively running in the background thread
    return WorkflowStatusResponse(session_id=session_id, status="running")


@router.post("/workflows/{session_id}/resume")
async def resume_workflow(session_id: str, req: HumanReviewRequest):
    """
    Provide human feedback to a suspended workflow.
    """
    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = graph_app.get_state(config)
    
    if not state_snapshot or not state_snapshot.next:
        raise HTTPException(status_code=400, detail="Workflow not found or not suspended.")
        
    payload = {
         "action": req.action,
         "feedback": req.feedback,
         "correct_diagnosis": req.correct_diagnosis
    }
    
    try:
        # Resume the workflow using the Command primitive
        # This streams the remaining execution synchronously. In true production,
        # you might invoke this in a background task again, but since it's just
        # returning back to Critic or END, we can run it here
        for _ in graph_app.stream(Command(resume=payload), config, stream_mode="values"):
             pass
             
        return {"session_id": session_id, "status": "resumed", "action": req.action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resume graph: {e}")


# =============================================================================
# LOG QUERY ENDPOINTS
# =============================================================================

@router.get("/logs/sessions")
async def list_sessions(
    limit: int = Query(50, ge=1, le=500),
    status: Optional[str] = Query(None, description="Filter by status: running, completed, failed"),
    patient_id: Optional[str] = Query(None, description="Filter by patient ID")
):
    """List all workflow sessions with optional filters."""
    try:
        sessions = AgentLogger.list_sessions(limit=limit, status=status, patient_id=patient_id)
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/sessions/{session_id}")
async def get_session_detail(session_id: str):
    """Get full details of a workflow session including all agent logs and debate rounds."""
    try:
        summary = AgentLogger.get_session_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/agents/{agent_name}")
async def get_agent_logs(
    agent_name: str,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get invocation history for a specific agent (radiologist, critic, historian, literature, debate, chief)."""
    valid_agents = {"radiologist", "critic", "historian", "literature", "debate", "chief", "finalize", "evidence_gathering"}
    if agent_name not in valid_agents:
        raise HTTPException(status_code=400, detail=f"Invalid agent name. Choose from: {valid_agents}")
    try:
        history = AgentLogger.get_agent_history(agent_name, limit=limit)
        return {"agent": agent_name, "invocations": history, "total": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/debates")
async def list_debates(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(50, ge=1, le=500)
):
    """Get debate logs with full round-by-round details and arguments."""
    try:
        debates = AgentLogger.get_debate_history(session_id=session_id, limit=limit)
        return {"debates": debates, "total": len(debates)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/stats")
async def get_stats():
    """Get aggregate diagnosis statistics: totals, averages, top diagnoses, debate consensus rate."""
    try:
        stats = AgentLogger.get_diagnosis_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
