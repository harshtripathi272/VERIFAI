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

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
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

# NEW: Safety Guardrails, Evidence Report, and Monitoring
from safety.guardrails import run_safety_check, SafetyReport
from utils.evidence_report import generate_evidence_report, save_evidence_report
from monitoring.metrics import metrics, track_agent_execution, track_diagnosis, get_metrics_summary
from monitoring.metrics import structured_logger

from fastapi.responses import PlainTextResponse, HTMLResponse

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
    current_state: dict[str, Any] | None = None
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
    # Literature
    lit = state.get("literature_output")
    if lit:
        if isinstance(lit, str):
            packet["literature"] = {
                "citations": [],
                "overall_strength": lit
            }
        else:
            packet["literature"] = {
                "citations": [c.model_dump() for c in getattr(lit, "citations", [])],
                "overall_strength": getattr(lit, "overall_evidence_strength", "")
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

def _run_workflow_background(file_paths: List[str], views: List[str], patient_id: str, fhir_content: Optional[str], session_id: str):
    """Background task to execute the graph."""
    metrics.start_workflow(session_id)
    try:
        initial_state: VerifaiState = {
            "_session_id": session_id,
            "image_paths": file_paths,
            "patient_id": patient_id,
            "current_fhir": fhir_content,
            "dicom_metadata": None,
            "views": views,
            "radiologist_output": None,
            "chexbert_output": None,
            "critic_output": None,
            "historian_output": None,
            "literature_output": None,
            "debate_output": None,
            "validator_output": None,
            "current_uncertainty": 1.0,
            "routing_decision": "",
            "steps_taken": 0,
            "radiologist_kle_uncertainty": None,
            "final_diagnosis": None,
            "trace": [f"[INIT] Processing async, Patient: {patient_id or 'N/A'}"],
            "is_feedback_iteration": False,
            "doctor_feedback": None,
            "uncertainty_history": []
        }
        
        # Thread config for memory checkpointer
        config = {"configurable": {"thread_id": session_id}}
        
        # This will run until it hits the `interrupt()` in `human_review_node`,
        # at which point it suspends and saves state to MemorySaver
        graph_app.invoke(initial_state, config=config)
        
    except Exception as e:
        print(f"[BACKGROUND] Workflow {session_id} failed: {e}")
    finally:
        # We deliberately DO NOT delete file_path here because the workflow is suspended.
        # It must be deleted when the graph reaches END.
        metrics.end_workflow(session_id)


@router.post("/workflows/start", response_model=WorkflowStartResponse)
async def start_workflow(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    views: List[str] = Form(..., description="List of view names (e.g., AP, PA, LATERAL)"),
    patient_id: Optional[str] = Form(None),
    fhir_report: Optional[UploadFile] = File(None)
):
    """
    Start the diagnostic workflow asynchronously.
    Returns immediately with a session_id you can poll.
    """
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    if len(images) != len(views):
        raise HTTPException(status_code=400, detail="Number of images must match number of views")
    
    os.makedirs("uploads", exist_ok=True)
    file_paths = []
    
    for img in images:
        file_path = f"uploads/{uuid.uuid4()}_{img.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(img.file, f)
        file_paths.append(file_path)
        
    fhir_content = None
    if fhir_report:
        try:
            import json
            content_bytes = await fhir_report.read()
            fhir_content = json.loads(content_bytes.decode("utf-8"))
        except Exception as e:
            print(f"[API] Error reading FHIR report: {e}")
            fhir_content = None
    
    session_id = str(uuid.uuid4())
    
    # Register SSE session for live streaming
    try:
        from app.streaming import register_session
        register_session(session_id)
    except Exception:
        pass
    
    # Launch in background
    background_tasks.add_task(_run_workflow_background, file_paths, views, patient_id, fhir_content, session_id)
    
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
    
    # Extract current state for live mirroring on frontend
    state_values = state_snapshot.values
    
    def try_model_dump(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return obj

    current_state = {
        "image_path": state_values.get("image_path"),
        "radiologist": try_model_dump(state_values.get("radiologist_output")),
        "chexbert": try_model_dump(state_values.get("chexbert_output")),
        "historian": try_model_dump(state_values.get("historian_output")),
        "literature": try_model_dump(state_values.get("literature_output")),
        "critic": try_model_dump(state_values.get("critic_output")),
        "debate": try_model_dump(state_values.get("debate_output")),
        "routing": state_values.get("routing_decision"),
        "trace": state_values.get("trace", []),
        "uncertainty_history": state_values.get("uncertainty_history", [])
    }

    # If the graph is not running and has next tasks, it's either suspended or interrupted
    if not state_snapshot.next:
        # No more nodes to run -> It is COMPLETED
        final_state = state_snapshot.values
        final_dx = final_state.get("final_diagnosis")
        evidence = _build_evidence_packet(final_state)
        
        return WorkflowStatusResponse(
            session_id=session_id,
            status="completed",
            current_state=current_state,
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
            current_state=current_state,
            pending_review_data=pending_data
        )
        
    # Otherwise, it's just actively running in the background thread
    return WorkflowStatusResponse(session_id=session_id, status="running", current_state=current_state)


@router.post("/workflows/{session_id}/resume")
async def resume_workflow(session_id: str, req: HumanReviewRequest, background_tasks: BackgroundTasks):
    """
    Provide human feedback to a suspended workflow.
    Executes the rest of the workflow in the background to prevent server deadlock.
    """
    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = graph_app.get_state(config)
    
    if not state_snapshot or not state_snapshot.next:
        raise HTTPException(status_code=400, detail="Workflow not found or not suspended.")
        
    # Enqueue background execution instead of blocking the event loop
    background_tasks.add_task(_resume_workflow_background, session_id, req, config, state_snapshot)
    
    return {"session_id": session_id, "status": "resumed", "action": req.action}


def _resume_workflow_background(session_id: str, req: HumanReviewRequest, config: dict, state_snapshot: Any):
    """Background task to run the rest of the workflow without blocking the API."""
    # Store mistake in database if rejected
    if req.action == "reject" and not getattr(settings, "MOCK_MODELS", False):
        try:
             current_state = state_snapshot.values
             
             image_paths = current_state.get("image_paths", [])
             image_path = image_paths[0] if image_paths else "unknown"
             
             final_diagnosis_obj = current_state.get("final_diagnosis")
             original_diagnosis = getattr(final_diagnosis_obj, "diagnosis", "Unknown") if hasattr(final_diagnosis_obj, "diagnosis") else "Unknown"
             
             chexbert_obj = current_state.get("chexbert_output")
             chexbert_labels = {}
             if chexbert_obj and hasattr(chexbert_obj, "labels"):
                  chexbert_labels = chexbert_obj.labels
             elif isinstance(chexbert_obj, dict):
                  chexbert_labels = chexbert_obj
             disease_type = "unknown"
             if chexbert_labels:
                  present = [k for k,v in chexbert_labels.items() if v in ["present", "uncertain"]]
                  if present:
                       disease_type = present[0].lower()
             if disease_type == "unknown":
                  disease_type = original_diagnosis.split()[0].lower()
                  
             kle = current_state.get("current_uncertainty", 0.5)
             
             critic = current_state.get("critic_output", {})
             safety_score = getattr(critic, "safety_score", 0.5) if hasattr(critic, "safety_score") else 0.5
             
             historian = current_state.get("historian_output", {})
             clinical_summary = getattr(historian, "clinical_summary", "") if hasattr(historian, "clinical_summary") else ""
             
             debate = current_state.get("debate_output", {})
             debate_summary = getattr(debate, "debate_summary", "") if hasattr(debate, "debate_summary") else ""
             
             embedding = generate_case_embedding_from_fields(
                 disease_type=disease_type,
                 original_diagnosis=original_diagnosis,
                 corrected_diagnosis=req.correct_diagnosis,
                 error_type="misdiagnosis",
                 uncertainty_score=kle,
                 chexbert_labels=chexbert_labels,
                 clinical_summary=clinical_summary,
                 debate_summary=debate_summary
             )
             
             insert_validated_mistake(
                 session_id=session_id,
                 image_path=image_path,
                 original_diagnosis=original_diagnosis,
                 corrected_diagnosis=req.correct_diagnosis,
                 disease_type=disease_type,
                 error_type="misdiagnosis",
                 severity_level=3,
                 case_embedding=embedding,
                 kle_uncertainty=kle,
                 safety_score=safety_score,
                 chexbert_labels=chexbert_labels,
                 clinical_summary=clinical_summary,
                 debate_summary=debate_summary
             )
             print(f"[VERIFAI] Successfully recorded rejected diagnosis for {session_id} to Past Mistakes DB.")
        except Exception as e:
             print(f"[VERIFAI] Warning: Failed to record mistake to DB: {e}")

    try:
        payload = {
             "action": req.action,
             "feedback": req.feedback,
             "correct_diagnosis": req.correct_diagnosis
        }

        # Ensure SSE queue exists for this session before agents start emitting
        # (the old queue may have been cleaned up after the first run)
        from app.streaming import register_session as _register_sse, emit_agent_event as _emit_sse, _event_queues
        if session_id not in _event_queues:
            _register_sse(session_id)

        # Resume the workflow using the Command primitive
        # Run synchronously since we are now in a BackgroundTask Thread
        for _ in graph_app.stream(Command(resume=payload), config, stream_mode="values"):
             pass

        # Check if the rerun has suspended again (reject looping to another review)
        # or if it has completed (approve ending the workflow)
        new_state_check = graph_app.get_state(config)
        if new_state_check and new_state_check.next:
            # Workflow suspended again (waiting for next human review)
            # Emit a signal so the frontend SSE knows to stop showing the feed
            _emit_sse(session_id, "system", "workflow_complete",
                      {"suspended": True}, "Workflow suspended for human review")
        else:
            # Workflow fully ended (approved)
            _emit_sse(session_id, "system", "workflow_complete",
                      {"approved": True}, "Workflow completed — diagnosis approved")
             
        # Extract the final completed state to populate Observability Metrics
        new_state = graph_app.get_state(config)
        if new_state and new_state.values:
             final_vals = new_state.values
             final_dx = final_vals.get("final_diagnosis")
             if final_dx:
                  confidence = getattr(final_dx, "calibrated_confidence", 0.0)
                  deferred = getattr(final_dx, "deferred", False)
                  uncertainty = max(0.01, 1.0 - confidence)
                  debate = final_vals.get("debate_output")
                  debate_rounds = len(debate.rounds) if debate and hasattr(debate, "rounds") else 0
                  
                  critic = final_vals.get("critic_output")
                  safety_score = getattr(critic, "safety_score", 1.0) if hasattr(critic, "safety_score") else 1.0
                  
                  track_diagnosis(
                      confidence=confidence,
                      uncertainty=uncertainty,
                      deferred=deferred,
                      debate_rounds=debate_rounds,
                      safety_score=safety_score
                  )
                  print(f"[VERIFAI] Successfully recorded diagnosis metrics for {session_id}.")
                  
        # Mark workflow ended in global metrics to trigger observability flush    
        metrics.end_workflow(session_id)
             
    except Exception as e:
        structured_logger.log("error", "Background workflow block failed.",
                              session_id=session_id, error=str(e))


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


# =============================================================================
# SAFETY GUARDRAILS ENDPOINTS
# =============================================================================

@router.get("/workflows/{session_id}/safety")
async def get_safety_report(session_id: str):
    """
    Run safety guardrails check on a completed/suspended workflow.
    Returns critical findings, red flags, and safety score.
    """
    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = graph_app.get_state(config)
    
    if not state_snapshot or not state_snapshot.created_at:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = state_snapshot.values
    
    try:
        safety_report = run_safety_check(state)
        
        # Track safety metrics
        metrics.safety_score.observe(safety_report.safety_score)
        for flag in safety_report.red_flags:
            metrics.safety_flags.labels(
                flag_type=flag.flag_type, severity=flag.severity
            ).inc()
        if safety_report.critical_findings:
            metrics.critical_findings.inc(len(safety_report.critical_findings))
        
        structured_logger.log("info", "Safety check completed",
                             session_id=session_id,
                             passed=safety_report.passed,
                             safety_score=safety_report.safety_score)
        
        return safety_report.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety check failed: {e}")


# =============================================================================
# EVIDENCE REPORT ENDPOINTS
# =============================================================================

@router.get("/workflows/{session_id}/report", response_class=HTMLResponse)
async def get_evidence_report(session_id: str):
    """
    Generate and return an interactive HTML evidence report.
    Self-contained HTML with embedded CSS — can be saved as a standalone file.
    """
    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = graph_app.get_state(config)
    
    if not state_snapshot or not state_snapshot.created_at:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = state_snapshot.values
    
    try:
        html = generate_evidence_report(state, session_id)
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


@router.post("/workflows/{session_id}/report/save")
async def save_report(session_id: str):
    """
    Generate and save evidence report to disk.
    Returns the file path of the saved HTML report.
    """
    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = graph_app.get_state(config)
    
    if not state_snapshot or not state_snapshot.created_at:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = state_snapshot.values
    
    try:
        filepath = save_evidence_report(state, session_id)
        return {"session_id": session_id, "report_path": filepath, "status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report save failed: {e}")


# =============================================================================
# OBSERVABILITY / METRICS ENDPOINTS
# =============================================================================

@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """
    Prometheus-compatible metrics endpoint.
    Scrape this endpoint with Prometheus to monitor VERIFAI in production.
    """
    return PlainTextResponse(
        content=metrics.to_prometheus_format(),
        media_type="text/plain; charset=utf-8"
    )


@router.get("/metrics/summary")
async def metrics_summary():
    """
    JSON summary of all collected metrics.
    Includes agent performance, diagnostic quality, safety stats, and system health.
    """
    return get_metrics_summary()

