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
            "dicom_metadata": None,  # Would parse DICOM header if applicable
            "radiologist_output": None,
            "critic_output": None,
            "historian_output": None,
            "literature_output": None,
            "debate_output": None,
            "current_uncertainty": 1.0,
            "routing_decision": "",
            "steps_taken": 0,
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
        "critic": None
    }
    
    # Visual evidence
    rad = state.get("radiologist_output")
    if rad:
        packet["visual"] = {
            "findings": [f.model_dump() for f in rad.findings],
            "hypotheses": [h.model_dump() for h in rad.hypotheses],
            "reasoning": rad.reasoning,
            "internal_signals": rad.internal_signals.model_dump()
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
            "overconfidence_probability": critic.overconfidence_probability,
            "counter_hypotheses": critic.counter_hypotheses,
            "concern_signals": critic.concern_signals,
            "calculated_uncertainty": critic.calculated_uncertainty
        }
    
    return packet


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
