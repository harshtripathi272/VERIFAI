"""
VERIFAI API Endpoints

/diagnose - Run diagnostic workflow
/health - Health check
/tools - List available MCP tools
"""

import os
import shutil
from typing import Optional, Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.config import settings
from graph.workflow import app as graph_app
from graph.state import VerifaiState
from tools.registry import registry


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
    
    try:
        # Initialize state
        initial_state: VerifaiState = {
            "image_path": file_path,
            "patient_id": patient_id,
            "dicom_metadata": None,  # Would parse DICOM header if applicable
            "radiologist_output": None,
            "critic_output": None,
            "historian_output": None,
            "literature_output": None,
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
