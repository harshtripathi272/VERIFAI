"""
VERIFAI FastAPI Backend

Provides REST API for running diagnostic workflow.
"""

import os
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.graph import verifai_graph
from app.state import VerifaiState


# --- Response Models ---

class DiagnosisResponse(BaseModel):
    """Response from /diagnose endpoint."""
    diagnosis: Optional[str]
    confidence: float
    deferred: bool
    deferral_reason: Optional[str]
    uncertainty: float
    trace: list[str]
    evidence_packet: dict[str, Any]


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str
    version: str


# --- App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown logic."""
    # Startup: could load models here
    print("VERIFAI API starting up...")
    yield
    # Shutdown: cleanup
    print("VERIFAI API shutting down...")


app = FastAPI(
    title="VERIFAI API",
    description="Clinically grounded, uncertainty-aware diagnostic AI for chest X-ray interpretation",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(
    image: UploadFile = File(...),
    patient_id: Optional[str] = None,
):
    """
    Run VERIFAI diagnostic workflow on a chest X-ray image.

    Args:
        image: The chest X-ray image file (DICOM, PNG, or JPEG)
        patient_id: Optional patient ID for FHIR context retrieval

    Returns:
        Complete diagnosis with evidence packet and audit trail
    """
    # Validate file type
    allowed_types = {"image/png", "image/jpeg", "application/dicom"}
    if image.content_type and not any(t in image.content_type for t in ["image", "dicom"]):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {image.content_type}")

    # Save uploaded file temporarily
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, image.filename or "upload.png")

    with open(file_path, "wb") as f:
        content = await image.read()
        f.write(content)

    try:
        # Initialize state
        initial_state: VerifaiState = {
            "image_path": file_path,
            "patient_id": patient_id,
            "radiologist_output": None,
            "critic_output": None,
            "historian_output": None,
            "literature_output": None,
            "current_uncertainty": 1.0,  # Start at maximum uncertainty
            "routing_decision": "",
            "steps_taken": 0,
            "final_diagnosis": None,
            "trace": [f"[INIT] Processing image: {image.filename}, Patient: {patient_id or 'N/A'}"],
        }

        # Run the graph
        final_state = verifai_graph.invoke(initial_state)

        # Build evidence packet
        evidence_packet = _build_evidence_packet(final_state)

        # Extract final diagnosis
        final_dx = final_state.get("final_diagnosis")
        if final_dx is None:
            raise HTTPException(status_code=500, detail="Workflow completed without diagnosis")

        return DiagnosisResponse(
            diagnosis=final_dx.diagnosis,
            confidence=final_dx.confidence,
            deferred=final_dx.deferred,
            deferral_reason=final_dx.deferral_reason,
            uncertainty=final_state.get("current_uncertainty", 0.0),
            trace=final_state.get("trace", []),
            evidence_packet=evidence_packet,
        )

    finally:
        # Cleanup uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


def _build_evidence_packet(state: VerifaiState) -> dict[str, Any]:
    """
    Constructs the verifiable evidence packet from workflow state.
    """
    packet = {
        "visual_evidence": {},
        "clinical_context": {},
        "literature_support": {},
        "reasoning_trace": [],
    }

    # Visual evidence (from radiologist)
    rad_output = state.get("radiologist_output")
    if rad_output:
        packet["visual_evidence"] = {
            "findings": [f.model_dump() for f in rad_output.findings],
            "differential": [d.model_dump() for d in rad_output.differential],
            "reasoning": rad_output.reasoning,
            # In production: include Grad-CAM heatmap path
            "saliency_map": None,
        }

    # Clinical context (from historian)
    hist_output = state.get("historian_output")
    if hist_output:
        packet["clinical_context"] = {
            "conditions": hist_output.relevant_conditions,
            "risk_factors": hist_output.risk_factors,
            "labs": hist_output.relevant_labs,
            "prior_imaging": hist_output.prior_imaging_comparison,
            "summary": hist_output.clinical_summary,
        }

    # Literature support
    lit_output = state.get("literature_output")
    if lit_output:
        packet["literature_support"] = {
            "supporting": [c.model_dump() for c in lit_output.supporting_evidence],
            "contradicting": [c.model_dump() for c in lit_output.contradicting_evidence],
            "evidence_strength": lit_output.evidence_strength,
        }

    # Reasoning trace
    packet["reasoning_trace"] = state.get("trace", [])

    return packet


# --- Main ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
