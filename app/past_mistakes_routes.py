from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from db.past_mistakes import (
    insert_validated_mistake,
    retrieve_similar_mistakes,
    get_mistake_by_id,
    delete_mistake,
    get_statistics
)
from uncertainty.case_embedding import generate_case_embedding_from_fields



# REQUEST/RESPONSE MODELS
class ValidatedMistakeRequest(BaseModel):
    """Request model for inserting a validated mistake."""
    
    session_id: str = Field(..., description="Original workflow session ID")
    image_path: str = Field(..., description="Path to X-ray image")
    original_diagnosis: str = Field(..., description="Incorrect diagnosis that was made")
    corrected_diagnosis: str = Field(..., description="Validated correct diagnosis")
    disease_type: str = Field(..., description="Primary pathology category (e.g., pneumonia, effusion)")
    error_type: str = Field(..., description="Type of error: overconfidence, misdiagnosis, missed_differential, calibration_error")
    severity_level: int = Field(..., ge=1, le=5, description="Error severity from 1 (minor) to 5 (critical)")
    
    # Optional fields
    kle_uncertainty: Optional[float] = Field(None, ge=0.0, le=1.0, description="KLE uncertainty score at time of mistake")
    safety_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Safety score at time of mistake")
    chexbert_labels: Optional[Dict[str, str]] = Field(None, description="CheXpert labels dict")
    clinical_summary: Optional[str] = Field(None, description="Clinical context summary")
    debate_summary: Optional[str] = Field(None, description="Debate/reasoning summary")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "workflow-session-12345",
                "image_path": "patient_001_xray.jpg",
                "original_diagnosis": "Normal chest X-ray",
                "corrected_diagnosis": "Community-Acquired Pneumonia (RLL)",
                "disease_type": "pneumonia",
                "error_type": "misdiagnosis",
                "severity_level": 4,
                "kle_uncertainty": 0.35,
                "safety_score": 0.45,
                "chexbert_labels": {"Consolidation": "present"},
                "clinical_summary": "Patient with fever and productive cough",
                "debate_summary": "Radiologist missed RLL consolidation"
            }
        }


class ValidatedMistakeResponse(BaseModel):
    """Response model for mistake records."""
    
    mistake_id: str
    session_id: str
    image_path: str
    created_at: datetime
    original_diagnosis: str
    corrected_diagnosis: str
    disease_type: str
    error_type: str
    severity_level: int
    kle_uncertainty: Optional[float]
    safety_score: Optional[float]
    chexbert_labels: Dict[str, str]
    clinical_summary: Optional[str]
    debate_summary: Optional[str]
    
    # Optional similarity score (for retrieval results)
    similarity: Optional[float] = None


class MistakeInsertResponse(BaseModel):
    """Response after inserting a mistake."""
    
    mistake_id: str
    message: str = "Mistake successfully recorded"


class MistakeListRequest(BaseModel):
    """Query parameters for listing mistakes."""
    
    disease_type: Optional[str] = None
    error_type: Optional[str] = None
    severity_min: Optional[int] = Field(None, ge=1, le=5)
    severity_max: Optional[int] = Field(None, ge=1, le=5)
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)


class StatisticsResponse(BaseModel):
    """Response model for aggregate statistics."""
    
    total_mistakes: int
    by_disease_type: List[Dict[str, Any]]
    by_error_type: List[Dict[str, Any]]
    by_severity: List[Dict[str, Any]]


# API ROUTER
router = APIRouter(
    prefix="/api/past-mistakes",
    tags=["past-mistakes"],
    responses={404: {"description": "Not found"}}
)