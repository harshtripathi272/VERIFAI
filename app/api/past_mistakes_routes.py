"""
Past Mistakes API Routes

REST API endpoints for CRUD operations on validated diagnostic errors.
Enables admin/clinician interface for managing the past mistakes memory database.
"""

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


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

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


# =============================================================================
# API ROUTER
# =============================================================================

router = APIRouter(
    prefix="/api/past-mistakes",
    tags=["past-mistakes"],
    responses={404: {"description": "Not found"}}
)


@router.post("", response_model=MistakeInsertResponse, status_code=status.HTTP_201_CREATED)
async def create_mistake(request: ValidatedMistakeRequest):
    """
    Insert a validated diagnostic mistake into the database.
    
    **Requirements:**
    - `original_diagnosis` and `corrected_diagnosis` must be non-empty
    - `error_type` must be one of: overconfidence, misdiagnosis, missed_differential, calibration_error
    - `severity_level` must be between 1 and 5
    - Case embedding is automatically generated
    
    **Returns:**
    - Unique mistake ID
    """
    try:
        # Generate case embedding
        embedding = generate_case_embedding_from_fields(
            disease_type=request.disease_type,
            original_diagnosis=request.original_diagnosis,
            corrected_diagnosis=request.corrected_diagnosis,
            error_type=request.error_type,
            kle_uncertainty=request.kle_uncertainty,
            chexbert_labels=request.chexbert_labels,
            clinical_summary=request.clinical_summary,
            debate_summary=request.debate_summary
        )
        
        # Insert into database
        mistake_id = insert_validated_mistake(
            session_id=request.session_id,
            image_path=request.image_path,
            original_diagnosis=request.original_diagnosis,
            corrected_diagnosis=request.corrected_diagnosis,
            disease_type=request.disease_type,
            error_type=request.error_type,
            severity_level=request.severity_level,
            case_embedding=embedding,
            kle_uncertainty=request.kle_uncertainty,
            safety_score=request.safety_score,
            chexbert_labels=request.chexbert_labels,
            clinical_summary=request.clinical_summary,
            debate_summary=request.debate_summary
        )
        
        return MistakeInsertResponse(mistake_id=mistake_id)
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to insert mistake: {str(e)}")


@router.get("/{mistake_id}", response_model=ValidatedMistakeResponse)
async def get_mistake(mistake_id: str):
    """
    Retrieve a single mistake by its ID.
    
    **Returns:**
    - Full mistake record
    
    **Raises:**
    - 404 if mistake not found
    """
    mistake = get_mistake_by_id(mistake_id)
    
    if not mistake:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Mistake {mistake_id} not found")
    
    return ValidatedMistakeResponse(**mistake)


@router.get("", response_model=List[ValidatedMistakeResponse])
async def list_mistakes(
    disease_type: Optional[str] = Query(None, description="Filter by disease type"),
    error_type: Optional[str] = Query(None, description="Filter by error type"),
    severity_min: Optional[int] = Query(None, ge=1, le=5, description="Minimum severity level"),
    severity_max: Optional[int] = Query(None, ge=1, le=5, description="Maximum severity level"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    List mistakes with optional filtering.
    
    **Query Parameters:**
    - `disease_type`: Filter by disease category (e.g., pneumonia, effusion)
    - `error_type`: Filter by error classification
    - `severity_min` / `severity_max`: Filter by severity range
    - `limit` / `offset`: Pagination
    
    **Returns:**
    - List of mistake records (max 500)
    """
    from db.past_mistakes import get_connection
    
    # Build SQL query with filters
    where_clauses = []
    params = []
    
    if disease_type:
        where_clauses.append("disease_type = ?")
        params.append(disease_type)
    
    if error_type:
        where_clauses.append("error_type = ?")
        params.append(error_type)
    
    if severity_min:
        where_clauses.append("severity_level >= ?")
        params.append(severity_min)
    
    if severity_max:
        where_clauses.append("severity_level <= ?")
        params.append(severity_max)
    
    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    query = f"""
        SELECT 
            mistake_id, session_id, image_path, created_at,
            original_diagnosis, corrected_diagnosis, disease_type,
            error_type, severity_level,
            kle_uncertainty, safety_score,
            chexbert_labels, clinical_summary, debate_summary
        FROM past_mistakes
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """
    
    params.extend([limit, offset])
    
    conn = get_connection()
    results = conn.execute(query, params).fetchall()
    
    import json
    
    mistakes = []
    for row in results:
        mistake = {
            'mistake_id': row[0],
            'session_id': row[1],
            'image_path': row[2],
            'created_at': row[3],
            'original_diagnosis': row[4],
            'corrected_diagnosis': row[5],
            'disease_type': row[6],
            'error_type': row[7],
            'severity_level': row[8],
            'kle_uncertainty': row[9],
            'safety_score': row[10],
            'chexbert_labels': json.loads(row[11]) if row[11] else {},
            'clinical_summary': row[12],
            'debate_summary': row[13]
        }
        mistakes.append(ValidatedMistakeResponse(**mistake))
    
    return mistakes


@router.delete("/{mistake_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_mistake(mistake_id: str):
    """
    Delete a mistake from the database.
    
    Use this to remove incorrectly validated errors or outdated entries.
    
    **Returns:**
    - 204 No Content on success
    - 404 if mistake not found
    """
    deleted = delete_mistake(mistake_id)
    
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Mistake {mistake_id} not found")
    
    return None


@router.get("/stats/summary", response_model=StatisticsResponse)
async def get_stats():
    """
    Get aggregate statistics about past mistakes.
    
    **Returns:**
    - Total mistake count
    - Breakdown by disease type (with average severity)
    - Breakdown by error type (with average severity)
    - Breakdown by severity level
    """
    stats = get_statistics()
    return StatisticsResponse(**stats)
