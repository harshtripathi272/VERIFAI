"""
VERIFAI Graph State

Shared TypedDict and Pydantic models for inter-agent communication.
"""

from typing import TypedDict, Optional, Annotated, Any, List
from pydantic import BaseModel, Field



# REDUCER FUNCTIONS


def append_trace(left: list[str], right: list[str]) -> list[str]:
    """Reducer to accumulate audit trail entries."""
    if not isinstance(left, list):
        left = [left] if left else []
    if not isinstance(right, list):
        right = [right] if right else []
    return left + right



# DOMAIN MODELS (Pydantic for validation)


class VisualFinding(BaseModel):
    """A single visual finding from radiologist."""
    location: str = Field(..., description="Anatomical location (e.g., RLL, LUL, Mediastinum)")
    observation: str = Field(..., description="What is observed (e.g., opacity, nodule, effusion)")
    severity: float = Field(..., ge=0.0, le=1.0, description="Severity score 0-1")
    bounding_box: Optional[List[float]] = Field(None, description="Optional [x, y, w, h] normalized coords")


class RadiologistOutput(BaseModel):
    """Plain-text output from Radiologist Agent.
    
    Contains narrative FINDINGS and IMPRESSION sections.
    Also includes disease probabilities and paths to interpretability heatmaps.
    Epistemic uncertainty is computed externally via KLE.
    """
    findings: str = Field(..., description="Textual FINDINGS section based on visual evidence")
    impression: str = Field(..., description="Textual IMPRESSION section with diagnostic interpretation")
    
    # New fields for Disease Classification & Interpretability
    disease_probabilities: dict[str, float] = Field(default_factory=dict, description="Probabilities for 14 CheXbert diseases")
    heatmap_paths: dict[str, str] = Field(default_factory=dict, description="Paths to saved heatmap images for positive detections")
    # Grad-CAM visualization fields (optional, for validation)
    gradcam_heatmap_b64: Optional[str] = Field(None, description="Base64-encoded Grad-CAM heatmap overlay")
    gradcam_peak_bbox: Optional[List[int]] = Field(None, description="Bounding box [x1, y1, x2, y2] of activation peak")
    gradcam_activation_mass: Optional[float] = Field(None, description="Diffuseness score 0.0-1.0 (higher = more diffuse)")
    gradcam_anatomical_region: Optional[str] = Field(None, description="Anatomical region of peak activation (e.g., 'right_lower_lobe')")


class CheXbertOutput(BaseModel):
    """Output from CheXbert labeling of radiologist report.
    
    Contains ONLY labels marked as 'present' or 'uncertain'.
    Absent/not_mentioned conditions are not stored.
    """
    labels: dict[str, str] = Field(..., description="CheXpert conditions with present or uncertain status only")


class CriticOutput(BaseModel):
    """Output from Critic Agent.
    
    Evaluates consistency between linguistic certainty in the IMPRESSION
    and the externally computed epistemic uncertainty score (KLE).
    
    Also checks for similarity to past validated mistakes.
    """
    is_overconfident: bool = Field(..., description="True if text is overly assertive given uncertainty")
    concern_flags: List[str] = Field(default_factory=list, description="Specific consistency issues detected")
    recommended_hedging: Optional[str] = Field(None, description="Suggested rephrasing to match uncertainty")
    safety_score: float = Field(..., ge=0.0, le=1.0, description="Overall safety/appropriateness score")
    
    # Historical mistake signals
    similar_mistakes_count: int = Field(default=0, description="Number of similar past errors found")
    historical_risk_level: str = Field(default="none", description="Risk level based on past mistakes: none/low/medium/high")


class HistorianFact(BaseModel):
    """A fact retrieved from FHIR."""
    fact_type: str  # "supporting" or "contradicting"
    description: str
    fhir_resource_id: str
    fhir_resource_type: str


class HistorianOutput(BaseModel):
    """Output from Historian Agent (FHIR context)."""
    supporting_facts: List[HistorianFact] = Field(default_factory=list)
    contradicting_facts: List[HistorianFact] = Field(default_factory=list)
    confidence_adjustment: float = Field(0.0, description="Numeric adjustment to radiologist confidence")
    clinical_summary: str = ""


class LiteratureCitation(BaseModel):
    """A literature citation."""
    pmid: str
    title: str
    authors: List[str] = Field(default_factory=list)
    journal: str = ""
    year: Optional[int] = None
    relevance_summary: str = ""
    evidence_strength: str = Field("low", description="low/medium/high")
    source: str = "pubmed"  # pubmed, europepmc, semanticscholar


class LiteratureOutput(BaseModel):
    """Output from Literature Agent (RAG over PubMed/PMC)."""
    citations: List[LiteratureCitation] = Field(default_factory=list)
    overall_evidence_strength: str = "low"


class FinalDiagnosis(BaseModel):
    """Final calibrated diagnosis or deferral."""
    diagnosis: Optional[str] = None
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)
    deferred: bool = False
    deferral_reason: Optional[str] = None
    recommended_next_steps: List[str] = Field(default_factory=list)
    explanation: str = ""



# DEBATE MODELS


class DebateArgument(BaseModel):
    """A single argument in the debate."""
    agent: str  # "critic", "historian", "literature"
    position: str  # "challenge", "support", "refine"
    argument: str
    confidence_impact: float = Field(0.0, description="How this affects confidence (-1 to +1)")
    evidence_refs: List[str] = Field(default_factory=list)


class DebateRound(BaseModel):
    """A single round of debate."""
    round_number: int
    critic_challenge: Optional[DebateArgument] = None
    historian_response: Optional[DebateArgument] = None
    literature_response: Optional[DebateArgument] = None
    round_consensus: Optional[str] = None
    confidence_delta: float = 0.0


class DebateOutput(BaseModel):
    """Final output from debate process."""
    rounds: List[DebateRound] = Field(default_factory=list)
    final_consensus: bool = False
    consensus_diagnosis: Optional[str] = None
    consensus_confidence: float = 0.0
    escalate_to_chief: bool = False
    escalation_reason: Optional[str] = None
    debate_summary: str = ""
    total_confidence_adjustment: float = 0.0


# DOCTOR FEEDBACK MODELS


class DoctorFeedback(BaseModel):
    """Doctor feedback for diagnosis review and reprocessing.
    
    Captured when a doctor rejects/corrects a diagnosis.
    Used to restart workflow from critic with doctor's context.
    """
    feedback_id: int = Field(..., description="Database ID of feedback record")
    original_session_id: str = Field(..., description="Session ID of original workflow that was rejected")
    feedback_type: str = Field(..., description="Type: 'rejection', 'correction', or 'approval'")
    doctor_notes: str = Field(..., description="Doctor's explanation of what's wrong")
    correct_diagnosis: Optional[str] = Field(None, description="What doctor believes is correct")
    rejection_reasons: List[str] = Field(default_factory=list, description="Categories of issues found")






# LANGGRAPH STATE


class VerifaiState(TypedDict):
    """
    Shared state passed between all nodes in the VERIFAI graph.
    
    The `trace` field uses a reducer to accumulate entries from each node,
    building a complete audit trail.
    """
    # === Session Tracking ===
    _session_id: Optional[str]  # DB logging session ID (auto-generated if not provided)
    
    # === Input ===
    image_path: str
    patient_id: Optional[str]
    dicom_metadata: Optional[dict[str, Any]]
    view : Optional[str]
    
    # === Agent Outputs ===
    radiologist_output: Optional[RadiologistOutput]
    chexbert_output: Optional[CheXbertOutput]  # NEW: Structured pathology labels
    critic_output: Optional[CriticOutput]
    historian_output: Optional[HistorianOutput]
    literature_output: Optional[LiteratureOutput]
    debate_output: Optional[DebateOutput]  # NEW: Debate results
    validator_output: Optional[dict]  # NEW: Validator tools output (when debate fails)
    
    # === Routing Control ===
    current_uncertainty: float
    routing_decision: str
    steps_taken: int
    
    # === KLE Uncertainty (for logging/analysis) ===
    radiologist_kle_uncertainty: Optional[float]  # Early epistemic instability score
    
    # === Final Result ===
    final_diagnosis: Optional[FinalDiagnosis]
    
    # === Doctor Feedback (NEW) ===
    doctor_feedback: Optional[DoctorFeedback]  # Present when reprocessing with doctor input
    is_feedback_iteration: bool  # True if this is a reprocessing run after feedback
    
    # === Audit Trail ===
    trace: Annotated[List[str], append_trace]
