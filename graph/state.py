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


def rolling_uncertainty_history(left: list, right: list) -> list:
    """Reducer: merges and keeps only last 2 system uncertainty entries.
    Each entry: {"agent": str, "system_uncertainty": float}
    """
    combined = (left if isinstance(left, list) else []) + (right if isinstance(right, list) else [])
    return combined[-2:]  # Always keep max 2 most recent


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
    
    # Structured contextual information about top matched past mistakes
    historical_context: List[dict] = Field(
        default_factory=list,
        description=(
            "Top matched past mistakes (up to 3), each containing: "
            "disease_type, error_type, severity_level, kle_uncertainty, "
            "clinical_summary, similarity"
        )
    )

from typing import Literal

class HistorianFact(BaseModel):
    fact_type: Literal["supporting", "contradicting"]
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
    url: str = ""


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
    reproducibility_hash: Optional[str] = Field(
        None,
        description="SHA-256 fingerprint of inputs (image + patient context + config). FDA 21 CFR Part 11 audit trail."
    )

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
    image_paths: List[str]
    patient_id: Optional[str]
    dicom_metadata: Optional[dict[str, Any]]
    views: List[str]
    current_fhir: Optional[dict[str, Any]]
    
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
    
    # === Uncertainty Tracking ===
    # current_uncertainty is the authoritative MUC system value (updated at each node)
    # radiologist_kle_uncertainty is a legacy alias kept solely for DB logger
    # compatibility (column name in radiologist_logs / critic_logs is unchanged)
    radiologist_kle_uncertainty: Optional[float]  # Legacy alias — DB compat only
    # Rolling window of last 2 system uncertainty values (agent name + value)
    # Used by LLM agent prompts for spike detection and by the UI for the entropy graph
    uncertainty_history: Annotated[List[dict], rolling_uncertainty_history]
    
    # === Final Result ===
    final_diagnosis: Optional[FinalDiagnosis]
    
    # === Doctor Feedback (NEW) ===
    doctor_feedback: Optional[DoctorFeedback]  # Present when reprocessing with doctor input
    is_feedback_iteration: bool  # True if this is a reprocessing run after feedback
    
    # === Audit Trail ===
    trace: Annotated[List[str], append_trace]
