"""
VERIFAI Graph State

Shared TypedDict and Pydantic models for inter-agent communication.
"""

from typing import TypedDict, Optional, Annotated, Any
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
    bounding_box: list[float] | None = Field(None, description="Optional [x, y, w, h] normalized coords")


class RadiologistOutput(BaseModel):
    """Plain-text output from Radiologist Agent.
    
    Contains only narrative FINDINGS and IMPRESSION sections.
    No confidence scores, probabilities, or structured outputs.
    Epistemic uncertainty is computed externally via KLE.
    """
    findings: str = Field(..., description="Textual FINDINGS section based on visual evidence")
    impression: str = Field(..., description="Textual IMPRESSION section with diagnostic interpretation")


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
    """
    is_overconfident: bool = Field(..., description="True if text is overly assertive given uncertainty")
    concern_flags: list[str] = Field(default_factory=list, description="Specific consistency issues detected")
    recommended_hedging: str | None = Field(None, description="Suggested rephrasing to match uncertainty")
    safety_score: float = Field(..., ge=0.0, le=1.0, description="Overall safety/appropriateness score")


class HistorianFact(BaseModel):
    """A fact retrieved from FHIR."""
    fact_type: str  # "supporting" or "contradicting"
    description: str
    fhir_resource_id: str
    fhir_resource_type: str


class HistorianOutput(BaseModel):
    """Output from Historian Agent (FHIR context)."""
    supporting_facts: list[HistorianFact] = Field(default_factory=list)
    contradicting_facts: list[HistorianFact] = Field(default_factory=list)
    confidence_adjustment: float = Field(0.0, description="Numeric adjustment to radiologist confidence")
    clinical_summary: str = ""


class LiteratureCitation(BaseModel):
    """A literature citation."""
    pmid: str
    title: str
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int | None = None
    relevance_summary: str = ""
    evidence_strength: str = Field("low", description="low/medium/high")
    source: str = "pubmed"  # pubmed, europepmc, semanticscholar


class LiteratureOutput(BaseModel):
    """Output from Literature Agent (RAG over PubMed/PMC)."""
    citations: list[LiteratureCitation] = Field(default_factory=list)
    overall_evidence_strength: str = "low"


class FinalDiagnosis(BaseModel):
    """Final calibrated diagnosis or deferral."""
    diagnosis: str | None = None
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)
    deferred: bool = False
    deferral_reason: str | None = None
    recommended_next_steps: list[str] = Field(default_factory=list)
    explanation: str = ""



# DEBATE MODELS


class DebateArgument(BaseModel):
    """A single argument in the debate."""
    agent: str  # "critic", "historian", "literature"
    position: str  # "challenge", "support", "refine"
    argument: str
    confidence_impact: float = Field(0.0, description="How this affects confidence (-1 to +1)")
    evidence_refs: list[str] = Field(default_factory=list)


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
    rounds: list[DebateRound] = Field(default_factory=list)
    final_consensus: bool = False
    consensus_diagnosis: Optional[str] = None
    consensus_confidence: float = 0.0
    escalate_to_chief: bool = False
    escalation_reason: Optional[str] = None
    debate_summary: str = ""
    total_confidence_adjustment: float = 0.0



# LANGGRAPH STATE


class VerifaiState(TypedDict):
    """
    Shared state passed between all nodes in the VERIFAI graph.
    
    The `trace` field uses a reducer to accumulate entries from each node,
    building a complete audit trail.
    """
    # === Input ===
    image_path: str
    patient_id: str | None
    dicom_metadata: dict[str, Any] | None
    view : str | None
    
    # === Agent Outputs ===
    radiologist_output: RadiologistOutput | None
    chexbert_output: CheXbertOutput | None  # NEW: Structured pathology labels
    critic_output: CriticOutput | None
    historian_output: HistorianOutput | None
    literature_output: LiteratureOutput | None
    debate_output: DebateOutput | None  # NEW: Debate results
    
    # === Routing Control ===
    current_uncertainty: float
    routing_decision: str
    steps_taken: int
    
    # === KLE Uncertainty (for logging/analysis) ===
    radiologist_kle_uncertainty: float | None  # Early epistemic instability score
    
    # === Final Result ===
    final_diagnosis: FinalDiagnosis | None
    
    # === Audit Trail ===
    trace: Annotated[list[str], append_trace]
