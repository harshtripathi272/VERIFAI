"""
VERIFAI State Definitions

Defines the LangGraph state and all Pydantic models for agent communication.
"""

from typing import TypedDict, Optional, Annotated
from pydantic import BaseModel, Field
import operator


# --- Pydantic Models for Structured Agent Outputs ---

class Finding(BaseModel):
    """A single visual finding from the radiologist."""
    location: str = Field(..., description="Anatomical location (e.g., 'RLL', 'LUL')")
    observation: str = Field(..., description="What is observed (e.g., 'opacity', 'nodule')")
    severity: float = Field(..., ge=0.0, le=1.0, description="Severity score 0-1")


class DiagnosisCandidate(BaseModel):
    """A single diagnostic hypothesis."""
    diagnosis: str
    probability: float = Field(..., ge=0.0, le=1.0)


class RawUncertaintySignals(BaseModel):
    """Raw signals from the radiologist used by the Critic."""
    logit_margin: float = Field(..., description="Diff between top 2 logits")
    entropy: float = Field(..., description="Prediction entropy")
    attention_dispersion: float = Field(..., description="Gini coeff of attention weights")
    prediction_stability: float = Field(..., description="Std across dropout runs")


class RadiologistOutput(BaseModel):
    """Structured output from the Radiologist Agent."""
    findings: list[Finding] = []
    differential: list[DiagnosisCandidate] = []
    raw_signals: RawUncertaintySignals
    reasoning: str = ""


class CriticOutput(BaseModel):
    """Output from the Critic Agent."""
    overconfidence_score: float = Field(..., ge=0.0, le=1.0, description="P(radiologist is overconfident)")
    critiques: list[str] = []
    calculated_uncertainty: float = Field(..., ge=0.0, le=1.0, description="Final combined uncertainty")


class HistorianOutput(BaseModel):
    """Output from the Historian Agent (FHIR context)."""
    relevant_conditions: list[str] = []
    risk_factors: list[str] = []
    relevant_labs: dict[str, float] = {}
    prior_imaging_comparison: str = ""
    clinical_summary: str = ""
    probability_adjustment: float = 0.0  # Can be negative or positive


class Citation(BaseModel):
    """A literature citation."""
    pmid: str
    title: str
    relevance: float = Field(..., ge=0.0, le=1.0)
    excerpt: str


class LiteratureOutput(BaseModel):
    """Output from the Literature Agent (PubMed RAG)."""
    supporting_evidence: list[Citation] = []
    contradicting_evidence: list[Citation] = []
    evidence_strength: str = "weak"  # weak, moderate, strong


class FinalDiagnosis(BaseModel):
    """The final diagnosis or deferral."""
    diagnosis: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    deferred: bool = False
    deferral_reason: Optional[str] = None


# --- LangGraph State Definition ---

def merge_traces(left: list[str], right: list[str]) -> list[str]:
    """Reducer to append trace entries."""
    return left + right


class VerifaiState(TypedDict):
    """
    The shared state passed between all nodes in the VERIFAI graph.

    Uses LangGraph's reducer pattern for the `trace` field to accumulate
    audit entries from each node.
    """
    # --- Input ---
    image_path: str
    patient_id: Optional[str]

    # --- Agent Outputs ---
    radiologist_output: Optional[RadiologistOutput]
    critic_output: Optional[CriticOutput]
    historian_output: Optional[HistorianOutput]
    literature_output: Optional[LiteratureOutput]

    # --- Routing Control ---
    current_uncertainty: float  # The key value for routing decisions
    routing_decision: str  # "diagnose", "historian", "literature", "chief"
    steps_taken: int  # Safety counter to prevent infinite loops

    # --- Final Result ---
    final_diagnosis: Optional[FinalDiagnosis]

    # --- Audit Trail (uses reducer to accumulate) ---
    trace: Annotated[list[str], merge_traces]
