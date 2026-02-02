"""
Radiologist Agent Node

LangGraph node that processes chest X-ray images.
"""

from graph.state import (
    VerifaiState, 
    RadiologistOutput,
    VisualFinding,
    DiagnosisHypothesis,
    InternalSignals
)
from .model import get_image_embedding, generate_findings


def radiologist_node(state: VerifaiState) -> dict:
    """
    Radiologist Agent: Visual analysis of chest X-ray.
    
    Uses MedSigLIP for visual encoding and MedGemma-4B for reasoning.
    Produces structured findings, hypotheses, and internal uncertainty signals.
    """
    image_path = state["image_path"]
    dicom_metadata = state.get("dicom_metadata")
    
    # Get visual embedding
    embedding = get_image_embedding(image_path)
    
    # Generate structured output
    raw_output = generate_findings(embedding, dicom_metadata)
    
    # Parse into Pydantic models
    findings = [
        VisualFinding(**f) for f in raw_output.get("findings", [])
    ]
    
    hypotheses = [
        DiagnosisHypothesis(**h) for h in raw_output.get("hypotheses", [])
    ]
    
    signals_data = raw_output.get("internal_signals", {})
    signals = InternalSignals(
        logits_top2=signals_data.get("logits_top2", [0.0, 0.0]),
        logit_margin=signals_data.get("logit_margin", 0.0),
        predictive_entropy=signals_data.get("predictive_entropy", 1.0),
        attention_dispersion=signals_data.get("attention_dispersion", 0.5),
        prediction_stability=signals_data.get("prediction_stability", 0.5)
    )
    
    output = RadiologistOutput(
        findings=findings,
        hypotheses=hypotheses,
        internal_signals=signals,
        reasoning=raw_output.get("reasoning", "")
    )
    
    # Build trace entry
    top_dx = hypotheses[0].diagnosis if hypotheses else "Unknown"
    top_conf = hypotheses[0].confidence if hypotheses else 0.0
    trace_entry = f"RADIOLOGIST: {len(findings)} findings, Top Dx: {top_dx} ({top_conf:.0%})"
    
    return {
        "radiologist_output": output,
        "trace": [trace_entry]
    }
