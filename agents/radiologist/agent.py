"""
Radiologist Agent Node

LangGraph node that processes chest X-ray images.
Includes KLE-based epistemic uncertainty estimation using multiple diagnosis samples.
"""

from graph.state import (
    VerifaiState, 
    RadiologistOutput,
    VisualFinding,
    DiagnosisHypothesis,
    InternalSignals
)
from .model import get_image_embedding, generate_findings
from app.config import settings


def _generate_diagnosis_samples(embedding, dicom_metadata, n_samples: int) -> list[str]:
    """
    Generate N independent diagnosis summaries for KLE uncertainty estimation.
    
    Each sample represents an independent inference pass, allowing us to
    measure epistemic instability in the diagnostic hypothesis space.
    
    Args:
        embedding: Visual embedding from MedSigLIP
        dicom_metadata: DICOM metadata dict
        n_samples: Number of samples to generate
        
    Returns:
        List of diagnosis summary strings
    """
    samples = []
    for _ in range(n_samples):
        raw_output = generate_findings(embedding, dicom_metadata)
        
        # Build a summary of this sample's diagnosis
        hypotheses = raw_output.get("hypotheses", [])
        if hypotheses:
            # Format: "Primary: Pneumonia (68%), Secondary: Viral Pneumonia (18%)"
            summary_parts = []
            for i, h in enumerate(hypotheses[:3]):  # Top 3
                label = ["Primary", "Secondary", "Tertiary"][i] if i < 3 else f"Hypothesis {i+1}"
                summary_parts.append(f"{label}: {h.get('diagnosis', 'Unknown')} ({h.get('confidence', 0):.0%})")
            samples.append("; ".join(summary_parts))
        else:
            samples.append("No diagnosis available")
    
    return samples


def radiologist_node(state: VerifaiState) -> dict:
    """
    Radiologist Agent: Visual analysis of chest X-ray.
    
    Uses MedSigLIP for visual encoding and MedGemma-4B for reasoning.
    Produces structured findings, hypotheses, and internal uncertainty signals.
    
    Also computes KLE-based semantic uncertainty using multiple diagnosis samples
    to estimate early epistemic instability. This score is stored in state for
    logging and analysis (e.g., biasing critic behavior) but does NOT gate consensus.
    """
    image_path = state["image_path"]
    dicom_metadata = state.get("dicom_metadata")
    
    # Get visual embedding
    embedding = get_image_embedding(image_path)
    
    # Generate structured output (primary inference)
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
    
    # === KLE SEMANTIC UNCERTAINTY ESTIMATION ===
    # Generate multiple diagnosis samples for epistemic uncertainty estimation
    kle_uncertainty = None
    n_samples = getattr(settings, 'KLE_NUM_SAMPLES', 5)
    
    if n_samples > 1:
        from uncertainty.kle import compute_semantic_uncertainty
        
        # Generate N-1 additional samples (we already have 1 from primary inference)
        # Build primary sample summary
        primary_summary = []
        for i, h in enumerate(hypotheses[:3]):
            label = ["Primary", "Secondary", "Tertiary"][i] if i < 3 else f"Hypothesis {i+1}"
            primary_summary.append(f"{label}: {h.diagnosis} ({h.confidence:.0%})")
        
        samples = ["; ".join(primary_summary)] if primary_summary else []
        
        # Generate additional samples
        additional_samples = _generate_diagnosis_samples(
            embedding, dicom_metadata, n_samples - 1
        )
        samples.extend(additional_samples)
        
        # Compute KLE uncertainty
        if len(samples) >= 2:
            kle_uncertainty = compute_semantic_uncertainty(samples)
    
    # Build trace entry
    top_dx = hypotheses[0].diagnosis if hypotheses else "Unknown"
    top_conf = hypotheses[0].confidence if hypotheses else 0.0
    trace_entries = [f"RADIOLOGIST: {len(findings)} findings, Top Dx: {top_dx} ({top_conf:.0%})"]
    
    if kle_uncertainty is not None:
        trace_entries.append(f"RADIOLOGIST KLE: Epistemic uncertainty={kle_uncertainty:.3f} (from {n_samples} samples)")
    
    result = {
        "radiologist_output": output,
        "trace": trace_entries
    }
    
    # Store KLE score in state for downstream use (logging, critic biasing)
    if kle_uncertainty is not None:
        result["radiologist_kle_uncertainty"] = kle_uncertainty
    
    return result

