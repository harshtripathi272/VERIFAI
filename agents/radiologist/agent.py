"""
Radiologist Agent Node

LangGraph node that processes chest X-ray images.
Uses MUC token entropy for single-pass uncertainty estimation.
"""

from graph.state import VerifaiState, RadiologistOutput
from .model import generate_findings
from app.config import settings
from uncertainty.muc import compute_token_entropy_from_text

def radiologist_node(state: VerifaiState) -> dict:
    """
    Radiologist Agent: Visual analysis of chest X-ray.
    
    Uses MedGemma-4B for reasoning.
    Produces plain-text FINDINGS and IMPRESSION sections.
    
    Computes uncertainty via MUC token entropy — a single-pass method that
    analyzes hedging vs confidence language in the generated text.
    When logits are available (future), switches to proper token-level entropy.
    """
    image_path = state["image_path"]
    
    # Verify file exists
    import os
    if not os.path.exists(image_path):
        return {
            "radiologist_output": None,
            "trace": [f"RADIOLOGIST: Error - Image not found: {image_path}"]
        }
        
    # Determine view (heuristic or default)
    view = state["view"]
    
    # === SINGLE-PASS GENERATION (MUC replaces multi-sample KLE) ===
    raw_output = generate_findings(image_path, view=view)
    
    # Create RadiologistOutput
    from .model import analyze_disease
    disease_analysis = analyze_disease(image_path)
    
    output = RadiologistOutput(
        findings=raw_output.get("findings", ""),
        impression=raw_output.get("impression", ""),
        disease_probabilities=disease_analysis.get("probabilities", {}),
        heatmap_paths=disease_analysis.get("heatmap_paths", {})
    )
    
    # === MUC TOKEN ENTROPY (single-pass uncertainty) ===
    # Combine findings + impression for uncertainty analysis
    full_text = f"{output.findings} {output.impression}"
    token_uncertainty = compute_token_entropy_from_text(full_text)
    
    # Build trace entry
    findings_preview = output.findings[:100] + "..." if len(output.findings) > 100 else output.findings
    impression_preview = output.impression[:100] + "..." if len(output.impression) > 100 else output.impression
    
    trace_entries = [
        f"RADIOLOGIST: Generated report (single-pass MUC)",
        f"RADIOLOGIST: Findings preview: {findings_preview}",
        f"RADIOLOGIST: Impression preview: {impression_preview}",
        f"RADIOLOGIST MUC: Token entropy uncertainty={token_uncertainty:.3f}"
    ]
    
    result = {
        "radiologist_output": output,
        "trace": trace_entries,
        "current_uncertainty": token_uncertainty,
        # Legacy key kept for DB logger compatibility (column name unchanged)
        "radiologist_kle_uncertainty": token_uncertainty,
    }
    
    return result
