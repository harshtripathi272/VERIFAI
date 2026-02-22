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
    image_paths = state["image_paths"]
    views = state.get("views", ["AP"])
    
    # Normalize to lists for processing
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if isinstance(views, str):
        views = [views] * len(image_paths)

    # === MULTI-SAMPLE GENERATION FOR KLE ===
    # Generate N independent samples for uncertainty estimation
    n_samples = getattr(settings, 'KLE_NUM_SAMPLES', 5)
    
    samples = []
    primary_report = None
    
    # Verify files exist
    import os
    for path in image_paths:
        if not os.path.exists(path):
            return {
                "radiologist_output": None,
                "trace": [f"RADIOLOGIST: Error - Image not found: {path}"]
            }
    
    for i in range(n_samples):
        print("Generating sample", i+1)
        # Call model with image path and view
        raw_output = generate_findings(
            image_paths=image_paths,
            views=views
        )
        
        if i == 0:
            # Use first sample as the primary report
            primary_report = raw_output
        
        # Collect impression text for KLE uncertainty calculation
        impression_text = raw_output.get("impression", "")
        if impression_text:
            samples.append(impression_text)
    
    # Create RadiologistOutput from primary sample
    # Run disease analysis (classification + heatmaps) on the first image for now
    from .model import analyze_disease
    disease_analysis = analyze_disease(image_paths[0])
    
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
