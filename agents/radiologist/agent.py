"""
Radiologist Agent Node

LangGraph node that processes chest X-ray images.
Uses KLE-based epistemic uncertainty estimation via multiple text samples.
"""

from graph.state import VerifaiState, RadiologistOutput
from .model import generate_findings
from app.config import settings
from uncertainty.kle import compute_semantic_uncertainty

def radiologist_node(state: VerifaiState) -> dict:
    """
    Radiologist Agent: Visual analysis of chest X-ray.
    
    Uses MedGemma-4B for reasoning.
    Produces plain-text FINDINGS and IMPRESSION sections.
    
    Computes KLE-based semantic uncertainty by generating multiple independent
    diagnosis samples and measuring their semantic dispersion using kernel language entropy.
    Uncertainty is computed externally and stored separately from the report text.
    """
    image_paths = state["image_path"]
    views = state.get("view", "AP")
    
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
        findings=primary_report.get("findings", ""),
        impression=primary_report.get("impression", ""),
        disease_probabilities=disease_analysis.get("probabilities", {}),
        heatmap_paths=disease_analysis.get("heatmap_paths", {})
    )
    
    # === KLE SEMANTIC UNCERTAINTY ESTIMATION ===
    kle_uncertainty = None
    
    if len(samples) >= 2:
        kle_uncertainty = compute_semantic_uncertainty(samples)
    
    # Build trace entry
    findings_preview = output.findings[:100] + "..." if len(output.findings) > 100 else output.findings
    impression_preview = output.impression[:100] + "..." if len(output.impression) > 100 else output.impression
    
    trace_entries = [
        f"RADIOLOGIST: Generated report from {n_samples} samples",
        f"RADIOLOGIST: Findings preview: {findings_preview}",
        f"RADIOLOGIST: Impression preview: {impression_preview}"
    ]
    
    if kle_uncertainty is not None:
        trace_entries.append(f"RADIOLOGIST KLE: Epistemic uncertainty={kle_uncertainty:.3f} (from {n_samples} samples)")
    
    result = {
        "radiologist_output": output,
        "trace": trace_entries
    }
    
    # Store KLE score in state for downstream use (Critic, logging)
    if kle_uncertainty is not None:
        result["radiologist_kle_uncertainty"] = kle_uncertainty
    
    return result

