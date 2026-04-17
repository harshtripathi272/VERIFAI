"""
Radiologist Agent Node

LangGraph node that processes chest X-ray images.
Uses MUC token entropy (single pass) OR KLE multi-sample semantic uncertainty
depending on MOCK_MODELS setting and KLE_NUM_SAMPLES config.
"""

from graph.state import VerifaiState, RadiologistOutput
from .model import generate_findings
from app.config import settings
from uncertainty.muc import compute_token_entropy_from_text
from uncertainty.kle import compute_semantic_uncertainty


def _generate_multi_sample(
    image_paths: list,
    views: list,
    n_samples: int,
) -> tuple[dict, list[str]]:
    """
    Generate n_samples independent impressions for KLE uncertainty estimation.

    Returns:
        primary_report: The first/primary report (greedy decode)
        impression_samples: All impression strings (for KLE)
    """
    from agents.radiologist.model import generate_findings_sampled

    impression_samples: list[str] = []
    primary_report: dict = {}

    for i in range(n_samples):
        # First sample: greedy (authoritative); rest: sampled (for diversity)
        use_sampling = (i > 0)
        try:
            report = generate_findings_sampled(
                image_paths=image_paths,
                views=views,
                do_sample=use_sampling,
                temperature=0.7 if use_sampling else 1.0,
            )
        except Exception:
            report = generate_findings(image_paths=image_paths, views=views)

        if i == 0:
            primary_report = report

        imp = report.get("impression", "")
        if imp:
            impression_samples.append(imp)

    return primary_report, impression_samples


def radiologist_node(state: VerifaiState) -> dict:
    """
    Radiologist Agent: Visual analysis of chest X-ray.

    Uses MedGemma-4B for reasoning.
    Produces plain-text FINDINGS and IMPRESSION sections.

    Uncertainty is computed as:
    - Real mode: KLE semantic uncertainty over KLE_NUM_SAMPLES independent impressions
    - Mock mode: Heuristic text entropy (hedging vs confidence markers)
    """
    image_paths = state["image_paths"]
    views = state.get("views", ["AP"])

    # Normalize to lists for processing
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if isinstance(views, str):
        views = [views] * len(image_paths)

    # Verify files exist
    import os
    for path in image_paths:
        if not os.path.exists(path):
            return {
                "radiologist_output": None,
                "trace": [f"RADIOLOGIST: Error - Image not found: {path}"]
            }

    n_samples = getattr(settings, 'KLE_NUM_SAMPLES', 3)
    impression_samples: list[str] = []
    token_uncertainty: float = 0.5

    if settings.MOCK_MODELS:
        # Mock mode: single pass + heuristic uncertainty
        raw_output = generate_findings(image_paths=image_paths, views=views)
        full_text = f"{raw_output.get('findings', '')} {raw_output.get('impression', '')}"
        token_uncertainty = compute_token_entropy_from_text(full_text)
        impression_samples = [raw_output.get("impression", "")]
        uncertainty_method = "heuristic_text_entropy (mock)"
    else:
        # Real mode: multi-sample KLE uncertainty
        try:
            primary_report, impression_samples = _generate_multi_sample(
                image_paths, views, n_samples
            )
            raw_output = primary_report
            if len(impression_samples) >= 2:
                token_uncertainty = compute_semantic_uncertainty(impression_samples)
                uncertainty_method = f"KLE (n={len(impression_samples)} samples)"
            else:
                full_text = f"{primary_report.get('findings', '')} {primary_report.get('impression', '')}"
                token_uncertainty = compute_token_entropy_from_text(full_text)
                uncertainty_method = "heuristic_text_entropy (single sample)"
        except Exception as e:
            # Fallback: single pass
            raw_output = generate_findings(image_paths=image_paths, views=views)
            full_text = f"{raw_output.get('findings', '')} {raw_output.get('impression', '')}"
            token_uncertainty = compute_token_entropy_from_text(full_text)
            uncertainty_method = f"heuristic_text_entropy (KLE failed: {e})"

    # Run disease analysis (classification + heatmaps) on the first image
    from .model import analyze_disease
    disease_analysis = analyze_disease(image_paths[0])

    output = RadiologistOutput(
        findings=raw_output.get("findings", ""),
        impression=raw_output.get("impression", ""),
        disease_probabilities=disease_analysis.get("probabilities", {}),
        heatmap_paths=disease_analysis.get("heatmap_paths", {})
    )

    findings_preview = output.findings[:100] + "..." if len(output.findings) > 100 else output.findings
    impression_preview = output.impression[:100] + "..." if len(output.impression) > 100 else output.impression

    trace_entries = [
        f"RADIOLOGIST: Generated report ({uncertainty_method})",
        f"RADIOLOGIST: Findings preview: {findings_preview}",
        f"RADIOLOGIST: Impression preview: {impression_preview}",
        f"RADIOLOGIST MUC: Uncertainty={token_uncertainty:.3f} (method={uncertainty_method})"
    ]

    # Initialise uncertainty_history for downstream agents
    uncertainty_history = [{
        "agent": "radiologist",
        "system_uncertainty": token_uncertainty,
    }]

    return {
        "radiologist_output": output,
        "trace": trace_entries,
        "current_uncertainty": token_uncertainty,
        "uncertainty_history": uncertainty_history,
        # Legacy key kept for DB logger compatibility
        "radiologist_kle_uncertainty": token_uncertainty,
    }
