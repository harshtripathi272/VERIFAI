"""
CheXbert Agent Node

LangGraph node that labels radiologist reports using F1-CheXbert.
Computes MUC Information Gain from label distribution uncertainty and alignment.
"""
from graph.state import VerifaiState, CheXbertOutput
from .model import label_report
from uncertainty.muc import (
    compute_ig,
    compute_chexbert_uncertainty,
    compute_chexbert_alignment,
)


def chexbert_node(state: VerifaiState) -> dict:
    """
    CheXbert Agent: Structured pathology labeling of radiologist report.
    
    Takes the plain-text FINDINGS and IMPRESSION from the radiologist,
    merges them, and runs CheXbert labeling to produce 14 structured
    pathology labels (present/absent/uncertain/not_mentioned).
    
    MUC Integration:
    - Computes uncertainty from label distribution (uncertain/not_mentioned ratio)
    - Computes alignment by comparing "present" labels to impression text
    - Applies IG formula to reduce system uncertainty
    
    Args:
        state: Current VerifaiState with radiologist_output populated
    
    Returns:
        Dictionary with chexbert_output, updated current_uncertainty, and trace
    """
    rad_output = state.get("radiologist_output")
    system_uncertainty = state.get("current_uncertainty", 0.5)
    
    # Validate input - Require both findings AND impression
    if not rad_output:
        return {
            "chexbert_output": None,
            "trace": ["CHEXBERT: No radiologist output available"]
        }
        
    if not rad_output.impression or not rad_output.findings:
        return {
            "chexbert_output": None,
            "trace": ["CHEXBERT: Missing findings or impression in radiologist report"]
        }
    
    # Merge FINDINGS and IMPRESSION for labeling
    report_text = f"FINDINGS: {rad_output.findings}\n\nIMPRESSION: {rad_output.impression}"
    
    try:
        # Run CheXbert labeling (all 14 labels for uncertainty computation)
        all_labels = label_report(report_text)
        
        # Filter to ONLY present and uncertain (for downstream agents)
        filtered_labels = {
            condition: status 
            for condition, status in all_labels.items() 
            if status in ["present", "uncertain"]
        }
        
        # Create output object
        output = CheXbertOutput(labels=filtered_labels)
        
        # === MUC: Compute Information Gain ===
        chexbert_uncertainty = compute_chexbert_uncertainty(all_labels)
        chexbert_alignment = compute_chexbert_alignment(all_labels, rad_output.impression)
        
        ig_result = compute_ig(
            agent_name="chexbert",
            agent_uncertainty=chexbert_uncertainty,
            alignment_score=chexbert_alignment,
            system_uncertainty=system_uncertainty,
        )
        
        # Build trace entries
        num_present = sum(1 for s in filtered_labels.values() if s == "present")
        num_uncertain = sum(1 for s in filtered_labels.values() if s == "uncertain")
        
        trace_entries = [
            f"CHEXBERT: Found {num_present} present and {num_uncertain} uncertain conditions",
            f"CHEXBERT MUC: uncertainty={chexbert_uncertainty:.3f}, "
            f"alignment={chexbert_alignment:.3f}, IG={ig_result.information_gain:.4f}",
        ]
        
        if filtered_labels:
            conditions_str = ", ".join([f"{c} ({s})" for c, s in filtered_labels.items()])
            trace_entries.append(f"CHEXBERT: {conditions_str}")
        
        return {
            "chexbert_output": output,
            "current_uncertainty": ig_result.system_uncertainty_after,
            "trace": trace_entries
        }
    
    except Exception as e:
        # Handle errors gracefully
        return {
            "chexbert_output": None,
            "trace": [f"CHEXBERT: Error during labeling - {str(e)[:100]}"]
        }
