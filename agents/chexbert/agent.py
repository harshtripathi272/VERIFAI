"""
CheXbert Agent Node

LangGraph node that labels radiologist reports using F1-CheXbert.
"""
from graph.state import VerifaiState, CheXbertOutput
from .model import label_report


def chexbert_node(state: VerifaiState) -> dict:
    """
    CheXbert Agent: Structured pathology labeling of radiologist report.
    
    Takes the plain-text FINDINGS and IMPRESSION from the radiologist,
    merges them, and runs CheXbert labeling to produce 14 structured
    pathology labels (present/absent/uncertain/not_mentioned).
    
    These labels are then available to downstream agents (Historian, Literature)
    for more precise evidence retrieval.
    
    Args:
        state: Current VerifaiState with radiologist_output populated
    
    Returns:
        Dictionary with chexbert_output and trace updates
    """
    rad_output = state.get("radiologist_output")
    
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
    # CheXbert works best with complete report text
    report_text = f"FINDINGS: {rad_output.findings}\n\nIMPRESSION: {rad_output.impression}"
    
    try:
        # Run CheXbert labeling
        all_labels = label_report(report_text)
        
        # Filter to ONLY present and uncertain (ignore absent and not_mentioned)
        filtered_labels = {
            condition: status 
            for condition, status in all_labels.items() 
            if status in ["present", "uncertain"]
        }
        
        # Create simplified output object
        output = CheXbertOutput(labels=filtered_labels)
        
        # Build trace entries
        num_present = sum(1 for s in filtered_labels.values() if s == "present")
        num_uncertain = sum(1 for s in filtered_labels.values() if s == "uncertain")
        
        trace_entries = [
            f"CHEXBERT: Found {num_present} present and {num_uncertain} uncertain conditions"
        ]
        
        if filtered_labels:
            conditions_str = ", ".join([f"{c} ({s})" for c, s in filtered_labels.items()])
            trace_entries.append(f"CHEXBERT: {conditions_str}")
        
        return {
            "chexbert_output": output,
            "trace": trace_entries
        }
    
    except Exception as e:
        # Handle errors gracefully
        return {
            "chexbert_output": None,
            "trace": [f"CHEXBERT: Error during labeling - {str(e)[:100]}"]
        }
