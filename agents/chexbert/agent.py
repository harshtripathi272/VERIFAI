"""
CheXbert Agent Node

LangGraph node that labels radiologist reports using F1-CheXbert.
Computes MUC Information Gain from label distribution uncertainty and alignment.
"""
from graph.state import VerifaiState, CheXbertOutput
from .model import label_report, CHEXPERT_CONDITIONS
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
    
    # Validate input: we just need rad_output
    if not rad_output:
        return {
            "chexbert_output": None,
            "trace": ["CHEXBERT: No radiologist output available"]
        }
        
    findings = rad_output.findings if getattr(rad_output, "findings", "") else ""
    impression = rad_output.impression if getattr(rad_output, "impression", "") else ""
    
    if not findings and not impression:
        return {
            "chexbert_output": None,
            "trace": ["CHEXBERT: Missing both findings and impression in radiologist report"]
        }
    
    # Merge FINDINGS and IMPRESSION for labeling
    report_text = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
    full_text_lower = report_text.lower()
    
    all_labels = {cond: "not_mentioned" for cond in CHEXPERT_CONDITIONS}
    
    # Merge for scanning
    findings = rad_output.findings if getattr(rad_output, "findings", "") else ""
    impression = rad_output.impression if getattr(rad_output, "impression", "") else ""
    combined_text = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"
    text_lower = combined_text.lower()

    try:
        # Attempt raw CheXbert labeling
        all_labels.update(label_report(combined_text))
    except Exception as e:
        print(f"[CHEXBERT] Library error fallback: {e}")
    
    # Filter initial
    filtered_labels = {c: s for c, s in all_labels.items() if s in ["present", "uncertain"]}
    
    # Keyword fallback with simple negation detection
    keywords = {
        "Cardiomegaly": ["cardiomegaly", "enlarged heart", "heart is enlarged", "big heart"],
        "Pleural Effusion": ["effusion", "pleural effusion", "fluid in lungs"],
        "Pneumonia": ["pneumonia", "consolidation", "infiltrate"],
        "Atelectasis": ["atelectasis", "collapse"],
        "Pneumothorax": ["pneumothorax", "collapsed lung"],
        "Edema": ["edema", "pulmonary edema", "heart failure"],
    }
    
    negations = ["no ", "none", "without", "clear", "negative", "normal", "rule out", "ruled out"]
    
    for condition, synonyms in keywords.items():
        for syn in synonyms:
            if syn in text_lower:
                # Basic negation check: look for "no" or "none" before the keyword
                start_idx = text_lower.find(syn)
                context = text_lower[max(0, start_idx-30):start_idx]
                
                is_negated = any(neg in context for neg in negations)
                
                if not is_negated:
                    filtered_labels[condition] = "present"
                    all_labels[condition] = "present"
                    break

    try:
        output = CheXbertOutput(labels=filtered_labels)
        u_chex = compute_chexbert_uncertainty(all_labels)
        align = compute_chexbert_alignment(all_labels, combined_text)
        
        ig = compute_ig("chexbert", u_chex, align, system_uncertainty)
        
        num_p = sum(1 for s in filtered_labels.values() if s == "present")
        trace = [
            f"CHEXBERT: Found {num_p} conditions (using fallback + negation check)",
            f"CHEXBERT MUC: unc={u_chex:.3f}, align={align:.3f}, IG={ig.information_gain:.4f}"
        ]
        if filtered_labels:
            trace.append(f"Labels: {', '.join(filtered_labels.keys())}")

        # Propagate uncertainty_history
        uncertainty_history = list(state.get("uncertainty_history", []))
        uncertainty_history.append({
            "agent": "chexbert",
            "system_uncertainty": ig.system_uncertainty_after,
        })

        return {
            "chexbert_output": output,
            "current_uncertainty": ig.system_uncertainty_after,
            "uncertainty_history": uncertainty_history,
            "trace": trace
        }
    except Exception as e:
        return {"chexbert_output": None, "trace": [f"CHEXBERT: Logic err - {e}"]}
