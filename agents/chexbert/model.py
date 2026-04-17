"""
CheXbert Labeler Singleton

Manages a single instance of F1CheXbert for efficient labeling across the workflow.
"""
from typing import Optional
import threading
from transformers import BertTokenizer

# Monkey-patch BertTokenizer for transformers 5.x compatibility
if not hasattr(BertTokenizer, "encode_plus"):
    def encode_plus_patch(self, *args, **kwargs):
        if args and isinstance(args[0], list):
            return {"input_ids": self.convert_tokens_to_ids(args[0])}
        return self(*args, **kwargs)
    BertTokenizer.encode_plus = encode_plus_patch
    print("[CheXbert] Patched BertTokenizer.encode_plus for transformers 5.x compatibility")


# Singleton pattern with thread safety
_CHEXBERT_MODEL: Optional[object] = None
_CHEXBERT_LOCK = threading.Lock()


def get_chexbert_labeler():
    """
    Get or create the singleton CheXbert labeler instance.
    
    Thread-safe lazy initialization to ensure the model is loaded only once.
    
    Returns:
        F1CheXbert: The labeler instance
    """
    global _CHEXBERT_MODEL
    
    # Quick check without lock (performance optimization)
    if _CHEXBERT_MODEL is not None:
        return _CHEXBERT_MODEL
    
    # Acquire lock for loading
    with _CHEXBERT_LOCK:
        # Double-check after acquiring lock
        if _CHEXBERT_MODEL is not None:
            return _CHEXBERT_MODEL
        
        print("[CheXbert] Loading F1-CheXbert model (one-time initialization)...")
        
        try:
            from f1chexbert import F1CheXbert
            _CHEXBERT_MODEL = F1CheXbert()
            print("[CheXbert] Model loaded successfully")
        except ImportError:
            print("[CheXbert] ERROR: f1chexbert not installed. Run: pip install f1chexbert")
            raise
        except Exception as e:
            print(f"[CheXbert] ERROR: Failed to load model: {e}")
            raise
        
        return _CHEXBERT_MODEL


# 14 Standard CheXpert Conditions (in order)
CHEXPERT_CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding"
]


def label_report(report_text: str) -> dict[str, str]:
    """
    Label a radiology report using CheXbert.
    
    Args:
        report_text: Combined FINDINGS and IMPRESSION text
    
    Returns:
        Dictionary mapping each condition to its status:
        'present', 'absent', 'uncertain', or 'not_mentioned'
    """
    labeler = get_chexbert_labeler()
    
    # Get raw label output (classification mode)
    # Returns list of 14 values: 1 (present), 0 (absent), -1 (uncertain), NaN (not mentioned)
    raw_results = labeler.get_label(report_text, mode="classification")
    
    labeled_data = {}
    
    for i, condition in enumerate(CHEXPERT_CONDITIONS):
        val = raw_results[i]
        
        if val == 1:
            status = "present"
        elif val == 0:
            status = "absent"
        elif val == -1:
            status = "uncertain"
        else:
            status = "not_mentioned"
        
        labeled_data[condition] = status
    
    return labeled_data
