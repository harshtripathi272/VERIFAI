"""
Radiologist Model

MedSigLIP (vision encoder) + MedGemma-4B-IT (reasoning) wrapper.
Supports mock mode for development without GPU.
"""

import torch
from typing import Any
from app.config import settings

# Lazy imports for optional ML dependencies
_models_loaded = False
_vision_encoder = None
_llm = None
_processor = None
_tokenizer = None


def _load_models():
    """Lazy-load models on first inference."""
    global _models_loaded, _vision_encoder, _llm, _processor, _tokenizer
    
    if _models_loaded:
        return
    
    if settings.MOCK_MODELS:
        print("[Radiologist] Running in MOCK mode - no models loaded")
        _models_loaded = True
        return
    
    try:
        from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        print(f"[Radiologist] Loading MedSigLIP on {device}...")
        _processor = AutoProcessor.from_pretrained(settings.MEDSIGLIP_MODEL)
        _vision_encoder = AutoModel.from_pretrained(
            settings.MEDSIGLIP_MODEL,
            torch_dtype=dtype,
            device_map="auto"
        )
        
        print(f"[Radiologist] Loading MedGemma-4B on {device}...")
        _tokenizer = AutoTokenizer.from_pretrained(settings.MEDGEMMA_4B_MODEL)
        _llm = AutoModelForCausalLM.from_pretrained(
            settings.MEDGEMMA_4B_MODEL,
            torch_dtype=dtype,
            device_map="auto"
        )
        
        _models_loaded = True
        print("[Radiologist] Models loaded successfully")
        
    except Exception as e:
        print(f"[Radiologist] Failed to load models: {e}")
        print("[Radiologist] Falling back to MOCK mode")
        settings.MOCK_MODELS = True
        _models_loaded = True


def get_image_embedding(image_path: str) -> Any:
    """
    Extract visual embedding using MedSigLIP.
    
    Returns embedding tensor or None in mock mode.
    """
    _load_models()
    
    if settings.MOCK_MODELS:
        return None
    
    from PIL import Image
    
    image = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt").to(_vision_encoder.device)
    
    with torch.no_grad():
        outputs = _vision_encoder(**inputs)
        embedding = outputs.last_hidden_state
    
    return embedding


def generate_findings(embedding: Any, dicom_metadata: dict | None) -> dict:
    """
    Generate structured findings using MedGemma-4B.
    
    Returns dict with findings, hypotheses, signals, reasoning.
    """
    _load_models()
    
    if settings.MOCK_MODELS:
        return _mock_generate()
    
    from .prompts import RADIOLOGIST_SYSTEM_PROMPT, RADIOLOGIST_USER_PROMPT
    
    # Format prompt
    user_prompt = RADIOLOGIST_USER_PROMPT.format(
        dicom_metadata=dicom_metadata or {}
    )
    
    # TODO: Implement actual VLM inference with embedding projection
    # For now, return mock until proper VLM pipeline is set up
    return _mock_generate()


def _mock_generate() -> dict:
    """Generate plausible mock output."""
    return {
        "findings": [
            {
                "location": "Right Lower Lobe",
                "observation": "Consolidation with air bronchograms",
                "severity": 0.75,
                "bounding_box": None
            },
            {
                "location": "Left Hilum", 
                "observation": "Mild prominence, possible reactive lymphadenopathy",
                "severity": 0.35,
                "bounding_box": None
            },
            {
                "location": "Cardiac Silhouette",
                "observation": "Normal size and contour",
                "severity": 0.0,
                "bounding_box": None
            }
        ],
        "hypotheses": [
            {"diagnosis": "Community-Acquired Pneumonia", "confidence": 0.68, "icd10_code": "J18.9"},
            {"diagnosis": "Viral Pneumonia", "confidence": 0.18, "icd10_code": "J12.9"},
            {"diagnosis": "Atelectasis", "confidence": 0.10, "icd10_code": "J98.11"},
            {"diagnosis": "Pulmonary Edema", "confidence": 0.04, "icd10_code": "J81.1"}
        ],
        "internal_signals": {
            "logits_top2": [3.2, 1.4],
            "logit_margin": 1.8,
            "predictive_entropy": 0.58,
            "attention_dispersion": 0.42,
            "prediction_stability": 0.85
        },
        "reasoning": (
            "Right lower lobe demonstrates dense consolidation with visible air bronchograms, "
            "a pattern highly suggestive of bacterial pneumonia. The consolidation is lobar in "
            "distribution without significant volume loss, distinguishing it from atelectasis. "
            "Cardiac silhouette is within normal limits, making cardiogenic pulmonary edema less likely."
        )
    }
