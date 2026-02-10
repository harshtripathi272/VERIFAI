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
    Generate narrative report using MedGemma-4B.
    
    Returns dict with 'findings' and 'impression' text strings.
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
    """Generate plausible mock narrative report.
    
    Includes some random variation to simulate model stochasticity
    for multi-sample KLE uncertainty estimation.
    """
    import random
    
    # Vary findings slightly each time
    variants = [
        {
            "findings": (
                "FINDINGS:\n"
                "Right lower lobe demonstrates consolidation with air bronchograms. "
                "The opacity is dense and homogeneous, measuring approximately 4-5 cm. "
                "Left hilum shows mild prominence, possibly representing reactive lymphadenopathy. "
                "Cardiac silhouette is normal in size and contour. "
                "No pleural effusion or pneumothorax is identified."
            ),
            "impression": (
                "IMPRESSION:\n"
                "Right lower lobe consolidation most consistent with community-acquired pneumonia. "
                "Differential diagnosis includes viral pneumonia or atypical infection. "
                "Mild hilar prominence likely reactive. "
                "Clinical correlation recommended."
            )
        },
        {
            "findings": (
                "FINDINGS:\n"
                "There is a focal area of increased density in the right lower lobe with visible air bronchograms. "
                "The finding measures roughly 4 cm and appears relatively homogeneous. "
                "Left hilum is mildly prominent, which may represent lymphadenopathy. "
                "Heart size is within normal limits. "
                "No significant pleural abnormality detected."
            ),
            "impression": (
                "IMPRESSION:\n"
                "Findings are suggestive of right lower lobe pneumonia, likely bacterial in etiology. "
                "Atypical pneumonia or early organizing pneumonia should also be considered. "
                "Hilar prominence possibly reactive in nature. "
                "Close clinical follow-up is advised."
            )
        },
        {
            "findings": (
                "FINDINGS:\n"
                "Dense consolidation is present in the right lower lobe region with air bronchograms noted. "
                "The consolidation spans approximately 4-5 cm in maximal dimension. "
                "Mild left hilar fullness is observed. "
                "Cardiac silhouette appears unremarkable. "
                "No pneumothorax or large pleural effusion identified."
            ),
            "impression": (
                "IMPRESSION:\n"
                "Right lower lobe consolidation, most likely representing community-acquired pneumonia. "
                "Differential considerations include aspiration pneumonia or less likely atelectasis with infection. "
                "Left hilar prominence may be reactive or inflammatory. "
                "Recommend clinical correlation and follow-up imaging if symptoms persist."
            )
        },
        {
            "findings": (
                "FINDINGS:\n"
                "Right lower lobe opacity with air bronchograms consistent with airspace disease. "
                "The consolidation measures approximately 4 cm and demonstrates homogeneous density. "
                "Subtle left hilar prominence noted, likely reactive. "
                "Heart size normal. "
                "No definite pleural effusion or pneumothorax."
            ),
            "impression": (
                "IMPRESSION:\n"
                "Airspace disease in the right lower lobe, findings raise concern for bacterial pneumonia. "
                "Alternative diagnoses to consider include viral pneumonia or organizing pneumonia. "
                "Hilar findings likely benign/reactive. "
                "Clinical correlation is recommended."
            )
        },
        {
            "findings": (
                "FINDINGS:\n"
                "Consolidation identified in the right lower lobe with characteristic air bronchograms. "
                "The affected area is approximately 4-5 cm with dense, homogeneous opacification. "
                "Left hilum shows mild prominence. "
                "Cardiac size and contour are normal. "
                "No pleural complications visualized."
            ),
            "impression": (
                "IMPRESSION:\n"
                "Right lower lobe consolidation consistent with pneumonia, most likely community-acquired. "
                "Differential includes viral etiology or early empyema, though less likely given the appearance. "
                "Hilar prominence may represent reactive adenopathy. "
                "Follow-up recommended based on clinical response."
            )
        }
    ]
    
    # Return a random variant to simulate sampling variability
    return random.choice(variants)
