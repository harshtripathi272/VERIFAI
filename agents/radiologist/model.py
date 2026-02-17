"""
Radiologist Model

Real MedGemma-4B VLM Inference with LoRA using standard Hugging Face pipeline.
"""

import torch
import torch.nn as nn
from typing import Any, Tuple, Optional
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForImageTextToText,
    AutoProcessor, 
    BitsAndBytesConfig,
    AutoImageProcessor
)
from peft import PeftModel
from app.config import settings
import os
import numpy as np
from agents.radiologist.classifier import MedSigLIPClassifier
from agents.radiologist.lrp import RelevanceGenerator
from agents.radiologist.data import CHEXBERT_CLASSES

# Global model cache
_models_loaded = False
_vision_encoder = None 
_classifier_model = None 
_lrp_generator = None 
_llm = None
_processor = None

def _load_models():
    """Load MedSigLIP Classifier and MedGemma VLM."""
    global _models_loaded, _classifier_model, _lrp_generator, _llm, _processor, _vision_encoder
    
    if _models_loaded:
        return

    print("[Radiologist] Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. MedSigLIP Classifier (for analysis and LRP)
    print(f"[Radiologist] Loading MedSigLIP Classifier: {settings.MEDSIGLIP_MODEL}")
    _classifier_model = MedSigLIPClassifier(
        settings.MEDSIGLIP_MODEL, 
        num_classes=len(CHEXBERT_CLASSES)
    )
    
    head_weights_path = getattr(settings, "CLASSIFIER_WEIGHTS_PATH", "classifier_head_best.pth")
    if os.path.exists(head_weights_path):
        print(f"[Radiologist] Loading classifier head weights from {head_weights_path}")
        _classifier_model.load_head(head_weights_path)
    else:
        print(f"[Radiologist] WARNING: Classifier weights not found at {head_weights_path}. Using random init for head.")
    
    _classifier_model.to(device)
    _classifier_model.eval()
    _vision_encoder = _classifier_model.vision_model
    _lrp_generator = RelevanceGenerator(_classifier_model)
    
    # 2. MedGemma VLM
    print(f"[Radiologist] Loading MedGemma: {settings.MEDGEMMA_4B_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    _processor = AutoProcessor.from_pretrained(
        settings.MEDGEMMA_4B_MODEL,
        token=settings.HUGGINGFACE_TOKEN
    )
    
    _llm = AutoModelForImageTextToText.from_pretrained(
        settings.MEDGEMMA_4B_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=settings.HUGGINGFACE_TOKEN
    )

    # Resize for special tokens if they exist in the checkpoint
    special_tokens = ["<PA>", "<AP>", "<LATERAL>"]
    
    if settings.MEDGEMMA_LORA_ADAPTERS and "path/to" not in settings.MEDGEMMA_LORA_ADAPTERS:
        print(f"[Radiologist] Loading LoRA adapters from {settings.MEDGEMMA_LORA_ADAPTERS}")
        
        # Add special tokens blindly if they aren't there, assuming adapter was trained with them
        num_added = _processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        if num_added > 0:
            _llm.resize_token_embeddings(len(_processor.tokenizer))

        _llm = PeftModel.from_pretrained(_llm, settings.MEDGEMMA_LORA_ADAPTERS)
    else:
        print("[Radiologist] WARNING: LoRA adapter path not set or invalid. Using base model.")

    _llm.eval()
    _models_loaded = True
    print("[Radiologist] Models loaded.")

def generate_findings(image_path: str, view: str = "AP") -> dict:
    # Ensure this import works or define it locally if needed
    try:
        from .prompts import INSTRUCTION
    except ImportError:
        INSTRUCTION = "Analyze the provided chest X-rays and write a careful radiology report."

    _load_models()
    
    # Load Image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return {"findings": "Error loading image.", "impression": "Error."}

    # Prepare Prompt with Chat Template
    view_token = f"<{view}>" 
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f" {view_token}\n\n{INSTRUCTION}"}
            ]
        }
    ]

    # Process Inputs
    text = _processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _processor(text=text, images=[image], return_tensors="pt").to(_llm.device)

    # Generate
    try:
        with torch.no_grad():
            output_ids = _llm.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                use_cache=True,
            )
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        return {"findings": "Generation failed.", "impression": "Error."}

    # Decode
    input_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_len:]
    generated_text = _processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("generated_text:-", generated_text)
    return _parse_report(generated_text)

def _parse_report(text: str) -> dict:
    """Parse raw text into separate findings and impression sections."""
    text = text.strip()
    lower_text = text.lower()
    
    findings_start = lower_text.find("findings:")
    impression_start = lower_text.find("impression:")
    
    findings = ""
    impression = ""
    
    if findings_start != -1 and impression_start != -1:
        if findings_start < impression_start:
            findings = text[findings_start + 9 : impression_start].strip()
            impression = text[impression_start + 11:].strip()
        else:
            impression = text[impression_start + 11 : findings_start].strip()
            findings = text[findings_start + 9:].strip()
    elif findings_start != -1:
        findings = text[findings_start + 9:].strip()
    elif impression_start != -1:
        impression = text[impression_start + 11:].strip()
    else:
        parts = text.split("\n\n")
        if len(parts) >= 2:
            findings = parts[0]
            impression = "\n".join(parts[1:])
        else:
            findings = text
            impression = "No distinct impression section generated."
            
    return {
        "findings": findings or "No findings generated.",
        "impression": impression or "No impression generated."
    }

def analyze_disease(image_path: str) -> dict:
    """Classify diseases and generate heatmaps."""
    _load_models()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoImageProcessor.from_pretrained(settings.MEDSIGLIP_MODEL)
    
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
    except Exception as e:
        print(f"[ERROR] Classifier image load failed: {e}")
        return {"probabilities": {}, "heatmap_paths": {}}
        
    # 1. Classification
    with torch.no_grad():
        logits = _classifier_model(pixel_values) 
        probs = torch.sigmoid(logits).squeeze(0) 
        
    prob_dict = {
        cls: float(prob.item()) 
        for cls, prob in zip(CHEXBERT_CLASSES, probs)
    }
    
    # 2. Heatmap Generation
    heatmap_paths = {}
    output_dir = "output/heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).rsplit(".", 1)[0]
    
    for i, (cls_name, prob) in enumerate(zip(CHEXBERT_CLASSES, probs)):
        if prob > 0.5:
            try:
                heatmap = _lrp_generator.generate(pixel_values, target_class_index=i, device=device)
                hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                hm_uint8 = (hm_norm * 255).astype(np.uint8)
                hm_img = Image.fromarray(hm_uint8, mode='L')
                hm_resized = hm_img.resize(image.size, resample=Image.BILINEAR)
                
                # Simple save (no colormap dependency)
                save_path = os.path.join(output_dir, f"{base_name}_{cls_name.replace(' ', '_')}_heatmap.png")
                hm_resized.save(save_path)
                heatmap_paths[cls_name] = save_path
                
            except Exception as e:
                print(f"[ERROR] Heatmap generation failed for {cls_name}: {e}")
                
    return {
        "probabilities": prob_dict,
        "heatmap_paths": heatmap_paths
    }
