"""
Radiologist Model

Real MedGemma-4B VLM Inference with LoRA using standard Hugging Face pipeline.
MedSigLIP Classifier for Disease Detection & Grad-CAM++.
"""

import torch
import torch.nn as nn
from typing import Any, Tuple, Optional
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor, 
    BitsAndBytesConfig,
    AutoImageProcessor
)
from peft import PeftModel
from app.config import settings
import os
import numpy as np
import cv2

# Classifier & LRP
from agents.radiologist.classifier import load_medsiglip_classifier
from agents.radiologist.data import CHEXBERT_CLASSES
from agents.radiologist.lrp import RelevanceGenerator
from .prompts import INSTRUCTION

# Global model cache
_models_loaded = False
_classifier_model = None 
_llm = None
_processor = None
_siglip_processor = None

def _load_models():
    """Load MedSigLIP Classifier and MedGemma VLM."""
    global _models_loaded, _classifier_model, _llm, _processor, _siglip_processor
    
    if _models_loaded:
        return

    print("[Radiologist] Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. MedGemma VLM (Base)
    print(f"[Radiologist] Loading MedGemma Base: {settings.MEDGEMMA_4B_MODEL}")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    
    _processor = AutoProcessor.from_pretrained(
        settings.MEDGEMMA_4B_MODEL,
        token=settings.HUGGINGFACE_TOKEN
    )
    
    _llm = AutoModelForImageTextToText.from_pretrained(
        settings.MEDGEMMA_4B_MODEL,
        quantization_config=bnb_config,
        device_map={"": device},
        torch_dtype=compute_dtype,
        token=settings.HUGGINGFACE_TOKEN
    )

    # 2. Apply LoRA Adapters (DISABLED for testing)
    print("[Radiologist] LoRA adapters DISABLED per user request. Using base MedGemma model.")
    
    # Add special tokens even for base model if using same prompt format
    special_tokens = ["<PA>", "<AP>", "<LATERAL>"]
    if "<PA>" not in _processor.tokenizer.get_vocab():
         num_added = _processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
         if num_added > 0:
            _llm.resize_token_embeddings(len(_processor.tokenizer))

    # _llm = PeftModel.from_pretrained(_llm, settings.MEDGEMMA_LORA_ADAPTERS)

    _llm.eval()
    
    # 3. Load Independent MedSigLIP Classifier
    print(f"[Radiologist] Loading MedSigLIP Classifier...")
    _siglip_processor = AutoImageProcessor.from_pretrained(
        settings.MEDSIGLIP_BASE_MODEL,
        token=settings.HUGGINGFACE_TOKEN
    )
    
    try:
        _classifier_model = load_medsiglip_classifier(
            checkpoint_path=settings.MEDSIGLIP_WEIGHTS_PATH,
            base_model_name=settings.MEDSIGLIP_BASE_MODEL,
            device=device
        )
    except FileNotFoundError:
        print(f"[Radiologist] ERROR: Classifier weights not found at {settings.MEDSIGLIP_WEIGHTS_PATH}")
        _classifier_model = None

    _models_loaded = True
    print("[Radiologist] Models loaded.")

def generate_findings(
    image_paths,
    views=None
) -> dict:
    """
    Production-safe MedGemma JSON generation.
    Supports 1 or multiple images.
    Stops exactly at closing brace.
    No retraining required.
    """

    from transformers import StoppingCriteria, StoppingCriteriaList
    from utils.inference import extract_json

    _load_models()

    # --------------------------------------------------
    # Normalize inputs (backward compatible)
    # --------------------------------------------------
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    if views is None:
        views = ["AP"] * len(image_paths)

    if len(views) != len(image_paths):
        return {
            "findings": "Mismatch between number of images and views.",
            "impression": "Error."
        }

    # --------------------------------------------------
    # Load images
    # --------------------------------------------------
    loaded_images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            loaded_images.append(img)
        except Exception as e:
            return {"findings": f"Error loading image: {e}", "impression": "Error."}

    if not loaded_images:
        return {"findings": "No valid images.", "impression": "Error."}

    # --------------------------------------------------
    # Build view string EXACTLY like training
    # --------------------------------------------------
    view_tokens = " | ".join([f"<{v}>" for v in views])

    user_content = []

    for img in loaded_images:
        user_content.append({"type": "image", "image": img})

    # Add text prompt
    user_content.append({
        "type": "text",
        "text": f"\nViews: {view_tokens}\n\n{INSTRUCTION}"
    })

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    dtype = next(_llm.parameters()).dtype

    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(_llm.device, dtype=dtype)

    class StopOnCloseBrace(StoppingCriteria):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs):
            decoded = self.tokenizer.decode(
                input_ids[0], skip_special_tokens=True
            )
            return decoded.strip().endswith("}")

    stopping_criteria = StoppingCriteriaList([
        StopOnCloseBrace(_processor.tokenizer)
    ])

    try:
        with torch.inference_mode():
            output_ids = _llm.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                eos_token_id=_processor.tokenizer.eos_token_id,
                pad_token_id=_processor.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
    except Exception as e:
        return {"findings": f"Generation failed: {e}", "impression": "Error."}

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_len:]

    generated_text = _processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    ).strip()

    # Safety: truncate after last closing brace
    if "}" in generated_text:
        generated_text = generated_text[:generated_text.rfind("}") + 1]


    try:
        data = extract_json(generated_text)
        return {
            "findings": data.get("findings", ""),
            "impression": data.get("impression", "")
        }
    except Exception:
        return {
            "findings": "Failed to parse JSON.",
            "impression": generated_text[:500]
        }


def generate_findings_sampled(
    image_paths,
    views=None,
    do_sample: bool = True,
    temperature: float = 0.7,
) -> dict:
    """
    Temperature-sampled variant of generate_findings for KLE uncertainty estimation.

    Identical to generate_findings but uses do_sample=True + temperature to
    produce semantically diverse outputs. Used by the radiologist agent to
    generate N impressions that are fed to compute_semantic_uncertainty (KLE).

    Args:
        image_paths: List of image file paths
        views: List of view tags (AP/PA/LATERAL)
        do_sample: If True, use temperature sampling (diversity); else greedy
        temperature: Sampling temperature (0.7 gives moderate diversity)

    Returns:
        dict with "findings" and "impression" keys
    """
    from transformers import StoppingCriteria, StoppingCriteriaList
    from utils.inference import extract_json

    _load_models()

    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if views is None:
        views = ["AP"] * len(image_paths)

    loaded_images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            loaded_images.append(img)
        except Exception as e:
            return {"findings": f"Error loading image: {e}", "impression": "Error."}

    if not loaded_images:
        return {"findings": "No valid images.", "impression": "Error."}

    view_tokens = " | ".join([f"<{v}>" for v in views])
    user_content = []
    for img in loaded_images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": f"\nViews: {view_tokens}\n\n{INSTRUCTION}"})

    messages = [{"role": "user", "content": user_content}]
    dtype = next(_llm.parameters()).dtype
    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(_llm.device, dtype=dtype)

    class StopOnCloseBrace(StoppingCriteria):
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        def __call__(self, input_ids, scores, **kwargs):
            decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            return decoded.strip().endswith("}")

    stopping_criteria = StoppingCriteriaList([StopOnCloseBrace(_processor.tokenizer)])

    gen_kwargs = dict(
        max_new_tokens=300,
        do_sample=do_sample,
        eos_token_id=_processor.tokenizer.eos_token_id,
        pad_token_id=_processor.tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    try:
        with torch.inference_mode():
            output_ids = _llm.generate(**inputs, **gen_kwargs)
    except Exception as e:
        return {"findings": f"Generation failed: {e}", "impression": "Error."}

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_len:]
    generated_text = _processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    if "}" in generated_text:
        generated_text = generated_text[:generated_text.rfind("}") + 1]

    try:
        data = extract_json(generated_text)
        return {"findings": data.get("findings", ""), "impression": data.get("impression", "")}
    except Exception:
        return {"findings": "Failed to parse JSON.", "impression": generated_text[:500]}


def analyze_disease(image_path: str) -> dict:
    """Classify diseases and generate heatmaps using MedSigLIP and Chefer LRP."""
    _load_models()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if _classifier_model is None:
        return {"probabilities": {}, "heatmap_paths": {}}

    try:
        image = Image.open(image_path).convert("RGB")
        # Use SigLIP processor
        inputs = _siglip_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
    except Exception as e:
        print(f"[ERROR] Classifier image load failed: {e}")
        return {"probabilities": {}, "heatmap_paths": {}}
        
    # 1. Classification
    # Get standard probabilities without gradient retaining to parse diseases
    with torch.no_grad():
        logits = _classifier_model(pixel_values, return_dict=False)
        probs = torch.sigmoid(logits).squeeze(0)

    prob_dict = {
        cls: float(prob.item()) 
        for cls, prob in zip(CHEXBERT_CLASSES, probs)
    }
    
    # 2. Setup Relevance Generator for Heatmaps
    lrp = RelevanceGenerator(_classifier_model)
    
    heatmap_paths = {}
    output_dir = "output/heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).rsplit(".", 1)[0]
    
    # Prepare image for visualization (float32, 0-1)
    rgb_img = np.array(image.resize((448, 448))) / 255.0
    
    for i, (cls_name, prob) in enumerate(zip(CHEXBERT_CLASSES, probs)):
        if prob > 0.5:
            try:
                # 3. Generate Heatmap for positive classes
                heatmap = lrp.generate(pixel_values, i, device=device)
                
                # Check for empty map
                if heatmap.max() == 0:
                    print(f"[Warning] Empty LRP heatmap generated for {cls_name}")
                    continue
                
                # Resize to image size (448x448)
                heatmap_resized = cv2.resize(heatmap, (448, 448))
                
                # Apply colormap (Jet)
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
                
                # Create visualization (Overlay)
                visualization = np.uint8(255 * (rgb_img * 0.5 + heatmap_color * 0.5))
                
                # Save
                save_path = os.path.join(output_dir, f"{base_name}_{cls_name.replace(' ', '_')}_heatmap.jpg")
                img_pil = Image.fromarray(visualization)
                img_pil.save(save_path)
                heatmap_paths[cls_name] = save_path
                
            except Exception as e:
                print(f"[ERROR] Heatmap generation failed for {cls_name}: {e}")
                
    return {
        "probabilities": prob_dict,
        "heatmap_paths": heatmap_paths
    }
