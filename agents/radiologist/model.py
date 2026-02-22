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

# Classifier & Grad-CAM
from agents.radiologist.classifier import load_medsiglip_classifier
from agents.radiologist.data import CHEXBERT_CLASSES
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
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
        device_map="auto",
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
    _siglip_processor = AutoImageProcessor.from_pretrained(settings.MEDSIGLIP_BASE_MODEL)
    
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

class ClassifierWrapper(nn.Module):
    """
    Wrapper for MedGemmaVisionHead to ensure it returns only logits.
    Required because pytorch-grad-cam expects a single tensor output.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Target layers must be accessible via this wrapper if using string targets,
        # but we pass the layer object directly, so this is fine.
        
    def forward(self, x):
        return self.model(x, return_dict=False)

def reshape_transform(tensor):
    """
    Reshape SigLIP output for Grad-CAM.
    SigLIP output: [B, H*W, C] -> [B, C, H, W]
    Assumes square grid because SigLIP/ViT.
    """
    # Tensor shape is [Batch, Tokens, Channels]
    # No CLS token in SigLIP usually, but verify via config if using different backbone
    # SigLIP 448x448 / 16 = 28x28 grid = 784 tokens
    
    b, t, c = tensor.shape
    h = w = int(t ** 0.5) 
    
    result = tensor.reshape(b, h, w, c)
    result = result.transpose(2, 3).transpose(1, 2) # [B, C, H, W]
    return result

def analyze_disease(image_path: str) -> dict:
    """Classify diseases and generate heatmaps using MedSigLIP and Grad-CAM++."""
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
    # Only need forward pass here, but GradCAM needs gradients enabled on inputs? 
    # No, usually we do forward pass with grad enabled for CAM but we want probs first.
    # We can do one pass.

    inputs_param = pixel_values.requires_grad_(True) # Ensure input grad for safety if needed
    
    # We need to manually run forward to get logits because the wrapper does pooling internally
    # _classifier_model is in eval() mode usually.
    
    # For GradCAM, we need to wrap the VISION MODEL encoded layers, not the whole wrapper necessarily.
    # Target Layer: Last Encoder Layer of SigLIP
    # Target Layer: Last Encoder Layer of SigLIP
    # Structure: MedGemmaVisionHead -> SiglipVisionModel -> SiglipVisionTransformer (vision_model) -> SiglipEncoder (encoder) -> layers
    target_layer = _classifier_model.vision_model.vision_model.encoder.layers[-1]


    # 2. Setup Grad-CAM
    cam = GradCAMPlusPlus(
        model=ClassifierWrapper(_classifier_model), 
        target_layers=[target_layer],
        reshape_transform=reshape_transform # VITAL for ViT
    )

    # We need a custom forward func for the wrapper if standard CAM doesn't work out of box
    # The wrapper's forward returns dict or logits. CAM expects logits.
    # Wrapper returns dict by default if return_dict=True?
    # Let's check wrapper. It calls self.vision_model and then classifier.
    # We need to ensure wrapper returns Logits for CAM to work.
    
    # Let's compute Probs first
    with torch.no_grad():
        logits = _classifier_model(pixel_values, return_dict=False)
        probs = torch.sigmoid(logits).squeeze(0)

    prob_dict = {
        cls: float(prob.item()) 
        for cls, prob in zip(CHEXBERT_CLASSES, probs)
    }
    
    # 3. Generate Heatmaps
    heatmap_paths = {}
    output_dir = "output/heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).rsplit(".", 1)[0]
    
    # Prepare image for visualization (float32, 0-1)
    rgb_img = np.array(image.resize((448, 448))) / 255.0
    
    for i, (cls_name, prob) in enumerate(zip(CHEXBERT_CLASSES, probs)):
        if prob > 0.5:
            try:
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                targets = [ClassifierOutputTarget(i)]
                
                # We must ensure the model returns TENSOR logits
                # Our wrapper needs to handle 'return_dict' check based on usage?
                # Actually, standard usage of library might fail if model signature is non-standard.
                # Monkey-patch or ensure wrapper handles it.
                # Wrapper signature: forward(pixel_values, return_dict=True)
                # CAM calls: model(input_tensor, ...)
                # So it passes 'pixel_values' as positional arg 0? No, it passes it as 'input_tensor' usually?
                # CAM usage: model(input_tensor)
                # SigLIP arg is 'pixel_values'.
                # We might need a wrapper lambda or modify call.
                
                # Correct way: Passing kwarg via enable_caching? No.
                # Actually, pytorch-grad-cam handles this via forward hooks.
                # But the forward call needs to match.
                
                # Let's rely on pixel_values being the first argument in our wrapper.
                
                grayscale_cam = cam(
                   input_tensor=pixel_values, 
                   targets=targets
                ) # This enables grad context internally

                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                
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
