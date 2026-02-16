"""
Radiologist Model

Real MedGemma-4B VLM Inference with LoRA and Hook-based Vision Injection.
"""

import torch
import torch.nn as nn
from typing import Any, Tuple
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
_vision_encoder = None # Now shared from classifier
_classifier_model = None # New
_lrp_generator = None    # New
_llm = None
_tokenizer = None
_image_processor = None
_projector = None

# Hook state (persistent across inference calls)
_vision_hook_handle = None
_pending_vision_embeds = None
_pending_image_token_id = None

# Special tokens
IMAGE_TOKEN = "<image>"
VIEW_TOKENS = ["<AP>", "<PA>", "<LATERAL>"]



class VisionProjector(nn.Module):
    """
    Projects mean-pooled SigLIP features to MedGemma hidden dim.

    Architecture:  Linear → GELU → LayerNorm
    Initialisation: Xavier-uniform with gain 0.1 (small scale to avoid
    disturbing the quantised LLM embeddings at the start of training).
    """

    def __init__(self, vision_hidden_size: int, lm_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, lm_hidden_size)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(lm_hidden_size)

        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.act(self.linear(x)))


def get_image_embedding(image_path: str) -> torch.Tensor:
    """
    Return projected visual embedding (1 vector per image)
    Shape: [1, hidden_size]
    """
    import time
    
    print("\n[DEBUG] === get_image_embedding START ===")
    start_time = time.time()
    
    _load_models()
    print(f"[DEBUG] Models loaded. Time: {time.time() - start_time:.2f}s")

    # Load and preprocess image
    t0 = time.time()
    image = Image.open(image_path).convert("RGB")
    print(f"[DEBUG] Image loaded: {image.size}. Time: {time.time() - t0:.2f}s")
    
    t0 = time.time()
    inputs = _image_processor(images=image, return_tensors="pt")
    print(f"[DEBUG] Image processed. Pixel values shape: {inputs.pixel_values.shape}. Time: {time.time() - t0:.2f}s")

    # Device transfer
    t0 = time.time()
    print(f"[DEBUG] Vision encoder device: {_vision_encoder.device}, dtype: {_vision_encoder.dtype}")
    pixel_values = inputs.pixel_values.to(
        _vision_encoder.device,
        dtype=_vision_encoder.dtype
    )
    print(f"[DEBUG] Pixel values transferred to {pixel_values.device}, dtype: {pixel_values.dtype}. Time: {time.time() - t0:.2f}s")

    # Vision encoding
    print("[DEBUG] Starting vision encoding...")
    t0 = time.time()
    with torch.no_grad():
        print("[DEBUG] Calling vision_encoder.forward()...")
        vision_outputs = _vision_encoder(pixel_values=pixel_values)
        print(f"[DEBUG] Vision encoding complete. Time: {time.time() - t0:.2f}s")
        
        t0 = time.time()
        patch_embeddings = vision_outputs.last_hidden_state  # [1, 576, 1152]
        print(f"[DEBUG] Patch embeddings extracted: {patch_embeddings.shape}. Time: {time.time() - t0:.2f}s")

        t0 = time.time()
        pooled = patch_embeddings.mean(dim=1)  # [1, 1152]
        print(f"[DEBUG] Mean pooling done: {pooled.shape}. Time: {time.time() - t0:.2f}s")
        
        # Ensure pooled embeddings are on the same device as projector
        t0 = time.time()
        print(f"[DEBUG] Projector device: {_projector.linear.weight.device}, dtype: {_projector.linear.weight.dtype}")
        pooled = pooled.to(device=_projector.linear.weight.device, dtype=_projector.linear.weight.dtype)
        print(f"[DEBUG] Pooled transferred. Time: {time.time() - t0:.2f}s")
        
        t0 = time.time()
        projected = _projector(pooled)         # [1, hidden_size]
        print(f"[DEBUG] Projection done: {projected.shape}. Time: {time.time() - t0:.2f}s")

    total_time = time.time() - start_time
    print(f"[DEBUG] === get_image_embedding END === Total time: {total_time:.2f}s\n")
    return projected

def _embedding_hook(module, inputs, output):
    """
    Persistent embedding hook that injects vision embeddings during forward passes.
    
    This hook is compatible with autoregressive generation:
    - During prompt encoding: swaps <image> tokens with vision embeddings
    - During decode steps: skips swapping (sequence too short)
    
    Matches the training implementation in qlora-medgemma.py.
    """
    global _pending_vision_embeds, _pending_image_token_id
    
    if _pending_vision_embeds is None or _pending_image_token_id is None:
        return output
    
    # Get input_ids from the forward call
    if isinstance(inputs, tuple) and len(inputs) > 0:
        input_ids = inputs[0]
    else:
        return output
    
    # Ensure input_ids is 2D [batch, seq]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    # Find <image> token positions in CURRENT sequence
    image_mask = input_ids == _pending_image_token_id
    b_idx, s_idx = image_mask.nonzero(as_tuple=True)
    
    # Only swap if we found <image> tokens and output is long enough
    if len(s_idx) > 0 and output.shape[1] > s_idx.max():
        # Verify we have the right number of embeddings
        if len(b_idx) != _pending_vision_embeds.shape[0]:
            print(f"[HOOK WARNING] Token count mismatch: {len(b_idx)} tokens != {_pending_vision_embeds.shape[0]} embeddings")
            return output
        
        new_output = output.clone()
        new_output[b_idx, s_idx] = _pending_vision_embeds.to(
            device=output.device,
            dtype=output.dtype
        )
        return new_output
    
    return output


def _ensure_vision_hook():
    """
    Ensure the persistent vision hook is registered on the embedding layer.
    
    Called during model loading to set up the hook once.
    """
    global _vision_hook_handle
    
    if _vision_hook_handle is not None:
        return  # Already registered
    
    embed_layer = _llm.get_input_embeddings()
    _vision_hook_handle = embed_layer.register_forward_hook(_embedding_hook)
    print("[Radiologist] Vision injection hook registered")


def set_pending_vision_embeds(projected, image_token_id):
    """
    Set vision embeddings and token ID for the hook to inject.
    
    Args:
        projected: Projected vision embeddings [num_images, hidden_size]
        image_token_id: Token ID for <image> token
    """
    global _pending_vision_embeds, _pending_image_token_id
    _pending_vision_embeds = projected
    _pending_image_token_id = image_token_id


def clear_pending_vision_embeds():
    """
    Clear pending vision embeddings after generation completes.
    """
    global _pending_vision_embeds, _pending_image_token_id
    _pending_vision_embeds = None
    _pending_image_token_id = None


def _load_models():
    """Load MedSigLIP, MedGemma (4-bit + LoRA), and Projector."""
    global _models_loaded, _vision_encoder, _llm, _tokenizer, _image_processor, _projector
    
    if _models_loaded:
        return

    print("[Radiologist] Loading real VLM models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Vision Encoder & Classifier
    # We load the Classifier wrapper which contains the SigLIP backbone.
    # This allows sharing the backbone for both VLM and Classification/LRP.
    
    print(f"\n[DEBUG] Loading MedSigLIP Classifier: {settings.MEDSIGLIP_MODEL}")
    print(f"[DEBUG] Target device: {device}")
    
    _image_processor = AutoImageProcessor.from_pretrained(
        settings.MEDSIGLIP_MODEL
    )
    
    # Initialize Classifier (wraps SigLIP)
    global _classifier_model, _lrp_generator
    _classifier_model = MedSigLIPClassifier(
        settings.MEDSIGLIP_MODEL, 
        num_classes=len(CHEXBERT_CLASSES)
    )
    
    # Load trained head weights if available
    # Default location: classifier_head_best.pth in current dir or specific path
    head_weights_path = getattr(settings, "CLASSIFIER_WEIGHTS_PATH", "classifier_head_best.pth")
    if os.path.exists(head_weights_path):
        print(f"[Radiologist] Loading classifier head weights from {head_weights_path}")
        _classifier_model.load_head(head_weights_path)
    else:
        print(f"[Radiologist] WARNING: Classifier weights not found at {head_weights_path}. Using random init for head.")
    
    _classifier_model.to(device)
    _classifier_model.eval()
    
    # Set the shared vision encoder to the classifier's backbone
    _vision_encoder = _classifier_model.vision_model
    
    # Initialize LRP Generator
    _lrp_generator = RelevanceGenerator(_classifier_model)
    
    print(f"[DEBUG] Vision encoder (shared) loaded.")

    # 2. LLM (MedGemma-4B with 4-bit LoRA)
    print(f"Loading MedGemma: {settings.MEDGEMMA_4B_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    _tokenizer = AutoTokenizer.from_pretrained(os.path.join(settings.MEDGEMMA_LORA_ROOT, "tokenizers"))
    print(_tokenizer.convert_tokens_to_ids("<image>"))
    print(_tokenizer.decode([_tokenizer.convert_tokens_to_ids("<image>")]))
    _tokenizer.padding_side = "right"
    
    # Add special tokens
    image_token_id = _tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    assert image_token_id != _tokenizer.unk_token_id

    
    _llm = AutoModelForImageTextToText.from_pretrained(
        settings.MEDGEMMA_4B_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Resize embeddings for new tokens
    if _llm.get_input_embeddings().weight.shape[0] != len(_tokenizer):
        _llm.resize_token_embeddings(len(_tokenizer))
    _llm.config.image_token_id = image_token_id  # Set config
    
    # Load LoRA adapters
    if settings.MEDGEMMA_LORA_ADAPTERS and "path/to" not in settings.MEDGEMMA_LORA_ADAPTERS:
        print(f"Loading LoRA adapters from {settings.MEDGEMMA_LORA_ADAPTERS}")
        _llm = PeftModel.from_pretrained(_llm, settings.MEDGEMMA_LORA_ADAPTERS)
    else:
        print("WARNING: LoRA adapter path not set. Using base model.")

    # 3. Projector
    print("Loading Projector...")
    
    # Handle different config structures:
    # - Gemma2: config.hidden_size
    # - Gemma3: config.hidden_dim
    # - Vision-language models: config.text_config.hidden_size
    llm_hidden_size = None
    
    # Try direct attributes first
    llm_hidden_size = getattr(_llm.config, 'hidden_size', None) or getattr(_llm.config, 'hidden_dim', None)
    
    # If not found, check text_config (for vision-language models)
    if llm_hidden_size is None and hasattr(_llm.config, 'text_config'):
        text_config = _llm.config.text_config
        llm_hidden_size = getattr(text_config, 'hidden_size', None) or getattr(text_config, 'hidden_dim', None)
    
    if llm_hidden_size is None:
        raise ValueError(f"Cannot find hidden size in model config or text_config")
    
    _projector = VisionProjector(
        _vision_encoder.config.hidden_size,
        llm_hidden_size
    )
    
    if settings.MEDGEMMA_PROJECTOR_WEIGHTS and "path/to" not in settings.MEDGEMMA_PROJECTOR_WEIGHTS:
        weights = torch.load(settings.MEDGEMMA_PROJECTOR_WEIGHTS, map_location=device)
        _projector.load_state_dict(weights)
    else:
        print("WARNING: Projector weights not found. Using random init (will output garbage).")
        
    embed_device = _llm.get_input_embeddings().weight.device
    _projector.to(device=embed_device, dtype=torch.float16)
    _projector.eval()

    # Register persistent vision hook
    _ensure_vision_hook()
    
    _models_loaded = True
    print("[Radiologist] Models loaded.")



def generate_findings(image_path: str, view: str = "AP") -> dict:
    from .prompts import INSTRUCTION

    _load_models()

    device = _llm.device

    # 1️⃣ Get projected image embedding (1 vector)
    projected = get_image_embedding(image_path)  # [1, hidden]

    # 2️⃣ Construct prompt
    view_token = f"<{view}>" if f"<{view}>" in VIEW_TOKENS else "<AP>"
    prompt = f"{IMAGE_TOKEN} {view_token}\n\n<report>\n\n{INSTRUCTION}"

    inputs = _tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    image_token_id = _tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    # 3️⃣ Set pending vision embeddings for hook to inject
    set_pending_vision_embeds(projected, image_token_id)

    # 4️⃣ Generate (hook will inject vision embeddings automatically)
    try:
        with torch.no_grad():
            # Validate projected embeddings before generation
            if torch.isnan(projected).any() or torch.isinf(projected).any():
                print("[WARNING] Projected embeddings contain NaN or Inf! Replacing with zeros.")
                projected = torch.zeros_like(projected)
                set_pending_vision_embeds(projected, image_token_id)  # Update with fixed embeddings
            
            output_ids = _llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,  # Reduced for testing untrained model (restore to 300 after retraining)
                do_sample=False,  # Use greedy decoding for stability
                use_cache=True,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id
            )
    except RuntimeError as e:
        print(f"[ERROR] Generation failed: {e}")
        print(f"[DEBUG] Input shape: {input_ids.shape}, projected shape: {projected.shape}")
        raise
    finally:
        # Always clear pending embeddings after generation
        clear_pending_vision_embeds()

    generated_text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("parsing the report")

    return _parse_report(generated_text)



def _parse_report(text: str) -> dict:
    """Parse raw text into separate findings and impression sections."""
    text = text.strip()
    
    # Normalize headers
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
            # Impression first? Unusual but possible
            impression = text[impression_start + 11 : findings_start].strip()
            findings = text[findings_start + 9:].strip()
    elif findings_start != -1:
        findings = text[findings_start + 9:].strip()
    elif impression_start != -1:
        impression = text[impression_start + 11:].strip()
    else:
        # No headers, treat whole text as findings? or verify user instruction execution failed
        # Fallback: split by double newline
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
    """
    Run disease classification and generate heatmaps for detected findings.
    
    Returns:
        {
            "probabilities": { "Pneumonia": 0.85, ... },
            "heatmap_paths": { "Pneumonia": "/path/to/heatmap.png", ... }
        }
    """
    _load_models()
    device = _classifier_model.vision_model.device
    
    # Load Image
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = _image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
    except Exception as e:
        print(f"[ERROR] Classifier image load failed: {e}")
        return {"probabilities": {}, "heatmap_paths": {}}
        
    # 1. Classification
    with torch.no_grad():
        logits = _classifier_model(pixel_values) # [1, 14]
        probs = torch.sigmoid(logits).squeeze(0) # [14]
        
    # Zip with classes
    prob_dict = {
        cls: float(prob.item()) 
        for cls, prob in zip(CHEXBERT_CLASSES, probs)
    }
    
    # 2. Heatmap Generation (for positives > 0.5)
    heatmap_paths = {}
    
    # Ensure output directory
    output_dir = "output/heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean image path for filename
    base_name = os.path.basename(image_path).replace(".jpg", "").replace(".png", "")
    
    for i, (cls_name, prob) in enumerate(zip(CHEXBERT_CLASSES, probs)):
        if prob > 0.5:
            # Generate LRP
            # Note: LRP implies backward pass, so we need grads enabled temporarily for the backward
            # But we are in inference. 
            # RelevanceGenerator handles the zero_grad and backward internally.
            
            try:
                # We need pixel_values again potentially if LRP modifies it? 
                # LRP generator clones or uses it fresh.
                
                heatmap = _lrp_generator.generate(pixel_values, target_class_index=i, device=device)
                
                # Save heatmap
                # Heatmap is [H, W] float
                
                # Normalize to 0-255 uint8 for saving
                hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                hm_uint8 = (hm_norm * 255).astype(np.uint8)
                
                # Resize to original image size for overlay? 
                # For now save the raw heatmap or resized.
                # User asked: "Heatmaps should be resized from patch resolution to full image resolution using bilinear interpolation"
                # We can do this with PIL
                
                hm_img = Image.fromarray(hm_uint8, mode='L')
                hm_resized = hm_img.resize(image.size, resample=Image.BILINEAR)
                
                # Create a colormap overlay? 
                # User just said "visual heatmaps... save as files".
                # Standard practice is jet/viridis. 
                # For simplicity, we save grayscale heatmap or apply a simple colormap if needed.
                # Let's save grayscale for now to avoid mpl dependency if not present.
                
                save_path = os.path.join(output_dir, f"{base_name}_{cls_name.replace(' ', '_')}_heatmap.png")
                hm_resized.save(save_path)
                heatmap_paths[cls_name] = save_path
                
            except Exception as e:
                print(f"[ERROR] Heatmap generation failed for {cls_name}: {e}")
                
    return {
        "probabilities": prob_dict,
        "heatmap_paths": heatmap_paths
    }

