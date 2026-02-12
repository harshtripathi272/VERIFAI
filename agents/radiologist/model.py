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
    AutoModelForCausalLM, 
    AutoProcessor, 
    BitsAndBytesConfig,
    AutoImageProcessor
)
from peft import PeftModel
from app.config import settings

# Global model cache
_models_loaded = False
_vision_encoder = None
_llm = None
_tokenizer = None
_image_processor = None
_projector = None

# Special tokens
IMAGE_TOKEN = "<image>"
VIEW_TOKENS = ["<AP>", "<PA>", "<Lateral>"]


class VisionProjector(nn.Module):
    """Projects MedSigLIP embeddings to MedGemma dimension."""
    def __init__(self, input_dim=1152, output_dim=2048):  # SigLIP-So400m -> Gemma-2B/4B
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def _load_models():
    """Load MedSigLIP, MedGemma (4-bit + LoRA), and Projector."""
    global _models_loaded, _vision_encoder, _llm, _tokenizer, _image_processor, _projector
    
    if _models_loaded:
        return

    print("[Radiologist] Loading real VLM models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Vision Encoder (MedSigLIP)
    print(f"Loading MedSigLIP: {settings.MEDSIGLIP_MODEL}")
    _image_processor = AutoImageProcessor.from_pretrained(
        settings.MEDSIGLIP_MODEL,
        size={"height": 384, "width": 384}  # User requested 384
    )
    # Load vision model (can use AutoModel or just the vision tower if available)
    from transformers import SiglipVisionModel
    _vision_encoder = SiglipVisionModel.from_pretrained(
        settings.MEDSIGLIP_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    ).eval()

    # 2. LLM (MedGemma-4B with 4-bit LoRA)
    print(f"Loading MedGemma: {settings.MEDGEMMA_4B_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    _tokenizer = AutoTokenizer.from_pretrained(settings.MEDGEMMA_4B_MODEL)
    _tokenizer.padding_side = "right"
    
    # Add special tokens
    _tokenizer.add_tokens([IMAGE_TOKEN] + VIEW_TOKENS, special_tokens=True)
    image_token_id = _tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    
    _llm = AutoModelForCausalLM.from_pretrained(
        settings.MEDGEMMA_4B_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Resize embeddings for new tokens
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
    _projector = VisionProjector(input_dim=1152, output_dim=_llm.config.hidden_size)
    
    if settings.MEDGEMMA_PROJECTOR_WEIGHTS and "path/to" not in settings.MEDGEMMA_PROJECTOR_WEIGHTS:
        weights = torch.load(settings.MEDGEMMA_PROJECTOR_WEIGHTS, map_location=device)
        _projector.load_state_dict(weights)
    else:
        print("WARNING: Projector weights not found. Using random init (will output garbage).")
        
    _projector.to(device=device, dtype=torch.float16) # Projector usually stays in float16
    _projector.eval()

    _models_loaded = True
    print("[Radiologist] Models loaded.")


def get_image_embedding(image_path: str) -> torch.Tensor:
    """Get projected visual embedding (Projector(Encoder(Image)))."""
    _load_models()
    
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess
    inputs = _image_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(_vision_encoder.device, dtype=_vision_encoder.dtype)
    
    with torch.no_grad():
        # Encode
        vision_outputs = _vision_encoder(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state  # [1, 576, 1152] for 384x384
        
        # Project
        projected_embeds = _projector(image_embeds) # [1, 576, 2048]
        
    return projected_embeds


def generate_findings(image_path: str, view: str = "AP") -> dict:
    """
    Generate findings using hook-based VLM inference.
    
    Args:
        image_path: Path to chest X-ray.
        view: "AP", "PA", or "Lateral".
    """
    from .prompts import INSTRUCTION
    
    _load_models()
    
    # 1. Get Visual Embeddings
    visual_embeds = get_image_embedding(image_path) # [1, S, H]
    
    # 2. Construct Prompt
    # <image> <VIEW> INSTRUCTION
    view_token = f"<{view}>" if f"<{view}>" in VIEW_TOKENS else "<AP>"
    prompt = f"{IMAGE_TOKEN} {view_token}\n{INSTRUCTION}"
    
    inputs = _tokenizer(prompt, return_tensors="pt").to(_llm.device)
    
    # Find <image> token position
    input_ids = inputs.input_ids
    image_token_id = _tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    
    # Find where image token is
    # Assuming only 1 image token at the start
    has_image = (input_ids == image_token_id).any()
    
    # 3. Define Hook for Injection
    def forward_hook(module, input, output):
        # input is tuple (embeddings,)
        # output is tensor [batch, seq, hidden]
        if not has_image:
            return output
            
        # Locate indices of <image> tokens
        # We need to replace the single <image> embedding with the sequence of visual embeddings
        # NOTE: This approach (replacing 1 token with N) requires recalculating position IDs 
        # or constructing inputs_embeds manually BEFORE calling generate.
        #
        # Better approach: Construct inputs_embeds manually.
        pass
        
    # === BETTER APPROACH: Construct inputs_embeds directly ===
    # This avoids complex hooking logic for sequence length changes
    
    # Embed all text tokens
    token_embeds = _llm.model.embed_tokens(input_ids) # [1, L, H]
    
    # Find index of <image>
    batch_indices, seq_indices = torch.where(input_ids == image_token_id)
    
    if len(seq_indices) > 0:
        idx = seq_indices[0]
        
        # Split: [prefix, image_token, suffix]
        prefix_embeds = token_embeds[:, :idx, :]
        suffix_embeds = token_embeds[:, idx+1:, :]
        
        # Concatenate: [prefix, visual_embeds, suffix]
        # visual_embeds is [1, 576, H]
        inputs_embeds = torch.cat([prefix_embeds, visual_embeds, suffix_embeds], dim=1)
        
        # New attention mask
        seq_len = inputs_embeds.shape[1]
        attention_mask = torch.ones((1, seq_len), device=_llm.device)
    else:
        inputs_embeds = token_embeds
        attention_mask = inputs.attention_mask
        
    # 4. Generate
    with torch.no_grad():
        output_ids = _llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            use_cache=True,
            pad_token_id=_tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 5. Parse Output
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
