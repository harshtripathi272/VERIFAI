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
    _load_models()

    image = Image.open(image_path).convert("RGB")
    inputs = _image_processor(images=image, return_tensors="pt")

    pixel_values = inputs.pixel_values.to(
        _vision_encoder.device,
        dtype=_vision_encoder.dtype
    )

    with torch.no_grad():
        vision_outputs = _vision_encoder(pixel_values=pixel_values)
        patch_embeddings = vision_outputs.last_hidden_state  # [1, 576, 1152]

        pooled = patch_embeddings.mean(dim=1)  # [1, 1152]
        projected = _projector(pooled)         # [1, hidden_size]

    return projected

def inject_vision_hook(projected_embeds, image_token_id, input_ids):
    """
    Registers forward hook to replace <image> token embedding.
    """

    embed_layer = _llm.get_input_embeddings()

    image_mask = input_ids == image_token_id
    b_idx, s_idx = image_mask.nonzero(as_tuple=True)

    assert len(b_idx) == projected_embeds.shape[0], \
        f"<image> count ({len(b_idx)}) != image count ({projected_embeds.shape[0]})"

    def hook(module, inputs, output):
        new_output = output.clone()
        new_output[b_idx, s_idx] = projected_embeds.to(
            device=output.device,
            dtype=output.dtype
        )
        return new_output

    handle = embed_layer.register_forward_hook(hook)
    return handle


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
    _projector = VisionProjector(
        _vision_encoder.config.hidden_size,
        _llm.config.hidden_size
    )
    
    if settings.MEDGEMMA_PROJECTOR_WEIGHTS and "path/to" not in settings.MEDGEMMA_PROJECTOR_WEIGHTS:
        weights = torch.load(settings.MEDGEMMA_PROJECTOR_WEIGHTS, map_location=device)
        _projector.load_state_dict(weights)
    else:
        print("WARNING: Projector weights not found. Using random init (will output garbage).")
        
    embed_device = _llm.get_input_embeddings().weight.device
    _projector.to(device=embed_device, dtype=torch.float16)
    _projector.eval()

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
    prompt = f"{IMAGE_TOKEN} {view_token}\n{INSTRUCTION}"

    inputs = _tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    image_token_id = _tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    # 3️⃣ Register hook (exact training behavior)
    hook_handle = inject_vision_hook(projected, image_token_id, input_ids)

    # 4️⃣ Generate normally (IMPORTANT: pass input_ids)
    with torch.no_grad():
        output_ids = _llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            use_cache=True,
            pad_token_id=_tokenizer.eos_token_id
        )

    # Remove hook after generation
    hook_handle.remove()

    generated_text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
