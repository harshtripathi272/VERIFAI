
"""
MedSigLIP Classifier

Wraps the frozen MedSigLIP vision encoder and adds a trainable classification head
for multi-label disease prediction.

Designed to support Chefer et al. Relevance Propagation (LRP) by allowing verification
of gradients and attention maps.
"""

import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipConfig

import torch
import torch.nn as nn

class MedGemmaVisionHead(nn.Module):
    def __init__(self, num_classes: int = 14, vision_model=None):
        super().__init__()

        if vision_model is None:
             # We allow None here if loading from full checkpoint via factory
             # But for initialization, it's expected to be set.
             # The factory method handles this.
             pass
        else:
             self.vision_model = vision_model
             self.hidden_size = self.vision_model.config.hidden_size
             
             # Freeze backbone if shared (default behavior)
             for param in self.vision_model.parameters():
                param.requires_grad = False
             self.vision_model.eval()

        # Classification head (always initialized, weights loaded later)
        # If vision model is None, we assume it will be set by factory before forward
        # But we need hidden_size. 
        # Actually, let's enforce vision_model being passed OR handled by factory better.
        # Factory passes it. So we are good.
        
        # If vision_model was passed, we use its config.
        # If NOT key change: The factory passes initialized vision_model.
        if vision_model:
             self.hidden_size = 1152 # MedSigLIP/SigLIP Default

        # Freeze backbone
        # This block needs to be inside the 'else' for vision_model or handled differently
        # as self.vision_model might not be set if vision_model is None.
        # Assuming the factory method will set self.vision_model before forward is called.
        if vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()

        # The following line is problematic if vision_model is None.
        # It's also redundant if hidden_size is already set above.
        # self.hidden_size = self.vision_model.config.hidden_size 

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, num_classes)
        )

        # Initialization
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.zeros_(self.classifier[0].bias)

        nn.init.normal_(self.classifier[3].weight, std=0.01)
        nn.init.zeros_(self.classifier[3].bias)

    def forward(self, pixel_values, return_dict=True):
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,   # required for Grad-CAM++ / Chefer
            output_hidden_states=False
        )

        # 🔥 KEY CHANGE — Use MAP pooled output
        pooled_output = outputs.pooler_output  # [B, hidden_size]
        pooled_output = pooled_output.float()

        logits = self.classifier(pooled_output)

        if return_dict:
            return {
                "logits": logits,
                "attentions": outputs.attentions,
                "last_hidden_state": outputs.last_hidden_state,
                "pooled_output": pooled_output
            }

        return logits

    def train_head_only(self):
        self.vision_model.eval()
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def enable_gradients_for_lrp(self):
        for p in self.parameters():
            p.requires_grad = True

    def save_head(self, path):
        torch.save(self.classifier.state_dict(), path)

    def load_head(self, path):
        """Deprecated: Use load_medsiglip_classifier instead."""
        self.classifier.load_state_dict(torch.load(path))

def load_medsiglip_classifier(checkpoint_path: str, base_model_name: str = "google/medsiglip-448", device="cpu") -> "MedGemmaVisionHead":
    """
    Loads a full fine-tuned MedSigLIP classifier from a .pt checkpoint.
    The checkpoint must contain 'model_state_dict' and 'num_classes'.
    """
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"[Classifier] Loading MedSigLIP checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. Load Base Vision Model (Must use eager attention for LRP attention map extraction)
    print(f"[Classifier] Loading base vision model: {base_model_name}")
    vision_model = SiglipVisionModel.from_pretrained(
        base_model_name,
        attn_implementation="eager"
    )
    
    # 2. Reconstruct Wrapper
    num_classes = checkpoint.get("num_classes", 14)
    model = MedGemmaVisionHead(num_classes=num_classes, vision_model=vision_model)
    
    # 3. Load State Dict
    # Handle optional 'model_state_dict' key or direct state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print("[Classifier] Model loaded successfully.")
    
    return model


