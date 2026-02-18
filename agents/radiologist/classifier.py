
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

class MedGemmaVisionHead(nn.Module):
    def __init__(self, num_classes: int = 14, vision_model=None):
        super().__init__()
        
        # 1. Backone: Frozen MedSigLIP (Shared from VLM)
        if vision_model is None:
            raise ValueError("MedGemmaVisionHead requires a pre-loaded 'vision_model' instance (shared from MedGemma).")
            
        print("[Classifier] Using shared vision model from VLM")
        self.vision_model = vision_model
        
        # Freeze backbone (should already be frozen, but ensure it)
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval()
        
        self.hidden_size = self.vision_model.config.hidden_size
        
        # 2. Classification Head
        # Attached to pooled representation.
        # MedSigLIP/SigLIP usually does mean pooling over patches.
        # We will duplicate that logic explicitly to ensure we can LRP through it.
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, num_classes)
        )
        
        # Initialize head
        # Top layer: Xavier
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.zeros_(self.classifier[0].bias)
        # Final layer: Small weights to start with low logits
        nn.init.normal_(self.classifier[3].weight, std=0.01)
        nn.init.zeros_(self.classifier[3].bias)

    def forward(self, pixel_values, return_dict=True):
        # 1. Vision Encoder
        # Context manager to ensure no gradients flow into backbone unless we want them
        # For LRP, we WILL need gradients, so during inference/LRP we might enable grad.
        # But for training the *head*, backbone is frozen.
        
        # When sharing vision tower with VLM, it might be in mixed precision or quantized
        # Ensure we can run it. If LRP is active, we need gradients.
        
        # We need output_attentions=True for LRP. 
        # Config might not be set if it's coming from VLM.
        # So we pass it in forward call if supported, or ensure config is set.
        
        # Check if validation run or LRP
        is_lrp = torch.is_grad_enabled() 
        
        outputs = self.vision_model(
            pixel_values=pixel_values, 
            output_attentions=True,  # Force attentions for LRP/Chefer
            output_hidden_states=True
        )
        
        # last_hidden_state: [batch, num_patches, hidden_size]
        # e.g., [B, 576, 1152] for 384x384 image (24x24 patches)
        last_hidden_state = outputs.last_hidden_state
        
        # 2. Pooling (Mean over patches)
        # dim=1 is patches 
        pooled_output = last_hidden_state.mean(dim=1) # [B, 1152]
        pooled_output = pooled_output.float()
        # 3. Classifier
        logits = self.classifier(pooled_output) # [B, 14]
        
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
                "last_hidden_state": last_hidden_state,
                "pooled_output": pooled_output
            }
        return logits

    def train_head_only(self):
        """Ensure only head is trainable."""
        self.vision_model.eval()
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        for param in self.classifier.parameters():
            param.requires_grad = True
            
    def enable_gradients_for_lrp(self):
        """Enable gradients for all parameters (required for LRP backward pass)."""
        for param in self.parameters():
            param.requires_grad = True # We need grad for inputs/weights for LRP
            
    def save_head(self, path):
        """Save only the classifier head weights."""
        torch.save(self.classifier.state_dict(), path)
        
    def load_head(self, path):
        """Load classifier head weights."""
        self.classifier.load_state_dict(torch.load(path))

