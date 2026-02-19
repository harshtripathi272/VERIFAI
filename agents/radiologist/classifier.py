
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
            raise ValueError(
                "MedGemmaVisionHead requires a shared 'vision_model' instance."
            )

        print("[Classifier] Using shared vision model (MAP pooling enabled)")
        self.vision_model = vision_model

        # Freeze backbone
        for param in self.vision_model.parameters():
            param.requires_grad = False

        self.vision_model.eval()

        self.hidden_size = self.vision_model.config.hidden_size

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
        self.classifier.load_state_dict(torch.load(path))


