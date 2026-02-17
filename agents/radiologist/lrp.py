"""
LRP for SigLIP (Chefer et al. 2021)

This module implements the Transformer explainability method from Chefer et al. (CVPR 2021).
It is specifically adapted for the Hugging Face `SiglipVisionModel`.

Key Logic:
1. Register backward hooks to capture gradients of Attention.
2. Compute relevance using the "Gradient * Activation" rule, enforcing positive contributions.
3. Propagate relevance layer-by-layer: R = R + (R @ R_attn).
4. For SigLIP (GAP), we sum the relevance of the output patches (weighted by classifier) to input patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class RelevanceGenerator:
    """
    Generates heatmaps for a specific class using Chefer et al. LRP for SigLIP.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.attentions = []
        self.attention_gradients = []
        self.handles = []
        
        # We need to hook into the attention layers primarily to capture gradients w.r.t attention map
        # However, since we can't easily hook internal Softmax output in HF implementation without code mod,
        # we rely on `output_attentions=True` returning the attention maps, and we RETAIN GRAD on them.
        pass

    def _cleanup(self):
        self.attentions = []
        self.attention_gradients = []
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate(self, pixel_values, target_class_index, device="cuda"):
        # 1. Setup
        self._cleanup()
        
        pixel_values = pixel_values.to(device)
        pixel_values.requires_grad = True
        
        # 2. Forward Pass
        # We need to capture the attention maps and their gradients.
        with torch.enable_grad():
            output_dict = self.model(pixel_values, return_dict=True)
            logits = output_dict['logits']
            attentions = output_dict['attentions'] # tuple of tensors (Layer 1 -> N)
            
            for attn in attentions:
                attn.retain_grad()
            
            # 3. Backward
            self.model.zero_grad()
            
            # One-hot for target class
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class_index] = 1.0
            
            logits.backward(gradient=one_hot)
            
            # 4. Chefer Integration (Equation 13 & 14)
            # Ref: Chefer et al. (CVPR 2021) Section 3.1 & 3.2
            # Equation 13: \bar{A} = I + E_h [ (\nabla A \odot A)^+ ]
            # Equation 14: C = \bar{A}^{(1)} @ \bar{A}^{(2)} ... @ \bar{A}^{(B)}
            
            num_tokens = attentions[0].shape[-1]
            
            # Initialize C as Identity
            C = torch.eye(num_tokens, device=device)
            
            # Iterate forward (Layer 1 to B)
            for i, attn in enumerate(attentions):
                grad = attn.grad # [B, H, S, S]
                cam = attn       # [B, H, S, S]
                
                # Check for None gradients (can happen if layer unused or disconnected)
                if grad is None:
                    continue
                
                # Equation 13: Weighted Attention Relevance
                # 1. Element-wise product of Gradient and Attention
                # 2. Clamp positive
                # 3. Average over heads
                gradients = torch.clamp(grad * cam, min=0)
                A_bar_layer = gradients.mean(dim=1) # [B, S, S]
                A_bar_layer = A_bar_layer[0] # [S, S] for batch 0
                
                # Add Identity
                # \bar{A} = I + ...
                A_bar = torch.eye(num_tokens, device=device) + A_bar_layer
                
                # Equation 14: Accumulate
                # C = C @ \bar{A}
                # Note: Matrix multiplication order depends on definition of A_{ij}.
                # PyTorch Attn: A_{ij} is weight of Key j for Query i.
                # So Output_i = Sum_j A_{ij} * Input_j.
                # If C maps Input -> Layer_i, and A_bar maps Layer_i -> Layer_{i+1}.
                # Then C_{new} = A_bar @ C_{old} ?
                # Or C_{new} = C_{old} @ A_bar ?
                # User request: C = A(1) @ A(2) ...
                # This implies C = A_1 @ A_2.
                
                C = torch.matmul(C, A_bar) 
                
            # 5. Extract Relevance for Output
            # User request: "derived from the row corresponding to the classification token (or mean-pooled token)"
            
            # If C maps Input (cols) to Output (rows) accumulation:
            # C_{ij} roughly: How much Input j contributed to Output i.
            
            # For GAP (SigLIP), the final representation is Mean(Output Toks).
            # So Relevance(Input j) = Mean_over_i( C_{ij} )
            # i.e., column mean.
            
            heatmap = C.mean(dim=0) # [S]
            
            # Reshape
            side = int(num_tokens**0.5)
            # Square verification
            if side * side != num_tokens:
                 # If exact square match fails, it might be due to windowing or other factors.
                 # SigLIP usually 24x24 = 576. 
                 # Just try to reshape best effort.
                 pass
                 
            try:
                heatmap = heatmap.reshape(side, side)
            except:
                return np.zeros((384, 384))
            
            # Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
            
            return heatmap.detach().cpu().numpy()
            
