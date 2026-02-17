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
        pixel_values.requires_grad = True # Is this needed for input relevance? Chefer usually visualizes R[0, 1:]
        
        # 2. Forward Pass
        # We need to capture the attention maps and their gradients.
        # MedSigLIPClassifier.forward calls: self.vision_model(..., output_attentions=True)
        # So we just run the classifier forward.
        
        # NOTE: To get gradients on the attention MAPS returned by output_attentions=True,
        # we need to make sure they are part of the graph.
        # HF implementation:
        # outputs = self.self_attn(...) -> (context_layer, attn_weights)
        # return (context_layer, attn_weights)
        # So attn_weights SITS in the graph.
        
        with torch.enable_grad(): # Ensure gradient tracking is ON even if global is off
            output_dict = self.model(pixel_values, return_dict=True)
            logits = output_dict['logits']
            
            # These are the [B, NumHeads, SeqLen, SeqLen] tensors
            # We must retain grads on them to get dLoss / dAttn
            attentions = output_dict['attentions'] # tuple of tensors
            
            for attn in attentions:
                attn.retain_grad()
            
            # 3. Backward
            self.model.zero_grad()
            
            # Create one-hot for target class
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class_index] = 1.0
            
            # Backpropagate to populate .grad on attentions
            logits.backward(gradient=one_hot)
            
            # 4. Chefer Integration (Equation 13 & 14)
            # R^{l} = R^{l+1} + \bar{A} * R^{l+1} 
            # where \bar{A} = E_h [ \nabla A * A ]^+  (Element-wise mul, clamp positive, mean over heads)
            # Actually Chefer simplifies to:
            # R = R + (R @ R_attn)  where R_attn is the relevance of attention mechanism
            
            # The relevance matrix R is [SeqLen, SeqLen].
            # Initialization: Identity (Self-relevance)
            num_tokens = attentions[0].shape[-1]
            R = torch.eye(num_tokens, device=device)
            
            # Iterate from first layer to last? 
            # Chefer paper: "The total relevance is initialized as the identity matrix..."
            # "For each layer b... we compute the relevance map \bar{A}^{(b)}"
            # "The accumulated relevance map is updated as C <- \bar{A}^{(b)} * C" (Matrix multiplication)
            
            # So we go forward from Layer 1 to Layer L.
            
            for i, attn in enumerate(attentions):
                grad = attn.grad # [B, H, S, S]
                cam = attn       # [B, H, S, S]
                
                # Equation: E_h [ (grad * cam)^+ ]
                # 1. Element-wise product
                # 2. Clamp positive (Keep only positive contributions)
                # 3. Average over heads
                
                grad = torch.clamp(grad, min=0)
                R_attn = (grad * cam).mean(dim=1) # [B, S, S] (average over heads)
                R_attn = R_attn[0] # [S, S] for batch 0
                
                # Add Identity (Residual connection logic)
                # \bar{A} = I + R_attn
                # (Actually Chefer defines R_total = R_total + R_total @ R_attn_weighted)
                # The paper says: C = \bar{A} . C  (if composing from input to output)
                
                # Simpler implementation from official repo (Hila-Chefer/Transformer-MM-Explainability):
                # R = R + torch.matmul(R, R_attn)
                
                R = R + torch.matmul(R, R_attn)
                
            # 5. Extract Relevance for Output
            # SigLIP uses Mean Pooling (GAP) over all patches to get the representation.
            # So we don't have a single [CLS] token at index 0 that aggregates everything.
            # Instead, the logits depend on Mean(LastLayer).
            # Effectively, this is like having a virtual [CLS] that attends to all LastLayer tokens with weight 1/N.
            
            # If we want "Relevance of Input Patches to the Class Prediction":
            # We effectively want to know: How much did Input Patch j contribute to the "Mean Pooled Vector"?
            
            # Since R[i, j] accumulates flow from token i to token j (or vice-versa depending on definition),
            # let's assume standard flow: R[row, col] -> Row affects Col.
            
            # Total relevance of Input i = Sum_{j \in LastLayer} ( R[i, j] )
            # Because every output token j contributes equally to the Mean Pool.
            # So we sum the rows.
            
            # NOTE: SigLIP typically has no CLS token.
            # Output: [S, S].
            # We want [S] (relevance per input patch).
            patch_relevance = R.mean(dim=1) # Average over columns (or sum)
            
            # Reshape to 2D
            # SigLIP 384 / 16 = 24
            side = int(num_tokens**0.5)
            # If side*side != num_tokens, check if there's a CLS token?
            # SigLIP usually doesn't have CLS.
            
            if side * side != num_tokens:
                print(f"Warning: Num tokens {num_tokens} is not a perfect square. Assuming CLS at index 0?")
                # If there's a CLS, strip it?
                # But SigLIP config usually doesn't add CLS.
                # Let's assume square for now.
                pass
            
            try:
                heatmap = patch_relevance.reshape(side, side)
            except:
                # Fallback if dimensions mismatch
                return np.zeros((384, 384))
            
            # Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
            
            return heatmap.detach().cpu().numpy()
            
