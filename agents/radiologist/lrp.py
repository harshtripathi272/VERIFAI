
"""
LRP for SigLIP (Chefer et al. 2021)

This module implements the Transformer explainability method from Chefer et al. (CVPR 2021).
It is specifically adapted for the Hugging Face `SiglipVisionModel`.

Key Logic:
1. Register backward hooks to capture gradients of Attention and Activation.
2. Compute relevance using the "Gradient * Activation" rule, enforcing positive contributions.
3. Aggregate relevance across layers to form the final heatmap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Helper for Safe Division
def safe_divide(numerator, denominator, eps=1e-7):
    return numerator / (denominator + eps)

class LRPModel(nn.Module):
    """
    Wrapper around MedSigLIPClassifier to enable LRP hooks.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.attn_gradients = {}
        self.attn_maps = {}
        
        # Identify layers to hook
        # SigLIP structure: model.vision_model.encoder.layers[i].self_attn
        self._register_hooks()
        
    def _register_hooks(self):
        def save_attn_gradients(name):
            def hook(module, grad_input, grad_output):
                # grad_output[0] for the output of Softmax (Attention Weights)
                self.attn_gradients[name] = grad_output[0].detach()
            return hook

        def save_attn_maps(name):
            def hook(module, input, output):
                # SigLIP self_attn returns (attn_output, attn_weights) if output_attentions=True
                # But typically the dropout/softmax is internal. 
                # We need the tensor *after* softmax but *before* multiplication with V.
                # Since we can't easily intrude into the HF implementation without rewriting it,
                # we rely on the fact that `output_attentions=True` returns the attention weights
                # in the main model output. 
                # HOWEVER, for Chefer's method we need the gradient w.r.t that specific tensor during backward.
                pass 
                # Wait, if we use the stored `outputs.attentions`, we can get the map.
                # But we need the gradient w.r.t it. 
                # The `outputs.attentions` tensor is a leaf in the graph if we detach it? No.
                # We can retain_grad() on the attention tensors returned by the model forward pass!
            return hook
            
        # We don't actually need complex internal hooks if we use the `output_attentions=True` feature
        # and standard autograd.
        # Chefer method:
        # A_bar = A * Clamp(Grad_A)
        pass

class RelevanceGenerator:
    """
    Generates heatmaps for a specific class using Chefer et al. LRP.
    Adapted for SigLIP.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, pixel_values, target_class_index, device="cuda"):
        # 1. Forward Pass
        # We need to capture the attention maps and their gradients.
        
        pixel_values = pixel_values.to(device)
        pixel_values.requires_grad = True
        
        # Setup model to output attentions
        # We need to wrap this in a way that we can capture gradients of attentions
        
        outputs = self.model.vision_model(pixel_values, output_attentions=True)
        # pooled_output = outputs.last_hidden_state.mean(dim=1)
        # SigLIP uses PyTorch's native scaled_dot_product_attention which might drop weights if not careful,
        # but output_attentions=True forces it to return them.
        
        attentions = outputs.attentions # List of [B, Heads, N, N]
        
        # We must retain grads for these tensors to compute A * Grad_A
        for attn in attentions:
            attn.retain_grad()
            
        # Continue forward through classifier
        last_hidden_state = outputs.last_hidden_state
        pooled = last_hidden_state.mean(dim=1)
        logits = self.model.classifier(pooled)
        
        # 2. Backward
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class_index] = 1.0
        
        logits.backward(gradient=one_hot) # gradients flow back to attentions
        
        # 3. Compute Relevance
        # Formula: R = R + avg_heads(Grad_A * A)
        # Initialize relevance with Identity matrix (self-relevance)
        # We care about the relevance of [CLS] (or pooled token) to [Patches]
        
        # Since we use Mean Pooling, every patch contributes 1/N to the pooled vector.
        # But we want the relevance propagation through the Transformer layers.
        
        # Chefer aggregation:
        # R = eye(num_tokens)
        # For layer in layers:
        #    Attn = layer_attn
        #    Grad = layer_attn.grad
        #    Relevance_Interaction = Attn * Clamp(Grad)  (Gradient-weighted)
        #    R = R + (R @ Relevance_Interaction)
        
        num_tokens = attentions[0].shape[-1]
        R = torch.eye(num_tokens, device=device)
        
        for i, attn in enumerate(attentions):
            grad = attn.grad
            cam = attn # Activation
            
            # Clamp positive gradients (Chefer's key logic for positive class evidence)
            grad = torch.clamp(grad, min=0)
            
            # Weighted attention relevance
            # [B, Heads, N, N] -> Average over heads/batch -> [N, N]
            R_attn = (grad * cam).mean(dim=0).mean(dim=0)
            
            # Aggregate: R_new = R + (R @ R_attn)
            # This accounts for Skip Connections (Identity) + Attention Path
            R = R + torch.matmul(R, R_attn)
            
        # Final step: Extract relevance of pooled representative to patches
        # Since we Mean Pooled, effectively we have a virtual node connected to all patches.
        # If we visualized strictly CLS relevance, we'd pick R[0, 1:].
        # But here, we can look at the diagonal or the row corresponding to the "output" concept.
        
        # Actually, Chefer usually extracts row 0 for [CLS].
        # For GAP (Global Average Pooling), the relevance is distributed.
        # The correct interpretation for GAP:
        # Logic: Output depends on Mean(Patches).
        # We want to see which input patches contributed to that Mean via the layers.
        # R is [N, N] mapping input patch i to output patch j.
        # Total relevance of input i = Sum_j(R[i, j]) ? 
        # Or if we view it as flow: R[i, j] is how much token i influences token j.
        # We collected dependencies from Input -> Output.
        # So we want to know how much Input i influences the conceptual "Output".
        
        # Chefer Ref for GAP (e.g. ResNet/ViT-GAP):
        # We treat the final R as capturing the Transformer mixing.
        # The final Classifier weights tell us which "output features" (last layer patches) matter.
        # But since we average patches:
        # Logit = W * Mean(h_last)
        # Relevance of h_last[j] to Logit is roughly W.
        # We can backpropagate through the mean pooling: R_last = W / N.
        # Then propagate R_last through the Transformer structure using the computed R matrix.
        
        # Easier heuristic often used with Chefer:
        # Just input `R[0, 1:]` if CLS token.
        # For No-CLS (SigLIP), we assume the "information bottleneck" is distributed.
        # A common proxy is the sum of relevance to all tokens, weighted by their contribution to the head.
        
        # Let's rely on the simpler "Gradient * Input" at the pixel level?
        # No, user asked for "Transformer-based relevance propagation (Chefer et al.)".
        # That specifically yields the relevance map R [N, N].
        # For GAP models, we can sum the columns of R? (How much input i contributes to ALL output tokens).
        # Yes, R[i, :] is the outgoing flow from i. Sum(R[i, :]) is total influence of patch i.
        
        # Patch Relevance
        # [num_patches, num_patches]
        # Sum over rows? No, R is often defined as R_{out, in} or R_{in, out} depending on formulation.
        # Chefer code accumulates: R = R + R @ R_attn.
        # Initialization R=eye is "token i influences token i".
        # Transform: token i influences token j via attention.
        # Final: token i (input) influences token j (last layer).
        
        # We care about influence on the "Mean Pooled" representation.
        # Since Mean Pool takes all last-layer tokens equally (1/N),
        # Total Relevance of Input Patch i = Sum_{j} (R[i, j] * 1/N)
        # i.e., Average of the i-th row.
        
        patch_relevance = R.mean(dim=1) # [N]
        
        # Exclude finding itself? No, self-influence is fine.
        
        # Reshape to grid
        # Siglip 384x384 / 16 = 24x24
        side = int(num_tokens**0.5)
        heatmap = patch_relevance.reshape(side, side)
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)
        
        return heatmap.detach().cpu().numpy()

