
"""
LRP for Transformers (Chefer et al., CVPR 2021)

Implements relevance propagation for Transformer models (SigLIP/MedSigLIP).
Key component: LRP for Self-Attention handling Softmax and "uncoupling" scoring.

Reference: "Transformer Interpretability Beyond Attention Visualization" (Chefer et al. 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LRP:
    """
    LRP engine for Chefer et al. method.
    Attaches hooks to the model to capture gradients and activations during backward pass.
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.grad_store = {}
        self.act_store = {}
        self.relevance_store = {}

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(file_name):
            def hook(module, input, output):
                self.act_store[file_name] = output
            return hook

        def backward_hook(file_name):
            def hook(module, grad_input, grad_output):
                self.grad_store[file_name] = grad_output[0]
            return hook
            
        # We need to traverse the model and attach hooks to specific layers
        # For SigLIP, layers are usually standard nn modules.
        # We need to identify Attention layers specifically for the custom rule.
        pass # To be implemented in specific generator class

    def safe_divide(self, numerator, denominator, eps=1e-6):
        return numerator / (denominator + eps)

    @staticmethod
    def relevance_linear(layer, R, input_tensor):
        """
        LRP-gamma/epsilon for Linear layers.
        z = w * x + b
        R_in = (z_in / z_out) * R_out
        """
        # Simplified LRP-0 / LRP-epsilon implementation for now
        # Chefer uses a specific gradient-based formulation: R = R * (grad * x)
        # We will use the gradient-based formulation exclusively as it handles complex graphs better.
        pass

# ─── Chefer Rule Implementations ──────────────────────────────────────

def apply_self_attention_rules(attn_layer, R_output, inputs, grads):
    """
    Compute relevance for Self-Attention layer.
    
    R_total = R_output
    Ref: Eq (5-9) in Chefer et al.
    
    The critical insight is:
    A = softmax(Q * K^T / sqrt(d))
    Relevance propagates through Softmax as:
    R_A = R_out * (Grad_A * A)  (Gradient x Activation)
    
    But we strictly enable positive contributions:
    Grad_A positive = Clamp(Grad_A)
    """
    # This requires access to internal Q, K, V, and Attention Map A during the forward pass.
    # Standard PyTorch hooks might be insufficient to get intermediate A inside the block
    # unless we rewrite the block or capture it via output_attentions=True.
    
    # Fortunately, SiglipVisionModel outputs attentions!
    pass

class RelevanceGenerator:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def generate_relevance(self, pixel_values, target_class_index):
        """
        Generates relevance map for a specific class.
        
        1. Forward pass (save attentions and gradients).
        2. Zero grads.
        3. Backward pass from logit[target_class].
        4. Compute aggregated relevance using Chefer rules.
        """
        self.model.zero_grad()
        
        # 1. Forward
        # Enable gradients for inputs to allow flow back to image
        pixel_values.requires_grad = True
        
        outputs = self.model(pixel_values)
        logits = outputs["logits"]
        
        # 2. Target
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class_index] = 1.0
        
        # 3. Backward
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # 4. Compute LRP
        # Chefer method accumulates relevance from the output to input.
        # But wait, their official code uses the "Gradient * Activation" trick for layers
        # and a special rule for Attention.
        
        # R = R_classifier
        # For Transformer layers:
        #   R_transformer = R + (Grad_Attn * Attn) + (Grad_MLP * MLP)
        
        # Actually, Chefer 2021 proposes:
        # A_bar = Avg(Grad * Attn) across heads => "Weighted Attention Relevance"
        # They accumulate this A_bar across layers.
        # Final Map = Transformer_LRP(Input)
        
        # Let's implement the core aggregation:
        # Relevancy map initialization
        # R = torch.eye(num_patches)
        
        attentions = outputs["attentions"] # Tuple of [B, NumHeads, SeqLen, SeqLen]
        gradients = [] 
        
        # We need gradients w.r.t attentions. 
        # Since we just ran .backward(), we can't easily get grad w.r.t intermediate tensor 
        # UNLESS we retained it or used a hook.
        
        return None

# The robust implementation requires hooks. Let's write the full class below.
