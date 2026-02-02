"""
Visual Evidence Tools

Grad-CAM and saliency map generation.
"""

from typing import Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def generate_gradcam(
    model: Any,
    input_tensor: Any,
    target_layer: str
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for visual explanation.
    
    Args:
        model: Vision model (MedSigLIP)
        input_tensor: Preprocessed input image tensor
        target_layer: Name of layer to compute gradients for
        
    Returns:
        Heatmap as numpy array
    """
    if not TORCH_AVAILABLE:
        return _mock_gradcam()
    
    # TODO: Implement actual Grad-CAM computation
    # This requires hooking into model layers and computing gradients
    return _mock_gradcam()


def _mock_gradcam() -> np.ndarray:
    """Generate mock heatmap for demo."""
    # Create gaussian blob for visualization
    x = np.linspace(-1, 1, 224)
    y = np.linspace(-1, 1, 224)
    xx, yy = np.meshgrid(x, y)
    
    # Center blob slightly off-center (like RLL region)
    blob = np.exp(-((xx - 0.3)**2 + (yy + 0.3)**2) / 0.3)
    
    return blob


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image (H, W, 3)
        heatmap: Heatmap (H, W)
        alpha: Blend factor
        
    Returns:
        Blended image
    """
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Apply colormap (red-yellow)
    colormap = np.zeros((*heatmap.shape, 3))
    colormap[..., 0] = heatmap  # Red channel
    colormap[..., 1] = heatmap * 0.5  # Green channel (yellow tint)
    
    # Blend
    blended = (1 - alpha) * image / 255.0 + alpha * colormap
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    return blended
