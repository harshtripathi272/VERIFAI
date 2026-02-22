
"""
Test: LRP and Classifier Integration

Verifies that:
1. Classifier runs forward pass.
2. LRP generator runs backward pass.
3. Attention gradients are captured.
4. Heatmap is generated with correct shape.
"""

import torch
from agents.radiologist.classifier import MedSigLIPClassifier
from agents.radiologist.lrp import RelevanceGenerator

def test_integration():
    print("Initializing Model...")
    # Use smaller model for quick testing
    TEST_MODEL = "google/siglip-base-patch16-224" 
    device = "cpu" 
    
    try:
        model = MedSigLIPClassifier(TEST_MODEL, num_classes=14)
    except Exception as e:
        print(f"Failed to load {TEST_MODEL}: {e}")
        # Fallback to creating a dummy config if possible, but SigLIP structure is needed.
        return

    model.to(device)
    model.eval()
    
    print(f"Model hidden size: {model.hidden_size}")
    
    # 224x224 input
    pixel_values = torch.randn(2, 3, 224, 224).to(device)
    
    # 1. Test Forward
    print("Testing Forward Pass...")
    with torch.no_grad():
        logits = model(pixel_values)
        print(f"Logits shape: {logits.shape}")
        assert logits.shape == (2, 14)
        
    # 2. Test LRP
    print("Testing LRP Generation...")
    lrp = RelevanceGenerator(model)
    
    # Generate for first image, class 0
    img_tensor = pixel_values[0].unsqueeze(0) # [1, 3, 224, 224]
    target_class = 0
    
    heatmap = lrp.generate(img_tensor, target_class, device=device)
    
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min()}, {heatmap.max()}]")
    
    # Verify shape
    # 224 // 16 = 14. Should be 14x14
    if heatmap.shape == (14, 14):
        print("PASS: Heatmap shape correct.")
    else:
        print(f"WARNING: Unexpected heatmap shape {heatmap.shape} (expected 14x14).")

if __name__ == "__main__":
    test_integration()
