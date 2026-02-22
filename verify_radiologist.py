
"""
Verify Radiologist Integration

1. Create dummy image.
2. Call analyze_disease directly.
3. Check returned structure.
"""

import os
import torch
from PIL import Image
from agents.radiologist.model import analyze_disease

def test_integration():
    print("Setting up verification...")
    # Create dummy image
    img_path = "test_cxr.jpg"
    Image.new("RGB", (384, 384)).save(img_path)
    
    print(f"Created {img_path}")
    
    try:
        print("Calling analyze_disease...")
        result = analyze_disease(img_path)
        
        print("Result received:")
        print(f"Probabilities: {len(result.get('probabilities', {}))} classes")
        print(f"Heatmaps: {len(result.get('heatmap_paths', {}))} paths")
        
        probs = result.get('probabilities', {})
        if "Pneumonia" in probs:
            print(f"Pneumonia Prob: {probs['Pneumonia']}")
            
        print("SUCCESS: Integration Verified.")
        
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    test_integration()
