
import os
import torch
from agents.radiologist.model import analyze_disease, generate_findings
from PIL import Image

# Use a sample chest X-ray from the dataset if available, or a dummy one
# Let's try to find one.
import glob

print("No sample images found. looking in current dir.")
sample_images = glob.glob("/data3/Pranshu/elephant_detection/med/VERIFAI/*.jpg")

if sample_images:
    image_path = sample_images[0]
    print(f"Testing with image: {image_path}")
    
    # 1. Test Analysis (MedSigLIP + Grad-CAM)
    print("\n--- Testing Disease Analysis (MedSigLIP) ---")
    results = analyze_disease(image_path)
    print("Probabilities:", results.get("probabilities"))
    print("Heatmap Paths:", results.get("heatmap_paths"))
    
    # Check if heatmaps exist
    for cls, path in results.get("heatmap_paths", {}).items():
        if os.path.exists(path):
            print(f"✅ Heatmap saved for {cls} at {path}")
        else:
            print(f"❌ Heatmap missing for {cls}")

    # 2. Test Report Generation (MedGemma)
    print("\n--- Testing Report Generation (MedGemma) ---")
    report = generate_findings(image_path, view="PA")
    print("Report:", report)

else:
    print("No sample images found. Creating a dummy image.")
    dummy_img = Image.new('RGB', (1024, 1024), color = 'gray')
    dummy_path = "test_cxr.jpg"
    dummy_img.save(dummy_path)
    
    try:
        results = analyze_disease(dummy_path)
        print("Probabilities:", results.get("probabilities"))
        print("Heatmap Paths:", results.get("heatmap_paths"))
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
