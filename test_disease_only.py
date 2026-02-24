import torch
import json
from agents.radiologist.model import analyze_disease

if __name__ == "__main__":
    print("Testing MedSigLIP disease classification and Chefer LRP heatmaps...")
    
    # Use the example image in the root directory
    image_path = "../dataset/med/official_data_iccv_final/files/p10/p10224633/s55588134/87b44387-4514b0f1-7c0dc5cd-ecfdbd1a-effef6ae.jpg"
    
    try:
        results = analyze_disease(image_path)
        
        print("\n--- Probabilities ---")
        for cls, prob in results.get("probabilities", {}).items():
            if prob > 0.1: # Show anything above 10%
                print(f"  {cls}: {prob:.4f}")
                
        print("\n--- Heatmaps Generated ---")
        for cls, path in results.get("heatmap_paths", {}).items():
            print(f"  {cls}: {path}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
