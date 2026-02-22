import torch
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from agents.radiologist import model

def verify():
    print("Loading models...")
    model._load_models()
    
    print("\nVerifying vision tower sharing...")
    vlm_tower = model._llm.model.vision_tower
    
    # Check if classifier exists
    if not hasattr(model, '_classifier_model') or model._classifier_model is None:
        print("Classifier not initialized!")
        exit(1)
        
    classifier_tower = model._classifier_model.vision_model
    
    print(f"VLM Tower: {vlm_tower}")
    print(f"Classifier Tower: {classifier_tower}")
    
    if vlm_tower is classifier_tower:
        print("SUCCESS: Vision towers are the same object.")
    else:
        print("FAILURE: Vision towers are different objects!")
        exit(1)

if __name__ == "__main__":
    verify()
