
"""
Disease Classification Dataset

Loads chest X-ray images and structured disease labels for multi-label classification.
Converts string-based structured findings (from CheXbert) into 14-dimensional binary vectors.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from typing import List, Dict, Tuple
from torchvision import transforms

# 14 CheXbert / CheXpert classes
CHEXBERT_CLASSES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

class DiseaseClassificationDataset(Dataset):
    """
    Dataset for multi-label disease classification using MedSigLIP.
    
    Each sample is a single image.
    Labels are 14-dim binary vectors.
    "Uncertain" labels are masked out (-100 or separate mask) during training.
    """
    
    def __init__(
        self,
        jsonl_path: str,
        image_root_dir: str,
        image_processor,
        transform=None,
        uncertain_policy: str = "mask" # "mask", "ones", "zeros"
    ):
        self.image_root_dir = image_root_dir
        self.image_processor = image_processor
        self.transform = transform
        self.uncertain_policy = uncertain_policy
        
        self.samples = []
        self._load_data(jsonl_path)
        
    def _load_data(self, jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                study = json.loads(line)
                
                # We train on individual images
                # Structured findings are at study level, so they apply to all images in study
                # (Weak supervision assumption)
                labels = study.get("structured_findings", {})
                
                for img_info in study["images"]:
                    self.samples.append({
                        "image_path": img_info["path"],
                        "view": img_info["view"],
                        "labels": labels
                    })
                    
        print(f"[DiseaseDataset] Loaded {len(self.samples)} images from {jsonl_path}")

    def __len__(self):
        return len(self.samples)
    
    def _encode_labels(self, findings: Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert structured findings to binary vector + mask.
        
        Returns:
            targets: [14] float tensor (0.0 or 1.0)
            mask: [14] float tensor (1.0 = valid, 0.0 = ignored/uncertain)
        """
        targets = torch.zeros(len(CHEXBERT_CLASSES), dtype=torch.float32)
        mask = torch.ones(len(CHEXBERT_CLASSES), dtype=torch.float32)
        
        for i, disease in enumerate(CHEXBERT_CLASSES):
            status = findings.get(disease, "not_mentioned")
            
            if status == "present":
                targets[i] = 1.0
            elif status == "absent" or status == "not_mentioned":
                targets[i] = 0.0
            elif status == "uncertain":
                if self.uncertain_policy == "mask":
                    mask[i] = 0.0 # Ignore this class for this sample
                    targets[i] = 0.0 # Value doesn't matter if mask is 0
                elif self.uncertain_policy == "ones":
                    targets[i] = 1.0
                elif self.uncertain_policy == "zeros":
                    targets[i] = 0.0
            else:
                # Fallback for unknown status
                targets[i] = 0.0
                
        return targets, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_root_dir, sample["image_path"])
        
        # Load Image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return dummy if failed (should handle better in prod)
            image = Image.new("RGB", (384, 384))
            
        # Process Image
        # If image_processor is passed (SigLIP processor), use it
        # It usually returns {'pixel_values': tensor}
        if self.image_processor:
            inputs = self.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.squeeze(0) # [3, H, W]
        else:
            # Manual transform fallback
            pixel_values = self.transform(image) if self.transform else transforms.ToTensor()(image)
            
        # Encode Labels
        targets, mask = self._encode_labels(sample["labels"])
        
        return {
            "pixel_values": pixel_values,
            "labels": targets,
            "label_mask": mask,
            "image_path": sample["image_path"]
        }
