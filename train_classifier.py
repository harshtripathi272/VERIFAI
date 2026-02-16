
"""
Train Disease Classification Head

Trains the lightweight classification head on top of frozen MedSigLIP.
Target: 14 CheXbert disease labels.
Loss: BCEWithLogitsLoss (ignoring masked 'uncertain' labels).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from agents.radiologist.classifier import MedSigLIPClassifier
from agents.radiologist.data_start import DiseaseClassificationDataset
from transformers import AutoImageProcessor
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# Config
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 5
MODEL_NAME = "google/medsiglip-448" # Actually typical SigLIP is 224, MedSigLIP might be 384/448
# Ensure this matches config
IMAGE_SIZE = 384 # Default for many MedSigLIP variants

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Setup Data
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    # Paths (adjust as needed for user environment)
    # We'll rely on the paths user has in qlora-medgemma as a reference
    TRAIN_JSONL = "../dataset/med/train_capped_clean.jsonl"
    VAL_JSONL = "../dataset/med/val_capped_clean.jsonl"
    IMAGE_ROOT = "../dataset/med/official_data_iccv_final"
    
    # Use fallback if files don't exist (for verifying code logic)
    if not os.path.exists(TRAIN_JSONL):
        print(f"Warning: {TRAIN_JSONL} not found. Using local test.json if available or dummy.")
        # Logic to be robust
    
    train_ds = DiseaseClassificationDataset(
        jsonl_path=TRAIN_JSONL,
        image_root_dir=IMAGE_ROOT,
        image_processor=image_processor
    )
    
    val_ds = DiseaseClassificationDataset(
        jsonl_path=VAL_JSONL,
        image_root_dir=IMAGE_ROOT,
        image_processor=image_processor
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Setup Model
    model = MedSigLIPClassifier(MODEL_NAME, num_classes=14)
    model.to(device)
    model.train_head_only()
    
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)
    
    # 3. Training Loop
    best_auc = 0.0
    
    for epoch in range(EPOCHS):
        model.classifier.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["label_mask"].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(pixel_values)
            
            # BCE Loss with Masking
            # We explicitly handle masking
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            loss_raw = loss_fn(logits, labels)
            
            # Apply mask
            loss = (loss_raw * mask).sum() / (mask.sum() + 1e-6)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        # Validation
        val_metrics = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} Val AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.classifier.state_dict(), "classifier_head_best.pth")
            print("Saved best model.")

def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(device)
            logits = model(pixel_values)
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu())
            all_labels.append(batch["labels"])
            all_masks.append(batch["label_mask"])
            
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_masks = torch.cat(all_masks).numpy()
    
    # Compute AUC per class and average
    aucs = []
    for i in range(14):
        # Only compute if we have positive samples and valid (unmasked) samples
        valid_indices = all_masks[:, i] == 1
        y_true = all_labels[valid_indices, i]
        y_pred = all_preds[valid_indices, i]
        
        if len(np.unique(y_true)) > 1:
            aucs.append(roc_auc_score(y_true, y_pred))
            
    return {"auc": np.mean(aucs), "f1": 0.0} # F1 requires threshold tuning, AUC is safer

if __name__ == "__main__":
    train()
