import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoImageProcessor
from peft import PeftModel
from PIL import Image
import numpy as np

from agents.radiologist.classifier import MedGemmaVisionHead
from agents.radiologist.lrp import RelevanceGenerator
from agents.radiologist.data import (
    DiseaseClassificationDataset,
    CHEXBERT_CLASSES
)
from app.config import settings

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4

TRAIN_JSONL = "../dataset/med/train_capped_clean.jsonl"
VAL_JSONL = "../dataset/med/val_capped_clean.jsonl"
IMAGE_ROOT = "../dataset/med/official_data_iccv_final"

SAVE_PATH = "../output/classifier_head_best.pth"
HEATMAP_DIR = "../output/heatmaps_val"

os.makedirs(HEATMAP_DIR, exist_ok=True)

# LOAD MEDGEMMA + SHARE VISION TOWER


def load_shared_vision():
    print("[Init] Loading MedGemma base (no LoRA)...")

    llm = AutoModelForImageTextToText.from_pretrained(
        settings.MEDGEMMA_4B_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        token=settings.HUGGINGFACE_TOKEN
    )

    vision_tower = llm.model.vision_tower
    vision_tower.eval()

    for p in vision_tower.parameters():
        p.requires_grad = False

    return vision_tower

# MASKED BCE LOSS
def masked_bce_loss(logits, targets, mask):
    """
    Multi-label BCE with masking for uncertain labels.
    """
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss_fn(logits, targets)
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1.0)

# TRAINING LOOP


def train():

    processor = AutoImageProcessor.from_pretrained(settings.MEDSIGLIP_MODEL)
    vision_tower = load_shared_vision()

    model = MedGemmaVisionHead(
        num_classes=len(CHEXBERT_CLASSES),
        vision_model=vision_tower
    )

    model.train_head_only()
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)

    # Real Dataset Integration
    train_dataset = DiseaseClassificationDataset(
        jsonl_path=TRAIN_JSONL,
        image_root_dir=IMAGE_ROOT,
        image_processor=processor,
        uncertain_policy="mask"
    )

    val_dataset = DiseaseClassificationDataset(
        jsonl_path=VAL_JSONL,
        image_root_dir=IMAGE_ROOT,
        image_processor=processor,
        uncertain_policy="mask"
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            targets = batch["labels"].to(DEVICE)
            mask = batch["label_mask"].to(DEVICE)

            outputs = model(pixel_values)
            logits = outputs["logits"]

            loss = masked_bce_loss(logits, targets, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch["pixel_values"].to(DEVICE)
                targets = batch["labels"].to(DEVICE)
                mask = batch["label_mask"].to(DEVICE)

                outputs = model(pixel_values)
                logits = outputs["logits"]

                loss = masked_bce_loss(logits, targets, mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_head(SAVE_PATH)
            print("Saved best classifier head.")

    return model, processor


# ============================================
# HEATMAP GENERATION
# ============================================

def generate_heatmaps(model, processor, dataset):

    print("[Heatmap] Generating heatmaps on validation set...")
    model.enable_gradients_for_lrp()
    model.eval()

    lrp_generator = RelevanceGenerator(model)

    for sample in tqdm(dataset.samples[:100]):  # limit for speed
        image_path = os.path.join(IMAGE_ROOT, sample["image_path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            continue

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(DEVICE)

        with torch.no_grad():
            logits = model(pixel_values)
            probs = torch.sigmoid(logits).squeeze(0)

        for i, (cls_name, prob) in enumerate(zip(CHEXBERT_CLASSES, probs)):
            if prob > 0.5:
                heatmap = lrp_generator.generate(pixel_values, i, device=DEVICE)

                hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                hm_uint8 = (hm_norm * 255).astype(np.uint8)
                hm_img = Image.fromarray(hm_uint8, mode='L')
                hm_resized = hm_img.resize(image.size, resample=Image.BILINEAR)

                save_path = os.path.join(
                    HEATMAP_DIR,
                    f"{os.path.basename(image_path).split('.')[0]}_{cls_name}.png"
                )
                hm_resized.save(save_path)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    model, processor = train()

    # Load best head before heatmap generation
    model.load_head(SAVE_PATH)

    val_dataset = DiseaseClassificationDataset(
        jsonl_path=VAL_JSONL,
        image_root_dir=IMAGE_ROOT,
        image_processor=processor
    )

    generate_heatmaps(model, processor, val_dataset)

    print("Training + Heatmap generation complete.")
