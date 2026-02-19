import torch
import torch.nn as nn

import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, SiglipVisionModel
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from agents.radiologist.classifier import MedGemmaVisionHead
from agents.radiologist.data import DiseaseClassificationDataset,CHEXBERT_CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 3
LR = 1e-4
TRAIN_JSONL = "../dataset/med/train_capped_clean.jsonl"
VAL_JSONL = "../dataset/med/val_capped_clean.jsonl"
IMAGE_ROOT = "../dataset/med/official_data_iccv_final"
FINAL_PATH= "../output/medsiglip_full_model.pt"
SAVE_PATH="../output/classifier_best_head.pth"
PROCESSOR_PATH="../output/processor"
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    set_seed(42)

    device = DEVICE
    print("[Init] Loading MedSigLIP...")

    processor = AutoImageProcessor.from_pretrained("google/medsiglip-448")

    vision_tower = SiglipVisionModel.from_pretrained(
        "google/medsiglip-448"
    ).to(device)

    # DO NOT call .eval() here

    model = MedGemmaVisionHead(
        num_classes=len(CHEXBERT_CLASSES),
        vision_model=vision_tower
    ).to(device)

    vision_core = model.vision_model.vision_model
    num_layers = len(vision_core.encoder.layers)

    print("Total encoder layers:", num_layers)

    # ---------------------------------------
    # Freeze entire backbone first
    # ---------------------------------------
    for p in vision_core.parameters():
        p.requires_grad = False

    # ---------------------------------------
    # Unfreeze last 2 transformer blocks
    # ---------------------------------------
    for i in range(num_layers - 2, num_layers):
        for p in vision_core.encoder.layers[i].parameters():
            p.requires_grad = True

    # Classifier always trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    # ---------------------------------------
    # Print trainable summary
    # ---------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n========== Model Parameter Summary ==========")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    print("=============================================\n")

    # ---------------------------------------
    # Optimizer (proper weight decay grouping)
    # ---------------------------------------
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "LayerNorm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-4
    )

    # ---------------------------------------
    # Datasets
    # ---------------------------------------
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

    MAX_TRAIN_SAMPLES = 70000
    MAX_VAL_SAMPLES = 1000

    train_dataset = Subset(
        train_dataset,
        random.sample(range(len(train_dataset)), MAX_TRAIN_SAMPLES)
    )

    val_dataset = Subset(
        val_dataset,
        random.sample(range(len(val_dataset)), MAX_VAL_SAMPLES)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ---------------------------------------
    # Scheduler
    # ---------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    # =======================================
    # TRAINING LOOP
    # =======================================
    for epoch in range(EPOCHS):

        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========\n")

        model.train()
        total_train_loss = 0.0

        train_bar = tqdm(train_loader)

        for batch in train_bar:

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)
            mask = batch["label_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):

                outputs = model(pixel_values)
                logits = outputs["logits"]

                loss = masked_bce_loss(logits, targets, mask)

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN detected. Stopping training.")
                return model, processor

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # ===================================
        # VALIDATION
        # ===================================
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:

                pixel_values = batch["pixel_values"].to(device)
                targets = batch["labels"].to(device)
                mask = batch["label_mask"].to(device)

                with torch.amp.autocast("cuda"):

                    outputs = model(pixel_values)
                    logits = outputs["logits"]

                    loss = masked_bce_loss(logits, targets, mask)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print("\n------------------------------------------")
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f}")
        print("------------------------------------------\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_head(SAVE_PATH)
            print("Saved best classifier head.\n")

    return model, processor

# MASKED BCE LOSS
def masked_bce_loss(logits, targets, mask):
    """
    Multi-label BCE with masking for uncertain labels.
    """
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss = loss_fn(logits, targets)
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1.0)

def main():

    print("=" * 60)
    print(" MedSigLIP Disease Classifier Training ")
    print("=" * 60)

    model, processor = train()

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": len(CHEXBERT_CLASSES),
    }, FINAL_PATH)

    processor.save_pretrained(PROCESSOR_PATH)

    print("\nTraining Complete.")
    print(f"Best classifier head saved at: {SAVE_PATH}")
    print(f"Full fine-tuned model saved at: {FINAL_PATH}")
    print(f"Processor saved at: {PROCESSOR_PATH}")

    return model, processor


if __name__ == "__main__":
    model, processor = main()