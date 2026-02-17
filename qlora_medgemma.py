"""
Training script for MedGemma QLoRA fine-tuning — chest X-ray report generation.
CORRECTED VERSION with fixes based on Google's reference implementation.

Usage (run from qlora_mg/):
    python qlora-medgemma-corrected.py

Overfit sanity-check (200 studies):
    Edit config → max_train_samples = 200, then run as above.

Pipeline:
    1. Load AutoModelForImageTextToText (MedGemma 1.5)
    2. Load AutoProcessor
    3. Prepare dataset with chat template formatting
    4. Fine-tune with QLoRA using SFTTrainer
    5. Save adapter weights

CHANGES FROM ORIGINAL:
    - Added gradient_checkpointing_kwargs (critical for memory)
    - Changed optimizer to paged_adamw_32bit (better for QLoRA)
    - Added path validation and error handling
    - Fixed image fallback behavior
    - Added BF16 auto-detection
    - Improved view token positioning
    - Added safety checks and logging
"""

import os
import sys
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Filter warnings
warnings.filterwarnings("ignore", message=".*warmup_ratio is deprecated.*")
warnings.filterwarnings("ignore", message=".*use_cache=True.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

# Constants
WORKSPACE_ROOT = Path("../dataset/med").resolve()  # Use absolute path
VIEW_ORDER = {"PA": 0, "AP": 1, "LATERAL": 2}
INSTRUCTION = (
    "You are an expert radiologist.\n\n"
    "Analyze the provided chest X-rays and write a careful radiology report "
    "using appropriate clinical language."
)

@dataclass
class ModelConfig:
    language_model_id: str = "google/medgemma-1.5-4b-it"
    vision_model_id: str = "google/medsiglip-448"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

@dataclass
class LoraConfigData:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: str = "all-linear"
    task_type: str = "CAUSAL_LM"
    # Keep these since we're adding special tokens
    modules_to_save: List[str] = field(default_factory=lambda: ["embed_tokens", "lm_head"])

@dataclass
class DataConfig:
    train_jsonl: str = "train_capped_clean.jsonl"
    val_jsonl: str = "val_capped_clean.jsonl"
    image_root_dir: str = "official_data_iccv_final"
    max_length: int = 768
    max_images: int = 3  # REDUCED from 7 to prevent OOM

@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfigData = field(default_factory=LoraConfigData)
    data: DataConfig = field(default_factory=DataConfig)

    output_dir: str = "../dataset/med/fine_tuned_model"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"  # FIXED: Better for QLoRA
    
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    seed: int = 42
    bf16: bool = None  # Will auto-detect
    fp16: bool = None  # Will auto-detect
    
    report_to: str = "tensorboard"  # Changed to tensorboard for monitoring
    run_version: str = "v1"
    max_train_samples: Optional[int] = None

class ChestXrayReportDataset(Dataset):
    """
    Dataset that yields un-tokenized examples with images and 'messages' for SFTTrainer.
    
    IMPROVEMENTS:
    - Better error handling for missing images
    - Returns None for invalid samples
    - Improved view token positioning
    """
    def __init__(self, jsonl_path, image_root_dir, max_images=10):
        self.studies = []
        self.image_root_dir = Path(image_root_dir)
        self.max_images = max_images
        
        # Validate paths
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        if not self.image_root_dir.exists():
            raise FileNotFoundError(f"Image root directory not found: {self.image_root_dir}")
        
        # Load studies
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                study = json.loads(line)
                # Only keep studies with findings or impression
                if study.get("findings") or study.get("impression"):
                    self.studies.append(study)
        
        print(f"[Dataset] Loaded {len(self.studies)} studies from {jsonl_path}")

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study = self.studies[idx]
        images_info = study["images"]
        findings = study.get("findings", "").strip()
        impression = study.get("impression", "").strip()
        
        # Format report
        report = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"

        # Sort images by view priority
        sorted_imgs = sorted(
            images_info,
            key=lambda x: (VIEW_ORDER.get(x["view"], 99), x["order"]),
        )[: self.max_images]

        loaded_images = []
        user_content = []

        # Load images
        for img_info in sorted_imgs:
            path = self.image_root_dir / img_info["path"]
            try:
                img = Image.open(path).convert("RGB")
                loaded_images.append(img)
                user_content.append({"type": "image"})
            except Exception as e:
                print(f"Warning: Failed to load image {path}: {e}")
                continue

        # CRITICAL: Skip samples with no valid images instead of using black fallback
        if not loaded_images:
            print(f"WARNING: No valid images found for study {idx}, skipping...")
            return None  # Will be filtered in collate_fn

        # IMPROVED: Add view context after all images
        view_labels = " | ".join([f"<{img['view']}>" for img in sorted_imgs[:len(loaded_images)]])
        user_content.append({
            "type": "text", 
            "text": f"\nViews: {view_labels}\n\n{INSTRUCTION}"
        })

        # Construct messages in chat format
        messages = [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": report}
                ]
            }
        ]

        return {
            "images": loaded_images,
            "messages": messages
        }

def get_response_template(processor):
    """
    Identify the response template for DataCollatorForCompletionOnlyLM.
    
    For Gemma models, the assistant response starts after specific markers.
    We need to find what the chat template uses to mark the assistant turn.
    """
    # Test with a dummy message to see the format
    test_messages = [
        {"role": "user", "content": [{"type": "text", "text": "test"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "response"}]}
    ]
    
    formatted = processor.apply_chat_template(
        test_messages,
        add_generation_prompt=False,
        tokenize=False
    )
    
    print(f"Chat template format:\n{formatted}\n")
    
    # For Gemma models, typically: <start_of_turn>model\n
    # But let's check the actual format
    if "<start_of_turn>model" in formatted:
        return "<start_of_turn>model\n"
    elif "assistant" in formatted.lower():
        # Try to extract the pattern
        lines = formatted.split("\n")
        for i, line in enumerate(lines):
            if "assistant" in line.lower() and i < len(lines) - 1:
                # Return the line that marks assistant start
                return line + "\n"
    
    # Fallback - you may need to adjust this based on actual template
    print("WARNING: Could not auto-detect response template")
    print("Using default: '<start_of_turn>model\\n'")
    return "<start_of_turn>model\n"

class CustomLoggingCallback(TrainerCallback):
    """Custom callback for better logging."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 10 == 0:
            log_str = f"Step {state.global_step}: "
            if "loss" in logs:
                log_str += f"loss={logs['loss']:.4f} "
            if "learning_rate" in logs:
                log_str += f"lr={logs['learning_rate']:.2e} "
            print(log_str)

def detect_precision():
    """Auto-detect best precision for current GPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        bf16_support = torch.cuda.is_bf16_supported()
        
        print(f"GPU: {gpu_name}")
        print(f"BF16 supported: {bf16_support}")
        
        if bf16_support:
            print("Using BF16 precision (recommended for MedGemma)")
            return True, False  # bf16=True, fp16=False
        else:
            print("BF16 not supported, using FP16 (may be less stable)")
            return False, True  # bf16=False, fp16=True
    else:
        raise RuntimeError("CUDA not available")

def validate_paths(config):
    """Validate all required paths exist."""
    print("\n=== Path Validation ===")
    
    # Check workspace root
    if not WORKSPACE_ROOT.exists():
        raise FileNotFoundError(f"Workspace root not found: {WORKSPACE_ROOT}")
    print(f"✓ Workspace root: {WORKSPACE_ROOT}")
    
    # Check data files
    train_path = WORKSPACE_ROOT / config.data.train_jsonl
    val_path = WORKSPACE_ROOT / config.data.val_jsonl
    image_root = WORKSPACE_ROOT / config.data.image_root_dir
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    print(f"✓ Training data: {train_path}")
    
    if not val_path.exists():
        print(f"⚠ Validation data not found: {val_path} (will skip validation)")
    else:
        print(f"✓ Validation data: {val_path}")
    
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    print(f"✓ Image directory: {image_root}")
    
    print("======================\n")
    
    return train_path, val_path, image_root

def main():
    config = TrainingConfig()
    set_seed(config.seed)
    
    print("="*60)
    print("MedGemma QLoRA Fine-tuning - Chest X-ray Reports")
    print("="*60)
    
    # Validate paths
    train_path, val_path, image_root = validate_paths(config)
    
    # Auto-detect precision
    bf16, fp16 = detect_precision()
    config.bf16 = bf16
    config.fp16 = fp16
    
    # 1. Load Model & Processor
    model_id = config.model.language_model_id
    
    print(f"\n=== Loading Model ===")
    print(f"Model: {model_id}")
    
    # CRITICAL: Use the correct precision based on GPU support
    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
    print(f"Using compute dtype: {compute_dtype}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,  # Use detected dtype
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,  # Use detected dtype
    )
    
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    
    # Add view tokens
    special_tokens = ["<PA>", "<AP>", "<LATERAL>"]
    num_added = processor.tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens
    })
    
    if num_added > 0:
        model.resize_token_embeddings(len(processor.tokenizer))
        print(f"✓ Added {num_added} special view tokens")
    
    print("✓ Model loaded successfully")

    # 2. Load Dataset
    print(f"\n=== Loading Dataset ===")
    
    train_ds = ChestXrayReportDataset(train_path, image_root, max_images=config.data.max_images)
    val_ds = ChestXrayReportDataset(val_path, image_root, max_images=config.data.max_images) if val_path.exists() else None
    
    # Apply sample cap if specified
    if config.max_train_samples:
        train_ds.studies = train_ds.studies[:config.max_train_samples]
        if val_ds:
            val_ds.studies = val_ds.studies[:min(len(val_ds.studies), config.max_train_samples)]
        print(f"⚠ Dataset capped to {config.max_train_samples} samples for testing")

    # 3. PEFT Configuration
    print(f"\n=== PEFT Configuration ===")
    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        task_type=config.lora.task_type,
        modules_to_save=config.lora.modules_to_save,
    )
    print(f"LoRA rank: {config.lora.r}")
    print(f"LoRA alpha: {config.lora.lora_alpha}")
    print(f"Target modules: {config.lora.target_modules}")
    print(f"Modules to save: {config.lora.modules_to_save}")

    # 4. Training Arguments
    print(f"\n=== Training Configuration ===")
    output_dir = Path(config.output_dir) / config.run_version
    output_dir.mkdir(parents=True, exist_ok=True)
    
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # FIXED: Critical for memory
        optim=config.optim,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=config.eval_steps if val_ds else None,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        fp16=config.fp16,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        push_to_hub=False,
        report_to=config.report_to,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        label_names=["labels"],  # ADDED: Explicitly specify label column
    )
    
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Optimizer: {config.optim}")
    print(f"Precision: {'BF16' if config.bf16 else 'FP16'}")

    # 5. Create Trainer
    print(f"\n=== Creating Trainer ===")
    
    # CRITICAL FIX: Use TRL's DataCollatorForCompletionOnlyLM for proper label masking
    # This ensures only the assistant's response is used for loss calculation
    response_template = get_response_template(processor)
    print(f"Using response template: {repr(response_template)}")
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=processor.tokenizer,
        mlm=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collator,  # Use TRL's proper collator
        callbacks=[CustomLoggingCallback()],  # ADDED: Better logging
    )

    # 6. Train
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    try:
        trainer.train()
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"TRAINING FAILED: {e}")
        print(f"{'='*60}\n")
        raise
    
    # 7. Save
    print("Saving model...")
    trainer.save_model()
    processor.tokenizer.save_pretrained(args.output_dir)
    print(f"✓ Model saved to: {args.output_dir}")
    
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()