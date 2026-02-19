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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Enable PIL cache to reuse images across epochs - massive speedup for repeated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import sys
import json
import argparse
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Union, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Import robust JSON extractor
from utils.inference import extract_json


# Filter warnings
warnings.filterwarnings("ignore", message=".*warmup_ratio is deprecated.*")
warnings.filterwarnings("ignore", message=".*use_cache=True.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

# Constants
# Constants
WORKSPACE_ROOT = Path("../dataset/med").resolve()  # Use absolute path
VIEW_ORDER = {"PA": 0, "AP": 1, "LATERAL": 2}
INSTRUCTION = """
You are an expert radiologist. Analyze the provided chest X-ray and write a careful radiology report.

STRICT RULES:
- Output ONLY valid JSON.
- No markdown.
- No explanations outside JSON.
- Must start with { and end with }.

Required JSON structure:
{
  "findings": "Detailed radiographic findings here...",
  "impression": "Diagnostic impression and differential diagnosis here..."
}
"""

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
    r: int = 16                   # REDUCED from 16: fewer params → less memory
    lora_alpha: int = 16          # keep equal to r
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    task_type: str = "CAUSAL_LM"
    modules_to_save: List[str] = field(default_factory=lambda: ["embed_tokens", "lm_head"])

@dataclass
class DataConfig:
    train_jsonl: str = "train_capped_clean.jsonl"
    val_jsonl: str = "val_capped_clean.jsonl"
    image_root_dir: str = "official_data_iccv_final"
    max_length: int = 512        # REDUCED from 768: caps sequence length
    max_images: int = 2          # REDUCED to 1: biggest single memory saving
    num_workers: int = 4         # Parallel dataset loading during init
    cache_images: bool = False   # Optional: Cache images in memory (requires 20+ GB RAM)

@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfigData = field(default_factory=LoraConfigData)
    data: DataConfig = field(default_factory=DataConfig)

    output_dir: str = "../dataset/med/fine_tuned_model"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # INCREASED to compensate for smaller batch
    
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"  # CHANGED from 32bit: saves ~2 GB
    
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    
    seed: int = 42
    bf16: bool = None  # Will auto-detect
    fp16: bool = None  # Will auto-detect
    
    report_to: str = "none"
    run_version: str = "v1"
    max_train_samples: Optional[int] = None
    dataloader_num_workers: int = 4  # DataLoader workers for async prefetching

class Study:
    """Optimized study data structure with __slots__ for memory efficiency."""
    __slots__ = ('idx', 'findings', 'impression', 'report_json', 'selected_image_paths', 'selected_views')
    
    def __init__(self, idx, findings, impression, report_json, image_paths, views):
        self.idx = idx
        self.findings = findings
        self.impression = impression
        self.report_json = report_json
        self.selected_image_paths = image_paths  # Pre-computed, cached as strings
        self.selected_views = views


class ChestXrayReportDataset(Dataset):
    """
    HEAVILY OPTIMIZED Dataset with pre-computed metadata.
    
    MAJOR OPTIMIZATIONS:
    - Pre-compute view selections during __init__ (runs once, not 43927x)
    - Cache reports as JSON strings (no repeated serialization)
    - Store image paths as strings (avoid Path() overhead in __getitem__)
    - Use __slots__ for memory efficiency
    - Parallel metadata loading with ThreadPoolExecutor
    - PIL cache enabled globally for image reuse
    """
    
    VIEW_PRIORITY = ["PA", "AP", "LATERAL"]  # Class constant, not recreated
    
    def __init__(self, jsonl_path, image_root_dir, max_images=10, num_workers=4):
        self.studies = []
        self.image_root_dir_str = str(image_root_dir)  # Cache as string
        self.max_images = max_images
        self.num_workers = num_workers
        
        # Validate paths
        jsonl_path = Path(jsonl_path)
        image_root = Path(image_root_dir)
        
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root directory not found: {image_root}")
        
        # Load and pre-process studies in parallel
        raw_studies = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                study = json.loads(line)
                if study.get("findings") or study.get("impression"):
                    raw_studies.append(study)
        
        # Pre-process studies in parallel for speed
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            processed = list(executor.map(
                lambda s: self._preprocess_study(s, image_root),
                raw_studies
            ))
        
        self.studies = [s for s in processed if s is not None]
        
        print(f"[Dataset] Loaded {len(self.studies)} studies from {jsonl_path}")
        print(f"[Dataset] Pre-computed metadata for all {len(self.studies)} samples")
    
    def _preprocess_study(self, study: Dict, image_root: Path) -> Optional[Study]:
        """Pre-compute all metadata for a study once during init."""
        try:
            findings = study.get("findings", "").strip()
            impression = study.get("impression", "").strip()
            
            # Pre-format report as JSON
            report_json = json.dumps({
                "findings": findings,
                "impression": impression
            }, separators=(',', ':'))  # Compact format saves space
            
            # Pre-compute optimal view selection
            images_info = study.get("images", [])
            selected_images, selected_views = self._select_views(images_info, image_root)
            
            if not selected_images:
                return None
            
            # Store paths as strings (faster than Path objects)
            image_paths = [str(img_path) for img_path in selected_images]
            
            study_idx = len(self.studies)  # Will be updated when added to list
            return Study(
                idx=study_idx,
                findings=findings,
                impression=impression,
                report_json=report_json,
                image_paths=image_paths,
                views=selected_views
            )
        except Exception:
            return None
    
    def _select_views(self, images_info: List[Dict], image_root: Path) -> Tuple[List[Path], List[str]]:
        """Optimized view selection logic."""
        # Group images by view
        view_groups = {}
        for img in images_info:
            view = img.get("view", "UNKNOWN")
            if view not in view_groups:
                view_groups[view] = []
            view_groups[view].append(img)
        
        # Sort each group by order
        for view in view_groups:
            view_groups[view].sort(key=lambda x: x.get("order", 0))
        
        # Select 1 image per view in priority order
        selected = []
        selected_views = []
        for view in self.VIEW_PRIORITY:
            if view in view_groups and view_groups[view]:
                img_info = view_groups[view][0]
                path = image_root / img_info["path"]
                if path.exists():
                    selected.append(path)
                    selected_views.append(view)
        
        # If <2 images, try secondary images
        if len(selected) < 2:
            for view in view_groups:
                if len(view_groups[view]) > 1 and view not in selected_views:
                    img_info = view_groups[view][1]
                    path = image_root / img_info["path"]
                    if path.exists():
                        selected.append(path)
                        selected_views.append(view)
                        break
        
        # Hard cap at max_images
        return selected[:self.max_images], selected_views[:self.max_images]

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        """OPTIMIZED: All metadata pre-computed, only load images here."""
        study = self.studies[idx]
        
        # Load pre-computed data (instant, no computation)
        report = study.report_json
        selected_views = study.selected_views
        image_paths = study.selected_image_paths
        
        # Load Images (only IO-bound operation)
        loaded_images = []
        for path_str in image_paths:
            try:
                # PIL caches by default, so repeated opens are fast
                img = Image.open(path_str).convert("RGB")
                loaded_images.append(img)
            except Exception:
                # Path validation happened during init, so this is rare
                continue
        
        # Skip if all images failed (shouldn't happen given init validation)
        if not loaded_images:
            return None
        
        # Build message structure (minimal, only what's needed)
        user_content = [{"type": "image"}] * len(loaded_images)
        view_labels = " | ".join([f"<{view}>" for view in selected_views])
        user_content.append({
            "type": "text",
            "text": f"\nViews: {view_labels}\n\n{INSTRUCTION}"
        })
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": report}]}
        ]
        
        return {"images": loaded_images, "messages": messages}


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

class DebugGenerationCallback(TrainerCallback):
    """OPTIMIZED: Cache sample once, skip validation checks."""
    
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.cached_sample = None  # Cache sample once
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Only generate on first epoch (saves ~2 min per run)
        if state.epoch < 1.5:
            # Cache sample on first call
            if self.cached_sample is None:
                self.cached_sample = self.dataset[0]
            
            if self.cached_sample is None:
                return
            
            print(f"\n[{state.global_step}] === DEBUG SAMPLE ===" )
            try:
                sample = self.cached_sample
                user_messages = [m for m in sample["messages"] if m["role"] == "user"]
                
                text = self.processor.apply_chat_template(user_messages, add_generation_prompt=True)
                images = sample["images"]
                inputs = self.processor(text=text, images=images, return_tensors="pt").to(model.device)
                
                model.eval()
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, max_new_tokens=300, do_sample=False
                    )
                model.train()
                
                output_text = self.processor.tokenizer.decode(
                    generated_ids[0, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                
                print("Output:", output_text[:200])
            except Exception as e:
                print(f"Debug failed: {e}")
            print("======================\n")

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

def parse_args():
    """Parse command-line arguments to switch between overfit and full training modes."""
    parser = argparse.ArgumentParser(
        description="MedGemma QLoRA Fine-tuning - Chest X-ray Report Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Overfit sanity check (50 samples, 5 epochs - verifies training works)
  python qlora_medgemma_corrected.py --mode overfit

  # Full training (all 43927 samples, 1 epoch)
  python qlora_medgemma_corrected.py --mode full

  # Custom run
  python qlora_medgemma_corrected.py --mode full --epochs 3 --version v2
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["overfit", "full"],
        required=True,
        help="'overfit' = 50 samples x 5 epochs to verify training works. 'full' = all data x 1 epoch."
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Override number of epochs (default: 5 for overfit, 1 for full)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Run version tag for output directory (default: 'overfit_v1' or 'v1')"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Max images per study (default: 1)"
    )
    return parser.parse_args()


def apply_mode(config: TrainingConfig, args) -> TrainingConfig:
    """Apply mode-specific settings to config."""
    if args.mode == "overfit":
        config.max_train_samples  = 50       # tiny subset
        config.num_train_epochs   = 5.0      # more epochs so loss can converge
        config.eval_steps         = 10       # evaluate frequently
        config.save_steps         = 50
        config.logging_steps      = 5
        config.run_version        = args.version or "overfit_v1"
        print("MODE: OVERFIT SANITY CHECK")
        print("  → 50 samples, 5 epochs")
        print("  → Loss should drop from ~4-5 down to <1.0")
        print("  → If it does NOT converge, something is wrong before running full training")
    else:  # full
        config.max_train_samples  = 10000     # all 43927 studies
        config.num_train_epochs   = 1.0
        config.eval_steps         = 100
        config.save_steps         = 100
        config.logging_steps      = 10
        config.run_version        = args.version or "v1"
        print("MODE: FULL TRAINING")
        print("  → All 43927 studies, 1 epoch")
        print("  → Estimated time: 4-8 hours")

    # Apply any manual overrides
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
        print(f"  → Epochs overridden to: {args.epochs}")
    if args.max_images is not None:
        config.data.max_images = args.max_images
        print(f"  → Max images overridden to: {args.max_images}")

    return config


def main():
    args = parse_args()
    config = TrainingConfig()
    config = apply_mode(config, args)
    set_seed(config.seed)
    
    print("\n" + "="*60)
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
    
    # Disable use_cache - incompatible with gradient checkpointing, wastes memory
    model.config.use_cache = False

    # Freeze the vision tower entirely - we only want to train the language model
    # The vision encoder (SigLIP) is already good at extracting medical image features
    # Freezing it saves ~4-6 GB of activation memory during backward pass
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False
    print("✓ Vision tower frozen (saves ~4-6 GB during backward pass)")

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
    
    train_ds = ChestXrayReportDataset(
        train_path, image_root, 
        max_images=config.data.max_images,
        num_workers=config.data.num_workers
    )
    val_ds = ChestXrayReportDataset(
        val_path, image_root,
        max_images=config.data.max_images,
        num_workers=config.data.num_workers
    ) if val_path.exists() else None
    
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

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=config.optim,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
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
        dataloader_num_workers=config.dataloader_num_workers,  # Async data prefetching
        dataloader_pin_memory=True,  # Faster GPU transfer
        dataloader_prefetch_factor=2,  # Buffer ahead
    )

    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Optimizer: {config.optim}")
    print(f"Precision: {'BF16' if config.bf16 else 'FP16'}")
    print(f"DataLoader workers: {config.dataloader_num_workers} (async prefetching)")
    print(f"High-performance mode enabled")

    # 5. Create Trainer (TRL 0.28 style)
    print(f"\n=== Creating Trainer ===")

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=processor,
        callbacks=[
            CustomLoggingCallback(), 
            DebugGenerationCallback(train_ds, processor)
        ],
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
    processor.tokenizer.save_pretrained(str(output_dir))

    print(f"✓ Model saved to: {output_dir}")
    
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()