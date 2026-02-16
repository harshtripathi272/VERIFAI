"""
Training script for MedGemma QLoRA fine-tuning — chest X-ray report generation.

Usage (run from qlora_mg/):
    python scripts/train.py

Overfit sanity-check (200 studies):
    Edit config → max_train_samples = 200, then run as above.

Pipeline:
    1. Load tokenizer → add special tokens (<image>, <PA>, <AP>, <LATERAL>, <report>)
    2. Load SigLIP image processor
    3. Build MedGemmaVLM  (frozen SigLIP + trainable projector + QLoRA MedGemma)
    4. Resize embeddings for new tokens
    5. Apply LoRA adapters
    6. Train with masked causal-LM loss (input portion masked, target portion supervised)
    7. Save LoRA adapters + projector weights + tokenizer
"""

import os
import sys

WORKSPACE_ROOT1 ="../dataset/med/"
WORKSPACE_ROOT2="../dataset/med/"

import torch
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
import transformers
import warnings
warnings.filterwarnings("ignore", message=".*warmup_ratio is deprecated.*")
warnings.filterwarnings("ignore", message=".*use_cache=True.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*image processor.*fast processor.*")


class CleanLogCallback(TrainerCallback):
    """Print one clean line per logging step instead of the default dict."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        total = state.max_steps
        epoch = logs.get("epoch", 0)
        loss = logs.get("loss", logs.get("train_loss", None))
        lr = logs.get("learning_rate", None)
        grad = logs.get("grad_norm", None)
        parts = [f"Step {step:>4d}/{total}"]
        parts.append(f"Epoch {epoch:.2f}")
        if loss is not None:
            parts.append(f"Loss {float(loss):.4f}")
        if lr is not None:
            parts.append(f"LR {float(lr):.2e}")
        if grad is not None:
            parts.append(f"Grad {float(grad):.2f}")
        print("  │ " + "  │ ".join(parts) + "  │")

import json
import os
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset
from PIL import Image


VIEW_ORDER = {"PA": 0, "AP": 1, "LATERAL": 2}

# FIXED: Removed duplicate "FINDINGS:\n" and "IMPRESSION:\n" headers
# These headers are added in build_target_text(), so they should NOT be in the instruction
INSTRUCTION = (
    "You are an expert radiologist.\n\n"
    "Analyze the provided chest X-rays and write a careful radiology report "
    "using appropriate clinical language.\n\n"
)

SPECIAL_TOKENS = ["<image>", "<PA>", "<AP>", "<LATERAL>", "<report>"]
def build_input_text(image_views: List[str]) -> str:
    """
    Build the input / prompt portion of the sequence (will be loss-masked).

    Example output for two images (PA, AP):
        <image> <PA>
        <image> <AP>

        <report>

        You are an expert radiologist.
        ...
        FINDINGS:
        IMPRESSION:
    """
    lines = [f"<image> <{v}>" for v in image_views]
    text = "\n".join(lines) + "\n"
    text += "\n<report>\n\n"
    text += INSTRUCTION
    return text


def build_target_text(findings: str, impression: str) -> str:
    """
    Build the target portion the model must generate.

    Example:
        FINDINGS:
        The lungs are clear ...

        IMPRESSION:
        No acute cardiopulmonary process.
    """
    f = findings.strip() if findings else ""
    i = impression.strip() if impression else ""
    return f"FINDINGS:\n{f}\n\nIMPRESSION:\n{i}"

# ── Constants ────────────────────────────────────────────────────────


class ChestXrayReportDataset(Dataset):
    """
    Per-study dataset for chest X-ray report generation.

    Each __getitem__ returns:
        input_ids      – [max_length]   (full sequence: input + target, padded)
        attention_mask – [max_length]
        labels         – [max_length]   (-100 on input portion & padding)
        pixel_values   – [num_images, 3, H, W]
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root_dir: str,
        tokenizer,
        image_processor,
        max_length: int = 768,
        max_images: int = 15,
    ):
        self.image_root_dir = image_root_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.max_images = max_images

        # Track sequence length statistics
        self.seq_length_stats = {"truncated": 0, "max_seen": 0, "total": 0}
        self.print_first_sample = True

        # Load studies — keep only those with at least findings or impression
        self.studies: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                study = json.loads(line)
                has_text = (
                    bool(study.get("findings", "").strip())
                    or bool(study.get("impression", "").strip())
                )
                if has_text:
                    self.studies.append(study)

        print(f"[Dataset] Loaded {len(self.studies)} valid studies from {jsonl_path}")

    # ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        study = self.studies[idx]
        images_info = study["images"]
        findings = study.get("findings", "")
        impression = study.get("impression", "")

        # Sort: PA → AP → LATERAL, then by `order` within each view
        sorted_imgs = sorted(
            images_info,
            key=lambda x: (VIEW_ORDER.get(x["view"], 99), x["order"]),
        )[: self.max_images]

        # ── Load images ──────────────────────────────────────────────
        loaded_views: List[str] = []
        pv_list: List[torch.Tensor] = []

        for img_info in sorted_imgs:
            path = os.path.join(self.image_root_dir, img_info["path"])
            try:
                pil_img = Image.open(path).convert("RGB")
                proc = self.image_processor(images=pil_img, return_tensors="pt")
                pv_list.append(proc["pixel_values"].squeeze(0))   # [3, H, W]
                loaded_views.append(img_info["view"])
            except Exception:
                continue  # skip unreadable images silently

        # Fallback: if every image failed, use a black dummy so training
        # doesn't crash.  The view token will still match.
        if not pv_list:
            # Use the first view from sorted_imgs (or AP as safe default)
            fallback_view = sorted_imgs[0]["view"] if sorted_imgs else "AP"
            pv_list.append(torch.zeros(3, 384, 384))
            loaded_views.append(fallback_view)

        pixel_values = torch.stack(pv_list)  # [num_images, 3, H, W]

        # ── Build token sequences ────────────────────────────────────
        input_text = build_input_text(loaded_views)
        target_text = build_target_text(findings, impression)

        # Tokenize input & target separately → guarantees exact mask boundary
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        target_ids.append(self.tokenizer.eos_token_id)          # model learns to stop

        full_ids = input_ids + target_ids
        labels = [-100] * len(input_ids) + target_ids            # mask input portion

        # Track sequence length statistics
        original_len = len(full_ids)
        self.seq_length_stats["total"] += 1
        self.seq_length_stats["max_seen"] = max(self.seq_length_stats["max_seen"], original_len)
        
        # Truncate from the right
        was_truncated = False
        if len(full_ids) > self.max_length:
            was_truncated = True
            self.seq_length_stats["truncated"] += 1
            full_ids = full_ids[: self.max_length]
            labels = labels[: self.max_length]

        # Print first sample for sanity check
        if self.print_first_sample:
            self.print_first_sample = False
            print("\n" + "="*70)
            print("[Dataset] FIRST TRAINING SAMPLE (sanity check)")
            print("="*70)
            print(f"Input text ({len(input_text)} chars):\n{input_text}")
            print("\n" + "-"*70)
            print(f"Target text ({len(target_text)} chars):\n{target_text}")
            print("\n" + "-"*70)
            print(f"Input tokens: {len(input_ids)}")
            print(f"Target tokens: {len(target_ids)}")
            print(f"Full sequence: {original_len} tokens")
            print(f"Max length: {self.max_length}")
            print(f"Truncated: {was_truncated}")
            print("="*70 + "\n")

        # Pad to max_length
        seq_len = len(full_ids)
        pad_len = self.max_length - seq_len
        pad_id = self.tokenizer.pad_token_id

        input_ids_t = torch.tensor(
            full_ids + [pad_id] * pad_len, dtype=torch.long
        )
        attn_mask_t = torch.tensor(
            [1] * seq_len + [0] * pad_len, dtype=torch.long
        )
        labels_t = torch.tensor(
            labels + [-100] * pad_len, dtype=torch.long
        )

        # Gemma3 requires token_type_ids during training (all zeros for single-sequence)
        token_type_ids_t = torch.zeros_like(input_ids_t)

        return {
            "input_ids": input_ids_t,
            "attention_mask": attn_mask_t,
            "labels": labels_t,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids_t,
        }


# ── Collator ─────────────────────────────────────────────────────────

class ReportDataCollator:
    """
    Collator for variable-image-count batches.

    Text tensors are already padded to max_length by the dataset, so we
    just stack them.  pixel_values are concatenated across the batch into
    a single [total_images, 3, H, W] tensor.  The model infers per-sample
    image counts from the number of <image> tokens in each sample's
    input_ids — no explicit `image_counts` field is needed.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ── Dynamic in-batch padding (not global max_length) ─────────
        # Find the longest *real* (non-pad) sequence in this batch
        # and pad everything to that length — saves memory.
        lengths = [int(f["attention_mask"].sum()) for f in features]
        max_len = max(lengths)

        input_ids_list, attn_list, labels_list, tti_list = [], [], [], []
        for f in features:
            input_ids_list.append(f["input_ids"][:max_len])
            attn_list.append(f["attention_mask"][:max_len])
            labels_list.append(f["labels"][:max_len])
            tti_list.append(f["token_type_ids"][:max_len])

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attn_list),
            "labels": torch.stack(labels_list),
            "token_type_ids": torch.stack(tti_list),
            # All images across the batch, flat:  [Σ num_images_i, 3, H, W]
            "pixel_values": torch.cat(
                [f["pixel_values"] for f in features], dim=0
            ),
        }
    
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """Model identifiers and quantisation settings."""

    # Language Model (causal, QLoRA target)
    language_model_id: str = "google/medgemma-1.5-4b-it"

    # Vision Encoder (frozen MedSigLIP)
    vision_model_id: str = "google/medsiglip-448"

    # 4-bit NF4 quantisation
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoraConfig:
    """LoRA adapter settings."""

    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"

    # Target ALL linear layers + make embeddings trainable.
    # This is the ideal config from MedGemma official notebook.
    target_modules: str = "all-linear"  # Must be STRING, not list

    task_type: str = "CAUSAL_LM"

    # Make embeddings trainable so model can learn custom tokens like <image>
    modules_to_save: List[str] = field(default_factory=lambda: ["embed_tokens"])


@dataclass
class DataConfig:
    """
    Data paths and processing limits.

    Paths are relative to the *workspace root* (parent of qlora_mg/).
    They are resolved to absolute paths at runtime in the training script.
    """

    # JSONL files produced by the upstream pipeline
    # Use _clean versions to skip studies with missing images
    train_jsonl: str = "train_capped_clean.jsonl"
    val_jsonl: str = "val_capped_clean.jsonl"

    # Root directory that contains the ``files/`` image tree
    image_root_dir: str = "official_data_iccv_final"

    # Sequence / image limits
    max_length: int = 512
    max_images: int = 7
    image_size: int = 384


@dataclass
class TrainingConfig:
    """Top-level training configuration."""

    # Nested configs
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Training hyperparameters
    output_dir: str = "../dataset/med/fine_tuned_model"
    num_train_epochs: int = 1.8
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"

    # Optimisation
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"

    # Logging & checkpointing
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Precision
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    

    report_to: str = "none"

    # Versioned output sub-directory.  Every run saves to
    # <output_dir>/<run_version>/  so previous runs are preserved.
    run_version: str = "v1"

    # Overfit test: set to e.g. 200 for a quick sanity check before
    # full training.  None = use entire training set.
    max_train_samples: Optional[int] = 10000

import torch
from typing import List, Optional, Union

from transformers import BitsAndBytesConfig
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
def get_quantization_config(
    load_in_4bit: bool = True,
    compute_dtype: str = "bfloat16",
    quant_type: str = "nf4",
    use_double_quant: bool = True,
) -> Optional[BitsAndBytesConfig]:
    """
    Create BitsAndBytes quantization config for 4-bit loading.
    
    Args:
        load_in_4bit: Whether to enable 4-bit quantization
        compute_dtype: Compute dtype string ('float16', 'bfloat16')
        quant_type: Quantization type ('nf4', 'fp4')
        use_double_quant: Whether to use nested quantization
    
    Returns:
        BitsAndBytesConfig or None if quantization disabled
    """
    if not load_in_4bit:
        return None
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=use_double_quant,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[Union[str, List[str]]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    modules_to_save: Optional[List[str]] = None,
) -> PeftLoraConfig:
    """
    Create PEFT LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: String "all-linear" or list of module names to apply LoRA to
        bias: Bias type ('none', 'all', 'lora_only')
        task_type: Task type for PEFT
        modules_to_save: List of modules to save fully (not LoRA)
    
    Returns:
        PeftLoraConfig instance
    """
    if target_modules is None:
        target_modules = "all-linear"
    
    # PEFT recognises "all-linear" only as a plain string, not as a
    # single-element list.  Convert ["all-linear"] → "all-linear".
    if isinstance(target_modules, list) and target_modules == ["all-linear"]:
        target_modules = "all-linear"

    kwargs = dict(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type,
    )
    if modules_to_save:
        kwargs["modules_to_save"] = modules_to_save
    
    return PeftLoraConfig(**kwargs)


def prepare_model_for_training(model, lora_config: LoraConfig):
    """
    Apply LoRA adapters to the language model component.
    
    This function:
        1. Prepares the model for k-bit training (non-reentrant gradient
           checkpointing — required for hook-based vision injection)
        2. Applies LoRA adapters to specified modules
        3. Ensures projector remains trainable
    
    Args:
        model: MedGemmaVLM instance
        lora_config: Application LoRA configuration
    
    Returns:
        Modified model with LoRA applied
    """
    # Prepare language model for quantized training
    # CRITICAL: use_reentrant=False avoids issues with forward hooks
    # (the official MedGemma fine-tuning notebook uses the same setting)
    model.language_model = prepare_model_for_kbit_training(
        model.language_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    # Create PEFT config
    peft_config = get_lora_config(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,  # Keep as-is (string or list)
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        modules_to_save=list(lora_config.modules_to_save) if hasattr(lora_config, 'modules_to_save') and lora_config.modules_to_save else None,
    )
    
    # Apply LoRA to language model
    model.language_model = get_peft_model(model.language_model, peft_config)
    
    # Ensure projector is trainable
    for param in model.projector.parameters():
        param.requires_grad = True
    
    return model

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from transformers import (
    SiglipVisionModel,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
class VisionProjector(nn.Module):
    """
    Projects mean-pooled SigLIP features to MedGemma hidden dim.

    Architecture:  Linear → GELU → LayerNorm
    Initialisation: Xavier-uniform with gain 0.1 (small scale to avoid
    disturbing the quantised LLM embeddings at the start of training).
    """

    def __init__(self, vision_hidden_size: int, lm_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, lm_hidden_size)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(lm_hidden_size)

        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.act(self.linear(x)))


# ── Main model ────────────────────────────────────────────────────────

class MedGemmaVLM(nn.Module):
    """
    Multimodal VLM:  frozen SigLIP  +  trainable projector  +  QLoRA MedGemma.

    Vision injection strategy (hook-based):
        We pass ``input_ids`` to the LLM (NOT ``inputs_embeds``) so that
        Gemma3 can compute position IDs, cache positions, and its special
        causal mask correctly.  A **forward hook** on the embedding layer
        intercepts the embedding output and replaces ``<image>`` positions
        with projected vision vectors.  This is compatible with gradient
        checkpointing and autoregressive generation.
    """

    def __init__(
        self,
        vision_model_id: str,
        language_model_id: str,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.vision_model_id = vision_model_id
        self.language_model_id = language_model_id

        # ── Vision encoder (frozen) ──────────────────────────────────
        import logging
        _siglip_logger = logging.getLogger("transformers.modeling_utils")
        _prev_level = _siglip_logger.level
        _siglip_logger.setLevel(logging.ERROR)

        self.vision_tower = SiglipVisionModel.from_pretrained(
            vision_model_id,
            torch_dtype=torch_dtype,
        )

        _siglip_logger.setLevel(_prev_level)
        for p in self.vision_tower.parameters():
            p.requires_grad = False
        self.vision_tower.eval()

        # ── Language model (quantised) ───────────────────────────────
        self.language_model = AutoModelForImageTextToText.from_pretrained(
            language_model_id,
            quantization_config=quantization_config,
            device_map=None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        
        # Disable KV cache for training (saves memory, not needed)
        self.language_model.config.use_cache = False

        # ── Projection layer (trainable, same dtype as vision tower) ─
        vis_h = self.vision_tower.config.hidden_size
        lm_cfg = self.language_model.config
        lm_h = getattr(lm_cfg, "hidden_size", None) or lm_cfg.text_config.hidden_size
        self.projector = VisionProjector(vis_h, lm_h)
        self.projector = self.projector.to(dtype=torch_dtype)

        # ── Hook state (set before each forward, read by the hook) ───
        self.image_token_id: Optional[int] = None
        self._pending_vision_embeds: Optional[torch.Tensor] = None
        self._pending_image_positions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.verbose_forward = False

        # ── Self-healing hook tracking ────────────────────────────────
        # The hook must be on the CURRENT embedding layer.  If the layer
        # changes (e.g., after resize_token_embeddings), _ensure_hook()
        # will re-register automatically.
        self._hooked_embed_layer = None
        self._embed_hook_handle = None

    # ── Self-healing hook registration ───────────────────────────────

    def _ensure_hook(self):
        """
        Guarantee the embedding hook is on the current embedding layer.

        Called at the start of every forward() and generate() so that
        resize_token_embeddings() or PEFT wrapping cannot orphan the hook.
        """
        current_embed = self.language_model.get_input_embeddings()
        if current_embed is not self._hooked_embed_layer:
            # Embedding layer changed → re-register
            if self._embed_hook_handle is not None:
                self._embed_hook_handle.remove()
            self._embed_hook_handle = current_embed.register_forward_hook(
                self._embedding_hook
            )
            self._hooked_embed_layer = current_embed
            if self.verbose_forward:
                print(f"    [VLM] Hook registered on {type(current_embed).__name__}")

    # ── Embedding hook ───────────────────────────────────────────────

    def _embedding_hook(self, module, args, output):
        """
        Fires every time the LLM embedding layer runs (including during
        gradient-checkpointing recomputation).  If vision embeddings are
        pending, swap them into the output at ``<image>`` positions.
        """
        if (
            self._pending_vision_embeds is not None
            and self._pending_image_positions is not None
        ):
            b_idx, s_idx = self._pending_image_positions

            if self.verbose_forward:
                print(f"    [HOOK] Pending vision embeds: {self._pending_vision_embeds.shape}")
                print(f"    [HOOK] Positions: batch={b_idx.tolist()}, seq={s_idx.tolist()}")
                print(f"    [HOOK] Output shape: {output.shape}")
                print(f"    [HOOK] Check: output.shape[1]={output.shape[1]} > s_idx.max()={s_idx.max()}")

            # During generation decode steps the sequence length is 1,
            # so the stored positions would be out of range → skip.
            if len(s_idx) > 0 and output.shape[1] > s_idx.max():
                projected = self._pending_vision_embeds.to(
                    device=output.device, dtype=output.dtype
                )
                # Clone so the original embedding output stays untouched
                # in the autograd graph — safe with gradient checkpointing.
                new_output = output.clone()
                new_output[b_idx, s_idx] = projected
                
                if self.verbose_forward:
                    print(f"    [HOOK] ✓ SWAPPED {len(b_idx)} vision embeddings")
                    print(f"    [HOOK] Before swap at pos {s_idx[0].item()}: {output[0, s_idx[0], :5]}")
                    print(f"    [HOOK] After swap at pos {s_idx[0].item()}: {new_output[0, s_idx[0], :5]}")
                
                return new_output
            else:
                if self.verbose_forward:
                    print(f"    [HOOK] ✗ SKIP (condition failed)")

        return output

    # ── Trainer compatibility (quantised gradient flow) ────────────

    def enable_input_require_grads(self):
        """
        Enable gradients on embedding outputs so that quantised (4-bit)
        models can still propagate gradients.  The HF Trainer calls
        this automatically when gradient_checkpointing is True.
        """
        if hasattr(self.language_model, 'enable_input_require_grads'):
            self.language_model.enable_input_require_grads()
        else:
            def _make_grads(module, input, output):
                output.requires_grad_(True)
            self.language_model.get_input_embeddings().register_forward_hook(_make_grads)

    def get_input_embeddings(self):
        """Delegate to the language model so Trainer can find embeddings."""
        return self.language_model.get_input_embeddings()

    # ── keep vision tower in eval even when Trainer calls .train() ───

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_tower.eval()
        return self

    # ── vision helpers ───────────────────────────────────────────────

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        SigLIP  →  mean-pool over patches  →  project.

        Args:
            pixel_values: (N, 3, H, W)
        Returns:
            projected: (N, lm_hidden_size)
        """
        if self.verbose_forward:
            print(f"    [VLM] Encoding {pixel_values.shape[0]} images via SigLIP")

        with torch.no_grad():
            out = self.vision_tower(
                pixel_values=pixel_values.to(
                    device=self.vision_tower.device,
                    dtype=self.vision_tower.dtype,
                ),
            )
            features = out.last_hidden_state          # (N, num_patches, vis_h)

        pooled = features.mean(dim=1)                  # (N, vis_h)

        proj_device = next(self.projector.parameters()).device
        projected = self.projector(pooled.to(proj_device))   # (N, lm_h)

        if self.verbose_forward:
            print(f"    [VLM] Projected: {projected.shape}, scale={projected.abs().mean():.4f}")

        return projected

    # ── forward (training) ───────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for training.

        We pass ``input_ids`` (NOT ``inputs_embeds``) to the LLM so that
        Gemma3 can correctly compute position IDs and its causal mask.
        Vision features are injected via the embedding hook.
        """
        # ── Ensure hook is on the correct (current) embedding layer ──
        self._ensure_hook()

        # ── Prepare vision injection (hook reads this) ───────────────
        self._pending_vision_embeds = None
        self._pending_image_positions = None

        if pixel_values is not None and pixel_values.shape[0] > 0:
            assert self.image_token_id is not None, (
                "image_token_id not set — call model.image_token_id = ... first"
            )

            projected = self.encode_images(pixel_values)

            image_mask = input_ids == self.image_token_id
            b_idx, s_idx = image_mask.nonzero(as_tuple=True)

            n_tok, n_img = len(b_idx), projected.shape[0]
            assert n_tok == n_img, (
                f"<image> count ({n_tok}) != image count ({n_img})"
            )

            self._pending_vision_embeds = projected
            self._pending_image_positions = (b_idx, s_idx)

            if self.verbose_forward:
                print(f"    [VLM] Prepared {n_img} vision embeds for hook injection")

        # ── Forward through LLM (embedding hook fires inside) ────────
        lm_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        if token_type_ids is not None:
            lm_kwargs["token_type_ids"] = token_type_ids

        outputs = self.language_model(**lm_kwargs)

        # NOTE: Do NOT clean up _pending_vision_embeds here!
        # With gradient checkpointing, the embedding layer's forward
        # may be recomputed during the backward pass.  The hook needs
        # the pending embeds to still be available at that point.
        # They are reset at the START of the next forward() call.

        return {"loss": outputs.loss, "logits": outputs.logits}

    # ── generate (inference) ─────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Generate text conditioned on images and a text prompt.

        Uses the same hook mechanism.  The hook auto-skips during decode
        steps (sequence length = 1) so it only injects during prefill.
        """
        # ── Ensure hook is on the correct (current) embedding layer ──
        self._ensure_hook()

        self._pending_vision_embeds = None
        self._pending_image_positions = None

        if pixel_values is not None and pixel_values.shape[0] > 0:
            assert self.image_token_id is not None

            projected = self.encode_images(pixel_values)

            image_mask = input_ids == self.image_token_id
            b_idx, s_idx = image_mask.nonzero(as_tuple=True)
            assert len(b_idx) == projected.shape[0], (
                f"<image> count ({len(b_idx)}) != image count ({projected.shape[0]})"
            )

            self._pending_vision_embeds = projected
            self._pending_image_positions = (b_idx, s_idx)

        result = self.language_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # Safe to clean up after generate (no gradient checkpointing in inference)
        self._pending_vision_embeds = None
        self._pending_image_positions = None

        return result

    # ── gradient checkpointing delegation ────────────────────────────

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()

    # ── parameter stats ──────────────────────────────────────────────

    def get_trainable_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def print_trainable_parameters(self):
        trainable, total = self.get_trainable_parameters()
        pct = 100 * trainable / total if total > 0 else 0
        print(f"  Trainable: {trainable:,} / {total:,}  ({pct:.4f}%)")
    

class SaveBestLoRACallback(TrainerCallback):
    """
    Custom callback that saves ONLY LoRA adapters + projector weights
    when validation loss improves. Replaces HF Trainer's default 
    full-checkpoint saving to stay under Kaggle's 20GB disk limit.
    """
    def __init__(self, model, tokenizer, save_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.best_eval_loss = float("inf")
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is None:
            return
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            print(f"\n  ★ New best eval_loss: {eval_loss:.6f} (step {state.global_step})")
            self._save_lightweight(state.global_step)
        else:
            print(f"\n  · eval_loss: {eval_loss:.6f} (best: {self.best_eval_loss:.6f})")
    def _save_lightweight(self, step):
        import shutil
        temp_dir = self.save_dir + "_tmp"
    
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
    
        # 1. Save LoRA adapters
        lora_dir = os.path.join(temp_dir, "lora_adapters")
        self.model.language_model.save_pretrained(lora_dir)
    
        # 2. Save projector
        torch.save(
            self.model.projector.state_dict(),
            os.path.join(temp_dir, "projector.pt"),
        )
    
        # 3. Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(temp_dir, "tokenizer"))
    
        # Atomic replace
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.rename(temp_dir, self.save_dir)
    
        # Size report
        total_size = 0
        for dirpath, _, filenames in os.walk(self.save_dir):
            for f in filenames:
                total_size += os.path.getsize(os.path.join(dirpath, f))
    
        print(f"    ✓ Saved BEST model at step {step}")
        print(f"    📦 Total save size: {total_size / (1024**2):.1f} MB")
    

def main():
    config = TrainingConfig()
    set_seed(config.seed)

    
    

    # ── Resolve data paths (relative to workspace root) ──────────────
    train_path = os.path.join(WORKSPACE_ROOT1, config.data.train_jsonl)
    val_path = os.path.join(WORKSPACE_ROOT1, config.data.val_jsonl)
    image_root = os.path.join(WORKSPACE_ROOT2, config.data.image_root_dir)

    print("\n" + "=" * 60)
    print("MedGemma QLoRA — Chest X-ray Report Generation")
    print("=" * 60)
    print(f"  LLM        : {config.model.language_model_id}")
    print(f"  Vision     : {config.model.vision_model_id}")
    print(f"  LoRA r/α   : {config.lora.r} / {config.lora.lora_alpha}")
    print(f"  Max length : {config.data.max_length}")
    print(f"  Train data : {train_path}")
    print(f"  Image root : {image_root}")
    if config.max_train_samples:
        print(f"  ⚠  OVERFIT TEST — capping to {config.max_train_samples} samples")
    print("=" * 60)

    # ── 1. Tokenizer + mandatory special tokens ─────────────────────
    print("\n[1/6] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.language_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    assert image_token_id != tokenizer.unk_token_id, (
        "<image> not in vocab — special token addition failed"
    )
    print(f"  Added {num_added} special tokens: {SPECIAL_TOKENS}")
    print(f"  <image> token id = {image_token_id}")

    # ── 2. Image processor ───────────────────────────────────────────
    print("\n[2/6] Loading image processor …")
    image_processor = AutoImageProcessor.from_pretrained(
        config.model.vision_model_id,
    )

    # ── 3. Build model ───────────────────────────────────────────────
    print("\n[3/6] Building model (4-bit quantisation) …")
    print("  Loading vision encoder (SigLIP) …")
    quant_config = get_quantization_config(
        load_in_4bit=config.model.load_in_4bit,
        compute_dtype=config.model.bnb_4bit_compute_dtype,
        quant_type=config.model.bnb_4bit_quant_type,
        use_double_quant=config.model.bnb_4bit_use_double_quant,
    )
    print("  Loading language model (MedGemma-4B in 4-bit) …")
    model = MedGemmaVLM(
        vision_model_id=config.model.vision_model_id,
        language_model_id=config.model.language_model_id,
        quantization_config=quant_config,
        torch_dtype=torch.float16,  # Use float16 for RTX 8000 compatibility
    )
    print("  ✓ Model loaded successfully")
    

    # Resize embeddings to accommodate new special tokens
    print("\n  Resizing embeddings for special tokens …")
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.image_token_id = image_token_id
    print(f"  ✓ Embeddings resized → {len(tokenizer)} tokens")

    # ── 4. Apply LoRA ────────────────────────────────────────────────
    print("\n[4/6] Applying LoRA adapters …")
    model = prepare_model_for_training(model, config.lora)
    model.print_trainable_parameters()
    

    # Move vision tower + projector to same device as LLM embeddings
    print("\n  Moving components to GPU …")
    embed_device = model.language_model.get_input_embeddings().weight.device
    print(f"    LLM embeddings device: {embed_device}")
    model.vision_tower = model.vision_tower.to(embed_device)
    print(f"    ✓ Vision tower → {embed_device}")
    model.projector = model.projector.to(embed_device)
    print(f"    ✓ Projector → {embed_device}")
    

    # ── 5. Datasets ──────────────────────────────────────────────────
    print("\n[5/6] Loading datasets …")
    print(f"  Reading training data from: {train_path}")
    train_ds = ChestXrayReportDataset(
        jsonl_path=train_path,
        image_root_dir=image_root,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config.data.max_length,
        max_images=config.data.max_images,
    )
    print(f"  ✓ Loaded {len(train_ds)} training studies")

    val_ds = None
    if os.path.exists(val_path):
        print(f"  Reading validation data from: {val_path}")
        val_ds = ChestXrayReportDataset(
            jsonl_path=val_path,
            image_root_dir=image_root,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_length=config.data.max_length,
            max_images=config.data.max_images,
        )
        print(f"  ✓ Loaded {len(val_ds)} validation studies")


    # Print sequence length statistics
    print(f"\n  " + "="*60)
    print(f"  SEQUENCE LENGTH STATISTICS (max_length={config.data.max_length})")
    print(f"  " + "="*60)
    print(f"  Training set:")
    print(f"    Total samples processed: {train_ds.seq_length_stats['total']}")
    print(f"    Max sequence length seen: {train_ds.seq_length_stats['max_seen']} tokens")
    print(f"    Truncated samples: {train_ds.seq_length_stats['truncated']}")
    if train_ds.seq_length_stats['total'] > 0:
        trunc_pct = 100.0 * train_ds.seq_length_stats['truncated'] / train_ds.seq_length_stats['total']
        print(f"    Truncation rate: {trunc_pct:.2f}%")
        if trunc_pct > 10:
            print(f"    ⚠️  WARNING: >10% truncation! Consider increasing max_length")
        else:
            print(f"    ✓ Truncation rate is acceptable")
    print(f"  " + "="*60 + "\n")
    # Overfit-test subset
    if config.max_train_samples and config.max_train_samples < len(train_ds):
        train_ds.studies = train_ds.studies[: config.max_train_samples]
        print(f"  ⚠ OVERFIT MODE: Training on first {len(train_ds)} samples only")

    # ── 6. Trainer ──────────────────────────────────────────────────
    print("\n[6/6] Setting up Trainer …")

    # Calculate dynamic steps based on dataset size
    eff_bs = config.per_device_train_batch_size * config.gradient_accumulation_steps
    # If running distributed, multiply by num_gpus (assuming 1 here for simplicity/safety)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        eff_bs *= torch.cuda.device_count()

    num_samples = len(train_ds) if config.max_train_samples is None else min(len(train_ds), config.max_train_samples)
    steps_per_epoch = max(1, num_samples // eff_bs)
    
    # Log 4 times per epoch
    logging_steps = max(1, int(steps_per_epoch * 0.2))
    save_steps=500

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        logging_strategy="steps",
        logging_steps=logging_steps,
        # ── CRITICAL: disable default checkpoint saving ──────────
        save_strategy="no",
        # eval still runs on schedule for the callback to track best
        eval_strategy="steps" if val_ds else "no",
        eval_steps=save_steps if val_ds else None,
        # ── Do NOT use load_best_model_at_end (requires save_strategy) ──
        load_best_model_at_end=False,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=config.report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        seed=config.seed,
    )

    save_dir = os.path.join(config.output_dir, config.run_version)
    collator = ReportDataCollator(pad_token_id=tokenizer.pad_token_id)
    # Create the best-model callback BEFORE the trainer
    best_model_callback = SaveBestLoRACallback(
        model=model,
        tokenizer=tokenizer,
        save_dir=save_dir,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[CleanLogCallback(), best_model_callback],  # ← added
    )
    # Ensure labels are properly recognized for loss computation
    trainer.label_names = ["labels"]
    # Disable default logging to avoid duplicate output
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    # ── Train ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"  Total training samples: {len(train_ds)}")
    print(f"  Batch size per device: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    effective_bs = config.per_device_train_batch_size * config.gradient_accumulation_steps
    if torch.cuda.is_available():
        effective_bs *= torch.cuda.device_count()
    print(f"  Effective batch size: {effective_bs}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Checkpoint saving: DISABLED (LoRA-only via callback)")
    print("=" * 60 + "\n")
   
    print()
    trainer.train()
    # ── Final save (in case no eval happened or as final snapshot) ───
    print(f"\n{'=' * 60}")
    print("SAVING FINAL MODEL ARTEFACTS")
    print(f"{'=' * 60}")
    print(f"  Best eval_loss seen: {best_model_callback.best_eval_loss:.6f}")
    
    # If callback already saved the best model, just confirm
    if best_model_callback.best_eval_loss < float("inf"):
        print(f"  ✓ Best model already saved at: {save_dir}")
    else:
        # No eval happened — save current state
        print("  ⚠ No evaluation was run — saving final model state")
        best_model_callback._save_lightweight(step="final")
    # ── Verify saved directory structure ─────────────────────────────
    print(f"\n  📁 Saved directory structure:")
    for root, dirs, files in os.walk(save_dir):
        level = root.replace(save_dir, "").count(os.sep)
        indent = "    " + "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "    " + "  " * (level + 1)
        for file in files:
            fsize = os.path.getsize(os.path.join(root, file)) / (1024**2)
            print(f"{sub_indent}{file}  ({fsize:.1f} MB)")
    print(f"\n{'=' * 60}")
    print("✓ TRAINING COMPLETE — Only LoRA + projector saved")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()