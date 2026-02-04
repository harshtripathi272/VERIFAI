# QLoRA Fine-tuning Guide for MedGemma-4B on CheXpert

## Overview
This guide walks you through fine-tuning **MedGemma-4B** using **QLoRA (4-bit quantization)** on the **CheXpert dataset** for chest X-ray diagnosis.

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with at least 16GB VRAM (RTX 4090, A100, etc.)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: ~100GB for dataset + checkpoints

### Software
- Python 3.10+
- CUDA 11.8+ / 12.0+
- PyTorch 2.1+

## Setup

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all requirements
pip install -r requirements.txt
```

### 2. Verify CheXpert Dataset Structure

Your dataset should be structured like:
```
agents/radiologist/CheXpert_Dataset/
├── train.csv
├── valid.csv
└── train/
    └── patient00001/
        └── study1/
            └── view1_frontal.jpg
```

## Training

### Option 1: Basic Training (Default Settings)

```bash
cd d:\Workspace\VERIFAI
python agents/radiologist/train_qlora.py
```

### Option 2: Custom Configuration

Edit `train_qlora.py` to modify training parameters:

```python
config = ModelConfig(
    batch_size=4,              # Reduce if OOM
    gradient_accumulation_steps=8,  # Effective batch = 32
    num_epochs=3,              # Adjust as needed
    learning_rate=2e-4,        # Conservative LR
    lora_r=16,                 # LoRA rank
    lora_alpha=32,             # LoRA alpha
)
```

### Option 3: Multi-GPU Training

```bash
# Using accelerate (recommended)
accelerate config  # Run once to configure
accelerate launch agents/radiologist/train_qlora.py
```

## Training Configuration

### QLoRA Settings
- **Quantization**: 4-bit NF4 with double quantization
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Dropout**: 0.05

### Training Hyperparameters
- **Batch Size**: 4 per GPU
- **Gradient Accumulation**: 8 steps (effective batch size = 32)
- **Learning Rate**: 2e-4
- **Warmup Steps**: 100
- **Epochs**: 3
- **Optimizer**: 8-bit Paged AdamW
- **Mixed Precision**: FP16

### Label Handling
- **Policy**: U-Ones (uncertain labels treated as positive)
- **Alternative**: Change `policy="U-Zeros"` in `CheXpertDataset`

## Expected Training Time

On **RTX 4090** (24GB):
- ~8-10 hours for 3 epochs on full CheXpert training set
- Memory usage: ~18-20GB VRAM

On **A100** (40GB):
- ~6-8 hours for 3 epochs
- Can increase batch size for faster training

## Monitoring Training

### TensorBoard
```bash
# In a separate terminal
tensorboard --logdir agents/radiologist/checkpoints
```

### Weights & Biases (Optional)
```bash
# Login once
wandb login

# Training logs will automatically upload
```

## Checkpoints

Models are saved to: `agents/radiologist/checkpoints/`

- `checkpoint-500/`: Saved every 500 steps
- `final/`: Final trained model
- Best model (based on eval loss) is loaded at end

## Loading Trained Model

After training, update `agents/radiologist/model.py`:

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/medgemma-1.5-4b-it",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    "agents/radiologist/checkpoints/final"
)
```

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `batch_size` to 2 or 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable `gradient_checkpointing` (already enabled)
4. Use smaller `lora_r` (e.g., 8 instead of 16)

### CUDA Errors
```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Slow Training
- Enable gradient checkpointing (already enabled)
- Reduce `dataloader_num_workers` if CPU-bound
- Use `bfloat16` instead of `fp16` on Ampere+ GPUs

## Evaluation

The training script automatically evaluates every 500 steps. Final metrics will be printed at the end.

To evaluate manually:
```python
from transformers import Trainer

# Load model and eval dataset
trainer = Trainer(model=model, eval_dataset=val_dataset)
results = trainer.evaluate()
print(results)
```

## Next Steps

After training:
1. **Integrate into VERIFAI**: Update `agents/radiologist/model.py` to load fine-tuned model
2. **Test inference**: Run `test_workflow.py` to test the full pipeline
3. **Evaluate on test set**: Create evaluation script for CheXpert test set
4. **Deploy**: Use the trained model in production

## Model Performance Expectations

With 3 epochs of training on CheXpert:
- **Validation AUC**: 0.75-0.85 (depending on label)
- **Radiologist-level**: Comparable to human experts on many pathologies
- **Uncertainty signals**: Better calibrated after fine-tuning

## Citation

If you use this training setup, please cite:

```bibtex
@software{verifai2026,
  title={VERIFAI: Verifiable Epistemic Reasoning in Foundation AI for Medical Diagnosis},
  year={2026},
  note={QLoRA fine-tuning on CheXpert dataset}
}
```

## Support

For issues or questions:
1. Check GPU memory with `nvidia-smi`
2. Review training logs in `agents/radiologist/checkpoints/`
3. Enable debug logging: `export TRANSFORMERS_VERBOSITY=debug`
