"""
QLoRA Fine-tuning for MedGemma-4B on CheXpert Dataset

Fine-tunes MedGemma-4B using QLoRA (4-bit quantization) for chest X-ray diagnosis.
"""

import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from torch.utils.data import Dataset
import numpy as np


# CheXpert label columns
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]


@dataclass
class ModelConfig:
    """Configuration for QLoRA training"""
    model_name: str = "google/medgemma-1.5-4b-it"
    vision_encoder: str = "google/medsiglip-448"
    
    # QLoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # Will be set to attention layers
    
    # Training config
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_length: int = 512
    
    # Paths
    data_dir: str = "agents/radiologist/CheXpert_Dataset"
    output_dir: str = "agents/radiologist/checkpoints"
    
    def __post_init__(self):
        if self.target_modules is None:
            # Target attention layers for LoRA
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class CheXpertDataset(Dataset):
    """CheXpert dataset for VLM training"""
    
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        processor,
        tokenizer,
        max_length: int = 512,
        policy: str = "U-Ones"  # U-Ones: treat uncertain as positive
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.policy = policy
        
        # Filter for frontal views only
        self.df = self.df[self.df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} frontal chest X-rays")
    
    def __len__(self):
        return len(self.df)
    
    def _get_labels(self, row: pd.Series) -> Dict[str, float]:
        """Extract and process labels according to policy"""
        labels = {}
        for label in CHEXPERT_LABELS:
            val = row[label]
            if pd.isna(val):
                labels[label] = 0.0
            elif val == -1.0:  # Uncertain
                if self.policy == "U-Ones":
                    labels[label] = 1.0
                elif self.policy == "U-Zeros":
                    labels[label] = 0.0
                else:  # U-Ignore
                    labels[label] = -1.0
            else:
                labels[label] = float(val)
        return labels
    
    def _create_prompt(self, row: pd.Series, labels: Dict[str, float]) -> str:
        """Create instruction prompt for VLM"""
        # Patient context
        age = row["Age"]
        sex = row["Sex"]
        view = f"{row['Frontal/Lateral']} ({row['AP/PA']})" if pd.notna(row['AP/PA']) else row['Frontal/Lateral']
        
        # Positive findings
        positive_findings = [k for k, v in labels.items() if v == 1.0]
        
        if not positive_findings or "No Finding" in positive_findings:
            diagnosis = "No acute findings. Clear lungs and normal cardiac silhouette."
        else:
            diagnosis = f"Findings present: {', '.join(positive_findings)}."
        
        # Create structured prompt
        prompt = f"""You are a radiologist analyzing a chest X-ray.

Patient Information:
- Age: {age} years
- Sex: {sex}
- View: {view}

Analyze this chest X-ray image and provide a structured report with findings and diagnosis.

Diagnosis: {diagnosis}"""
        
        return prompt
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.data_root / row["Path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return next item
            return self.__getitem__((idx + 1) % len(self))
        
        # Get labels and create prompt
        labels = self._get_labels(row)
        prompt = self._create_prompt(row, labels)
        
        # Process image
        image_inputs = self.processor(images=image, return_tensors="pt")
        
        # Tokenize text
        text_inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": text_inputs["input_ids"].squeeze(0)  # For causal LM
        }


def setup_qlora_model(config: ModelConfig):
    """Setup model with QLoRA quantization"""
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    print(f"Loading {config.model_name} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def train(config: ModelConfig):
    """Main training function"""
    
    # Load tokenizer and processor
    print("Loading tokenizer and vision processor...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    processor = AutoProcessor.from_pretrained(config.vision_encoder)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup model
    model = setup_qlora_model(config)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = CheXpertDataset(
        csv_path=f"{config.data_dir}/train.csv",
        data_root=config.data_dir,
        processor=processor,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    val_dataset = CheXpertDataset(
        csv_path=f"{config.data_dir}/valid.csv",
        data_root=config.data_dir,
        processor=processor,
        tokenizer=tokenizer,
        max_length=config.max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=True,
        optim="paged_adamw_8bit",  # Memory efficient optimizer
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="tensorboard",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train!
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    model.save_pretrained(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    
    print("Training complete!")


if __name__ == "__main__":
    config = ModelConfig()
    train(config)
