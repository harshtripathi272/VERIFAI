"""
Test script to validate MedGemma QLoRA setup before full training.
UPDATED to work with qlora-medgemma-corrected.py

Run this BEFORE starting the full training to catch issues early:
    python test_medgemma_setup_fixed.py

This will:
1. Check GPU and memory
2. Validate all file paths
3. Test dataset loading
4. Test tokenization and collation with DataCollatorForCompletionOnlyLM
5. Run a single forward/backward pass
6. Estimate memory usage
"""

import os
import sys
import json
from pathlib import Path

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM

# Import from your corrected script
try:
    from qlora_medgemma import (
        ChestXrayReportDataset,
        get_response_template,
        WORKSPACE_ROOT,
        TrainingConfig,
    )
    print("✓ Successfully imported from qlora_medgemma_corrected")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nMake sure qlora-medgemma-corrected.py is renamed to qlora_medgemma_corrected.py")
    print("Or adjust the filename in the import statement above")
    sys.exit(1)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")

def check_gpu():
    """Check GPU availability and capabilities."""
    print_section("GPU CHECK")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16_support = torch.cuda.is_bf16_supported()
    
    # Check current GPU usage
    torch.cuda.reset_peak_memory_stats()
    current_allocated = torch.cuda.memory_allocated(0) / 1e9
    current_reserved = torch.cuda.memory_reserved(0) / 1e9
    
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ Total memory: {gpu_memory:.2f} GB")
    print(f"✓ Currently allocated: {current_allocated:.2f} GB")
    print(f"✓ Currently reserved: {current_reserved:.2f} GB")
    print(f"✓ BF16 supported: {bf16_support}")
    
    # Calculate available memory (accounting for other processes)
    # nvidia-smi shows total usage, but we can estimate available for us
    available_estimate = gpu_memory - 17  # Rough estimate based on your output
    print(f"✓ Estimated available for training: ~{available_estimate:.0f} GB")
    
    if available_estimate < 15:
        print(f"⚠ Warning: Only ~{available_estimate:.0f} GB may be available")
        print("  Other processes are using significant GPU memory")
    
    return True

def check_paths(config):
    """Validate all required paths."""
    print_section("PATH VALIDATION")
    
    issues = []
    
    # Workspace root
    if not WORKSPACE_ROOT.exists():
        issues.append(f"Workspace root not found: {WORKSPACE_ROOT}")
    else:
        print(f"✓ Workspace root: {WORKSPACE_ROOT}")
    
    # Training data
    train_path = WORKSPACE_ROOT / config.data.train_jsonl
    if not train_path.exists():
        issues.append(f"Training data not found: {train_path}")
    else:
        print(f"✓ Training data: {train_path}")
        # Count lines
        with open(train_path) as f:
            num_lines = sum(1 for _ in f)
        print(f"  → {num_lines} studies in training set")
    
    # Validation data
    val_path = WORKSPACE_ROOT / config.data.val_jsonl
    if not val_path.exists():
        print(f"⚠ Validation data not found: {val_path}")
    else:
        print(f"✓ Validation data: {val_path}")
        with open(val_path) as f:
            num_lines = sum(1 for _ in f)
        print(f"  → {num_lines} studies in validation set")
    
    # Image directory
    image_root = WORKSPACE_ROOT / config.data.image_root_dir
    if not image_root.exists():
        issues.append(f"Image directory not found: {image_root}")
    else:
        print(f"✓ Image directory: {image_root}")
        # Count image files
        num_images = sum(1 for _ in image_root.rglob("*.png"))
        num_images += sum(1 for _ in image_root.rglob("*.jpg"))
        num_images += sum(1 for _ in image_root.rglob("*.jpeg"))
        print(f"  → ~{num_images} image files found")
    
    if issues:
        print(f"\n❌ Found {len(issues)} path issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

def test_dataset_loading(config):
    """Test dataset loading."""
    print_section("DATASET LOADING TEST")
    
    train_path = WORKSPACE_ROOT / config.data.train_jsonl
    image_root = WORKSPACE_ROOT / config.data.image_root_dir
    
    try:
        dataset = ChestXrayReportDataset(
            train_path, 
            image_root, 
            max_images=config.data.max_images
        )
        print(f"✓ Dataset initialized: {len(dataset)} studies")
        print(f"✓ Max images per sample: {config.data.max_images}")
        
        # Test loading first sample
        print("\nLoading sample 0...")
        sample = dataset[0]
        
        if sample is None:
            print("❌ Sample 0 returned None (image loading failed)")
            return False
        
        print(f"✓ Sample loaded successfully")
        print(f"  - Images: {len(sample['images'])}")
        print(f"  - Image sizes: {[img.size for img in sample['images']]}")
        print(f"  - Messages: {len(sample['messages'])} roles")
        
        # Print message structure
        for msg in sample['messages']:
            role = msg['role']
            content_types = [c['type'] for c in msg['content']]
            print(f"  - {role}: {content_types}")
        
        # Test a few more samples
        print("\nTesting samples 1-5...")
        failures = 0
        for i in range(1, min(6, len(dataset))):
            s = dataset[i]
            if s is None:
                failures += 1
        
        if failures > 0:
            print(f"⚠ {failures}/5 samples failed to load")
        else:
            print(f"✓ All test samples loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading(config):
    """Test model and processor loading."""
    print_section("MODEL LOADING TEST")
    
    model_id = config.model.language_model_id
    print(f"Loading model: {model_id}")
    
    try:
        # Detect best precision
        bf16_support = torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if bf16_support else torch.float16
        
        print(f"BF16 supported: {bf16_support}")
        print(f"Using compute dtype: {compute_dtype}")
        
        # Create quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,  # Use BF16 if supported
            bnb_4bit_use_double_quant=True,
        )
        
        print("Loading model (this may take a few minutes)...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,  # Use BF16 if supported
        )
        print("✓ Model loaded")
        
        # Disable use_cache (incompatible with gradient checkpointing)
        model.config.use_cache = False
        
        # Freeze vision tower - saves ~4-6 GB during backward pass
        for param in model.model.vision_tower.parameters():
            param.requires_grad = False
        print("✓ Vision tower frozen")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        print("✓ Processor loaded")
        
        # Add special tokens
        special_tokens = ["<PA>", "<AP>", "<LATERAL>"]
        num_added = processor.tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        
        if num_added > 0:
            model.resize_token_embeddings(len(processor.tokenizer))
            print(f"✓ Added {num_added} special tokens")
        
        # Print model info
        print(f"\nModel info:")
        print(f"  - Device: {model.device}")
        print(f"  - Dtype: {model.dtype}")
        print(f"  - Vocab size: {len(processor.tokenizer)}")
        
        # Check memory after loading
        model_memory = torch.cuda.memory_allocated(0) / 1e9
        print(f"  - Model memory: {model_memory:.2f} GB")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_tokenization(model, processor, config):
    """Test tokenization and collation with DataCollatorForCompletionOnlyLM."""
    print_section("TOKENIZATION TEST (with DataCollatorForCompletionOnlyLM)")
    
    train_path = WORKSPACE_ROOT / config.data.train_jsonl
    image_root = WORKSPACE_ROOT / config.data.image_root_dir
    
    try:
        # Load dataset
        dataset = ChestXrayReportDataset(
            train_path, 
            image_root,
            max_images=config.data.max_images
        )
        sample = dataset[0]
        
        if sample is None:
            print("❌ Sample is None")
            return False
        
        # Get response template
        print("Detecting response template...")
        response_template = get_response_template(processor)
        print(f"✓ Response template: {repr(response_template)}")
        
        # Create DataCollatorForCompletionOnlyLM
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=processor.tokenizer,
            mlm=False,
        )
        
        # First, we need to format the data as the trainer would
        print("\nFormatting sample with chat template...")
        formatted_text = processor.apply_chat_template(
            sample["messages"],
            add_generation_prompt=False,
            tokenize=False
        ).strip()
        
        print(f"Formatted text length: {len(formatted_text)} chars")
        print(f"First 200 chars:\n{formatted_text[:200]}...")
        
        # Tokenize with processor (includes images)
        print("\nTokenizing with processor...")
        batch_inputs = processor(
            text=[formatted_text],
            images=[sample["images"]],
            return_tensors="pt",
            padding=True
        )
        
        print("✓ Batch tokenized")
        print(f"\nBatch keys: {list(batch_inputs.keys())}")
        for key, value in batch_inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
        
        # Now apply the collator to create labels
        # The collator expects a list of dicts with 'input_ids' and optionally 'attention_mask'
        print("\nApplying DataCollatorForCompletionOnlyLM...")
        
        # Prepare data in format expected by collator
        collator_input = [{
            "input_ids": batch_inputs["input_ids"][0],
            "attention_mask": batch_inputs["attention_mask"][0],
        }]
        
        batch = collator(collator_input)
        
        print("✓ Labels created by collator")
        
        # Add back the other keys (pixel_values, etc.)
        for key in ["pixel_values", "token_type_ids"]:
            if key in batch_inputs:
                batch[key] = batch_inputs[key]
        
        print(f"\nFinal batch contents:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
        
        # Verify labels - THIS IS THE CRITICAL CHECK
        if 'labels' in batch:
            labels = batch['labels'][0] if batch['labels'].dim() > 1 else batch['labels']
            num_mask = (labels == -100).sum().item()
            num_valid = (labels != -100).sum().item()
            total = num_mask + num_valid
            
            print(f"\n{'='*50}")
            print("CRITICAL LABEL CHECK:")
            print(f"{'='*50}")
            print(f"  - Total tokens: {total}")
            print(f"  - Masked tokens (-100): {num_mask}")
            print(f"  - Valid tokens: {num_valid}")
            print(f"  - Valid ratio: {num_valid/total*100:.1f}%")
            
            # Check if masking is correct
            if num_valid / total > 0.5:
                print(f"\n⚠️  WARNING: {num_valid/total*100:.1f}% valid tokens is TOO HIGH")
                print("   Expected: 20-40% (only assistant response should be valid)")
                print("   This suggests label masking may not be working correctly")
            elif num_valid / total < 0.15:
                print(f"\n⚠️  WARNING: {num_valid/total*100:.1f}% valid tokens is TOO LOW")
                print("   Expected: 20-40%")
            else:
                print(f"\n✅ CORRECT: {num_valid/total*100:.1f}% valid tokens is in expected range (20-40%)")
            print(f"{'='*50}\n")
        
        # Test batch of 2
        print("Testing batch of 2 samples...")
        sample2 = dataset[1]
        if sample2 is None:
            print("⚠ Second sample is None, skipping batch test")
        else:
            formatted2 = processor.apply_chat_template(
                sample2["messages"],
                add_generation_prompt=False,
                tokenize=False
            ).strip()
            
            batch_inputs2 = processor(
                text=[formatted_text, formatted2],
                images=[sample["images"], sample2["images"]],
                return_tensors="pt",
                padding=True
            )
            
            collator_input2 = [
                {
                    "input_ids": batch_inputs2["input_ids"][0],
                    "attention_mask": batch_inputs2["attention_mask"][0],
                },
                {
                    "input_ids": batch_inputs2["input_ids"][1],
                    "attention_mask": batch_inputs2["attention_mask"][1],
                }
            ]
            
            batch2 = collator(collator_input2)
            print(f"✓ Batch of 2 created: input_ids shape {batch2['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass(model, processor, config):
    """Test a single forward/backward pass."""
    print_section("FORWARD/BACKWARD PASS TEST")
    
    train_path = WORKSPACE_ROOT / config.data.train_jsonl
    image_root = WORKSPACE_ROOT / config.data.image_root_dir
    
    try:
        # Load sample
        dataset = ChestXrayReportDataset(
            train_path, 
            image_root,
            max_images=config.data.max_images
        )
        sample = dataset[0]
        
        if sample is None:
            print("❌ Sample is None")
            return False
        
        # Create batch using the same process as tokenization test
        response_template = get_response_template(processor)
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=processor.tokenizer,
            mlm=False,
        )
        
        formatted_text = processor.apply_chat_template(
            sample["messages"],
            add_generation_prompt=False,
            tokenize=False
        ).strip()
        
        batch_inputs = processor(
            text=[formatted_text],
            images=[sample["images"]],
            return_tensors="pt",
            padding=True
        )
        
        collator_input = [{
            "input_ids": batch_inputs["input_ids"][0],
            "attention_mask": batch_inputs["attention_mask"][0],
        }]
        
        batch = collator(collator_input)
        
        # Add back pixel_values
        for key in ["pixel_values", "token_type_ids"]:
            if key in batch_inputs:
                batch[key] = batch_inputs[key]
        
        # Move to GPU
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        print("Testing forward pass...")
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            outputs = model(**batch)
        
        forward_memory = torch.cuda.max_memory_allocated() / 1e9
        loss_value = outputs.loss.item()
        
        print(f"✓ Forward pass successful")
        print(f"  - Loss: {loss_value:.4f}")
        print(f"  - Memory used: {forward_memory:.2f} GB")
        
        # Check for NaN loss
        if torch.isnan(outputs.loss):
            print(f"\n❌ CRITICAL: Loss is NaN!")
            print("   This indicates a problem with label masking or data")
            return False
        else:
            print(f"  ✓ Loss is valid (not NaN)")
        
        # Test backward pass
        print("\nTesting backward pass...")
        torch.cuda.reset_peak_memory_stats()
        
        # Clear previous computation
        model.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        
        if torch.isnan(loss):
            print(f"❌ Loss is NaN, skipping backward pass")
            return False
        
        loss.backward()
        
        backward_memory = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"✓ Backward pass successful")
        print(f"  - Peak memory: {backward_memory:.2f} GB")
        
        # Cleanup
        del outputs, loss, batch
        torch.cuda.empty_cache()
        
        # Estimate full training memory
        estimated_training = backward_memory * 1.2  # Add buffer for optimizer states
        
        print(f"\n💾 Memory estimate:")
        print(f"  - Forward: ~{forward_memory:.2f} GB")
        print(f"  - Backward: ~{backward_memory:.2f} GB")
        print(f"  - Estimated training: ~{estimated_training:.2f} GB")
        
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        available_estimate = gpu_total - 17  # Account for other processes
        
        if estimated_training > available_estimate:
            print(f"  ⚠ Warning: Estimated usage ({estimated_training:.2f} GB) may exceed available ({available_estimate:.0f} GB)")
            print(f"     Consider reducing max_images or batch size")
        else:
            print(f"  ✓ Should fit in available memory (~{available_estimate:.0f} GB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print(f"\n{'#'*70}")
    print(f"{'MedGemma QLoRA Setup Validation (UPDATED)':^70}")
    print(f"{'#'*70}")
    
    config = TrainingConfig()
    
    # Show config
    print(f"\nConfiguration:")
    print(f"  - max_images: {config.data.max_images}")
    print(f"  - batch_size: {config.per_device_train_batch_size}")
    print(f"  - gradient_accumulation: {config.gradient_accumulation_steps}")
    print(f"  - optimizer: {config.optim}")
    
    # Run tests
    tests_passed = 0
    tests_total = 6
    
    # 1. GPU check
    if check_gpu():
        tests_passed += 1
    else:
        print("\n❌ GPU check failed - cannot proceed")
        sys.exit(1)
    
    # 2. Path validation
    if check_paths(config):
        tests_passed += 1
    else:
        print("\n❌ Path validation failed - cannot proceed")
        sys.exit(1)
    
    # 3. Dataset loading
    if test_dataset_loading(config):
        tests_passed += 1
    else:
        print("\n⚠ Dataset loading had issues - check errors above")
    
    # 4. Model loading
    model, processor = test_model_loading(config)
    if model is not None:
        tests_passed += 1
    else:
        print("\n❌ Model loading failed - cannot proceed with remaining tests")
        sys.exit(1)
    
    # 5. Tokenization (with proper label masking check)
    if test_tokenization(model, processor, config):
        tests_passed += 1
    else:
        print("\n⚠ Tokenization had issues - check errors above")
    
    # 6. Forward/backward pass
    if test_forward_pass(model, processor, config):
        tests_passed += 1
    else:
        print("\n⚠ Forward/backward pass had issues - check errors above")
    
    # Summary
    print_section("SUMMARY")
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED - Ready to train!")
        print("\nNext steps:")
        print("1. Review the memory estimate above")
        print("2. If memory looks good, start with small overfit test:")
        print("   - Set max_train_samples = 50 in config")
        print("   - Run: python qlora-medgemma-corrected.py")
        print("3. If overfit works, remove sample limit for full training")
    elif tests_passed >= 4:
        print("⚠ SOME TESTS PASSED - Review errors before training")
        print("\nYou may be able to proceed, but check warnings above")
        print("Especially check the label masking ratio (should be 20-40%)")
    else:
        print("❌ MULTIPLE TESTS FAILED - Fix issues before training")
        print("\nReview errors above and fix configuration")
    
    print(f"\n{'#'*70}\n")

if __name__ == "__main__":
    main()