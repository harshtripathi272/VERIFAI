"""
Simple script to log in to Hugging Face Hub and verify GPU.

Usage:
    1. Make sure you have a valid HF token below
    2. Run: python hf_login.py
"""

import torch
import sys
from huggingface_hub import login

# Workaround for Windows terminal encoding issues
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


# Your Hugging Face token (already set)

token = "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"

print("\n" + "=" * 60)
print("SYSTEM CHECK & HF LOGIN")
print("=" * 60)

# Check GPU
print("\n[1/2] GPU Check:")
if torch.cuda.is_available():
    print(f"  ✓ CUDA Available: YES")
    print(f"  GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"    Total Memory: {mem_total:.2f} GB")
else:
    print("  ✗ CUDA Available: NO")
    print("  WARNING: Training will be EXTREMELY slow on CPU!")

# Login to HF
print("\n[2/2] Hugging Face Login:")
try:
    login(token=token)
    print("  ✓ Successfully logged in to Hugging Face!")
except Exception as e:
    print(f"  ✗ Login failed: {e}")
    print("\nMake sure you've accepted the MedGemma license:")
    print("  → https://huggingface.co/google/medgemma-4b-it")
    print("=" * 60 + "\n")
    exit(1)

print("\n" + "=" * 60)
print("✓ ALL CHECKS PASSED")
print("=" * 60)
print("\nYou can now run:")
print("  python scripts/overfit_test.py")
print("  python scripts/train.py")
print("=" * 60 + "\n")