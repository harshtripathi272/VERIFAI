"""
Shared MedGemma Model Loader

Provides a thread-safe singleton for loading and sharing a single instance
of the MedGemma-4B model across multiple agents (Historian, Literature, Critic).

This prevents duplicate model loading and reduces VRAM usage from ~27GB to ~9GB.
"""

import threading
import torch
from typing import Optional, Tuple
from transformers import AutoProcessor, AutoModelForImageTextToText
from app.config import settings

# Global singleton cache
_MODEL_CACHE: Optional[Tuple] = None
_MODEL_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def load_shared_medgemma() -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """
    Load or retrieve the shared MedGemma-4B model instance.
    
    Thread-safe singleton pattern ensures only one model is loaded in memory
    and shared across Historian, Literature, and LLM Critic agents.
    
    Returns:
        Tuple of (model, processor)
    """
    global _MODEL_CACHE
    
    # Quick check without lock (performance optimization)
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    # Acquire lock for loading
    with _MODEL_LOAD_LOCK:
        # Double-check after acquiring lock
        if _MODEL_CACHE is not None:
            return _MODEL_CACHE
        
        print("[SharedModelLoader] Loading MedGemma-4B (16-bit FP16) - one-time initialization...")
        
        # Verify CUDA availability
        if not torch.cuda.is_available():
            print("[SharedModelLoader] WARNING: CUDA not available, falling back to CPU")
            device = "cpu"
            dtype = torch.float32
        else:
            device = torch.device("cuda:0")  # Explicitly use GPU 0
            dtype = torch.bfloat16  # MedGemma 1.5 prefers bfloat16
            print(f"[SharedModelLoader] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"[SharedModelLoader] CUDA memory before loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        # Use AutoProcessor instead of AutoTokenizer for MedGemma 1.5
        processor = AutoProcessor.from_pretrained(
            settings.MEDGEMMA_4B_MODEL,
            token=settings.HUGGINGFACE_TOKEN
        )
        
        print(f"[SharedModelLoader] Loading model to {device}...")
        # Use AutoModelForImageTextToText for MedGemma 1.5 (even for text-only)
        model = AutoModelForImageTextToText.from_pretrained(
            settings.MEDGEMMA_4B_MODEL,
            torch_dtype=dtype,
            token=settings.HUGGINGFACE_TOKEN,
            low_cpu_mem_usage=True
        )
        
        # Explicitly move to GPU if available
        if device != "cpu":
            print(f"[SharedModelLoader] Moving model to {device}...")
            model = model.to(device)
        
        _MODEL_CACHE = (model, processor)
        
        # Verify GPU placement
        if device != "cpu":
            print(f"[SharedModelLoader] Model loaded successfully on {device} with dtype={dtype}")
            print(f"[SharedModelLoader] CUDA memory after loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"[SharedModelLoader] Model device: {next(model.parameters()).device}")
        else:
            print(f"[SharedModelLoader] Model loaded on CPU (CUDA not available)")
        
        print("[SharedModelLoader] This model instance is shared by Historian, Literature, and Critic agents")
        
        return _MODEL_CACHE


def get_inference_lock() -> threading.Lock:
    """
    Get the global inference lock for thread-safe model usage.
    
    When multiple agents run in parallel (e.g., Historian + Literature),
    they must acquire this lock before calling model.generate().
    
    Returns:
        threading.Lock instance for model inference
    """
    return _INFERENCE_LOCK


def is_model_loaded() -> bool:
    """
    Check if the shared model is already loaded.
    
    Returns:
        True if model is loaded, False otherwise
    """
    return _MODEL_CACHE is not None
