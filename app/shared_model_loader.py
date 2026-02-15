"""
Shared MedGemma Model Loader

Provides a thread-safe singleton for loading and sharing a single instance
of the MedGemma-4B model across multiple agents (Historian, Literature, Critic).

This prevents duplicate model loading and reduces VRAM usage from ~27GB to ~9GB.
"""

import threading
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import settings

# Global singleton cache
_MODEL_CACHE: Optional[Tuple] = None
_MODEL_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def load_shared_medgemma() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load or retrieve the shared MedGemma-4B model instance.
    
    Thread-safe singleton pattern ensures only one model is loaded in memory
    and shared across Historian, Literature, and LLM Critic agents.
    
    Returns:
        Tuple of (model, tokenizer)
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
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        tokenizer = AutoTokenizer.from_pretrained(
            settings.MEDGEMMA_4B_MODEL,
            token=settings.HUGGINGFACE_TOKEN
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            settings.MEDGEMMA_4B_MODEL,
            device_map="auto",
            torch_dtype=dtype,
            token=settings.HUGGINGFACE_TOKEN
        )
        
        _MODEL_CACHE = (model, tokenizer)
        print(f"[SharedModelLoader] Model loaded successfully on {device} with dtype={dtype}")
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
