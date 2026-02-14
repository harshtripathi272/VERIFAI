"""
Uncertainty module for VERIFAI

Provides external, deterministic semantic uncertainty estimation
using Kernel Language Entropy (KLE).
"""

from .kle import (
    compute_semantic_uncertainty,
    compute_semantic_uncertainty_with_details,
)

__all__ = [
    "compute_semantic_uncertainty",
    "compute_semantic_uncertainty_with_details",
]
