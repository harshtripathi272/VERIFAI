"""
Uncertainty module for VERIFAI

Provides the Monotonic Uncertainty Cascade (MUC) framework:
- Information Gain (IG) computation per agent
- Token entropy for radiologist prior
- Dempster-Shafer fusion for debate consensus
- Per-agent uncertainty and alignment helpers

Legacy KLE utilities are retained for case_embedding similarity search only.
"""

# === MUC System (primary) ===
from .muc import (
    compute_ig,
    compute_token_entropy,
    compute_token_entropy_from_text,
    compute_chexbert_uncertainty,
    compute_chexbert_alignment,
    compute_historian_uncertainty,
    compute_historian_alignment,
    compute_literature_uncertainty,
    compute_literature_alignment,
    compute_critic_uncertainty,
    compute_critic_alignment,
    compute_debate_ds_fusion,
    compute_validator_uncertainty,
    compute_validator_alignment,
    build_mass_function,
    dempster_combine,
    IGResult,
    CascadeResult,
    SCALING_FACTORS,
)

# === Legacy KLE (kept for case_embedding only) ===
from .kle import (
    compute_semantic_uncertainty,
    compute_semantic_uncertainty_with_details,
)

__all__ = [
    # MUC
    "compute_ig",
    "compute_token_entropy",
    "compute_token_entropy_from_text",
    "compute_chexbert_uncertainty",
    "compute_chexbert_alignment",
    "compute_historian_uncertainty",
    "compute_historian_alignment",
    "compute_literature_uncertainty",
    "compute_literature_alignment",
    "compute_critic_uncertainty",
    "compute_critic_alignment",
    "compute_debate_ds_fusion",
    "compute_validator_uncertainty",
    "compute_validator_alignment",
    "build_mass_function",
    "dempster_combine",
    "IGResult",
    "CascadeResult",
    "SCALING_FACTORS",
    # Legacy
    "compute_semantic_uncertainty",
    "compute_semantic_uncertainty_with_details",
]
