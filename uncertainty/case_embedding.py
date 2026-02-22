"""
Case Embedding Generator

Generates semantic embeddings for diagnostic cases to enable similarity search
across past mistakes. Reuses the sentence-transformers embedding infrastructure.

Compact case summaries are created from diagnostic fields and embedded using
the same model (all-MiniLM-L6-v2) to produce 384-dimensional vectors.
"""

import numpy as np
from typing import Optional, Dict

from uncertainty.kle import _get_embeddings


def generate_case_summary(
    disease_type: str,
    original_diagnosis: str,
    corrected_diagnosis: str,
    error_type: str,
    uncertainty_score: Optional[float] = None,
    chexbert_labels: Optional[Dict[str, str]] = None,
    clinical_summary: Optional[str] = None,
    debate_summary: Optional[str] = None
) -> str:
    """
    Generate a compact textual summary of a diagnostic case for embedding.
    
    The summary format emphasizes:
    - Disease category (for filtering)
    - Diagnostic error pattern (original → corrected)
    - Error classification
    - Clinical context (abbreviated)
    - Key findings (abbreviated)
    - Uncertainty metrics
    
    Args:
        disease_type: Primary pathology category (e.g., 'pneumonia')
        original_diagnosis: Incorrect diagnosis that was made
        corrected_diagnosis: Validated correct diagnosis
        error_type: Classification of error
        uncertainty_score: MUC system uncertainty score (optional)
        chexbert_labels: CheXpert labels dict (optional)
        clinical_summary: Clinical context (optional, truncated to 200 chars)
        debate_summary: Debate/reasoning summary (optional, truncated to 200 chars)
    
    Returns:
        Compact case summary string suitable for embedding
    """
    parts = []
    
    # Core diagnostic information
    parts.append(f"Disease: {disease_type}")
    parts.append(f"Original Diagnosis: {original_diagnosis}")
    parts.append(f"Corrected Diagnosis: {corrected_diagnosis}")
    parts.append(f"Error Type: {error_type}")
    
    # Clinical context (abbreviated)
    if clinical_summary:
        truncated = clinical_summary[:200].strip()
        if len(clinical_summary) > 200:
            truncated += "..."
        parts.append(f"Clinical Context: {truncated}")
    
    # Findings pattern (abbreviated)
    if debate_summary:
        truncated = debate_summary[:200].strip()
        if len(debate_summary) > 200:
            truncated += "..."
        parts.append(f"Findings Pattern: {truncated}")
    
    # Uncertainty metrics
    if uncertainty_score is not None:
        parts.append(f"Uncertainty Score: {uncertainty_score:.2f}")
    
    # CheXbert labels (only present/uncertain ones)
    if chexbert_labels:
        present = [k for k, v in chexbert_labels.items() if v in ['present', 'uncertain']]
        if present:
            parts.append(f"CheXbert Labels: {', '.join(present)}")
    
    return "\n\n".join(parts)


def generate_case_embedding(case_summary: str) -> np.ndarray:
    """
    Generate a semantic embedding for a case summary.
    
    Uses the same sentence-transformers model as the embedding utility
    (configured via settings.TEXT_EMBEDDING_MODEL).
    
    Args:
        case_summary: Textual case summary (from generate_case_summary)
    
    Returns:
        384-dimensional numpy array embedding
    
    Example:
        >>> summary = generate_case_summary(
        ...     disease_type='pneumonia',
        ...     original_diagnosis='Normal chest X-ray',
        ...     corrected_diagnosis='Community-Acquired Pneumonia (RLL)',
        ...     error_type='misdiagnosis',
        ...     uncertainty_score=0.35
        ... )
        >>> embedding = generate_case_embedding(summary)
        >>> embedding.shape
        (384,)
    """
    # Use the shared embedding function (handles model loading, mock mode, etc.)
    embeddings = _get_embeddings([case_summary])
    return embeddings[0]  # Return single embedding


def generate_case_embedding_from_fields(
    disease_type: str,
    original_diagnosis: str,
    corrected_diagnosis: str,
    error_type: str,
    uncertainty_score: Optional[float] = None,
    chexbert_labels: Optional[Dict[str, str]] = None,
    clinical_summary: Optional[str] = None,
    debate_summary: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function: generate embedding directly from case fields.
    
    Combines generate_case_summary() and generate_case_embedding().
    
    Args:
        Same as generate_case_summary()
    
    Returns:
        384-dimensional numpy array embedding
    """
    summary = generate_case_summary(
        disease_type=disease_type,
        original_diagnosis=original_diagnosis,
        corrected_diagnosis=corrected_diagnosis,
        error_type=error_type,
        uncertainty_score=uncertainty_score,
        chexbert_labels=chexbert_labels,
        clinical_summary=clinical_summary,
        debate_summary=debate_summary
    )
    return generate_case_embedding(summary)
