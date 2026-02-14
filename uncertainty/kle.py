"""
Kernel Language Entropy (KLE) Uncertainty Calculator

Computes semantic uncertainty based on the KLE principle:
1. Generate/accept N sampled textual outputs
2. Convert each to semantic embeddings
3. Compute pairwise cosine similarity matrix
4. Normalize matrix (trace = 1)
5. Compute von Neumann entropy
6. Normalize to [0, 1] uncertainty score

Reference: Kernel Language Entropy paper (semantic uncertainty via density matrices)
"""

import threading
import numpy as np
from typing import List, Optional
from app.config import settings

# Thread-safe singleton for embedding model
_embedding_model = None
_LOAD_LOCK = threading.Lock()


def _load_embedding_model():
    """Lazy-load the text embedding model."""
    global _embedding_model

    if _embedding_model is not None:
        return

    with _LOAD_LOCK:
        if _embedding_model is not None:
            return

        if settings.MOCK_MODELS:
            print("[KLE] Running in MOCK mode - no embedding model loaded")
            return

        try:
            from sentence_transformers import SentenceTransformer
            print(f"[KLE] Loading embedding model: {settings.TEXT_EMBEDDING_MODEL}")
            _embedding_model = SentenceTransformer(settings.TEXT_EMBEDDING_MODEL)
            print("[KLE] Embedding model loaded successfully")
        except ImportError:
            print("[KLE] sentence-transformers not installed, falling back to mock")
        except Exception as e:
            print(f"[KLE] Failed to load embedding model: {e}")


def _get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts.

    Args:
        texts: List of text strings to embed

    Returns:
        numpy array of shape (N, embedding_dim)
    """
    _load_embedding_model()

    if settings.MOCK_MODELS or _embedding_model is None:
        # Return deterministic embeddings for mock mode
        # Each text gets an embedding based on its own hash
        # This ensures identical texts get identical embeddings
        embeddings = []
        for text in texts:
            # Use hash of individual text for per-text determinism
            text_hash = hash(text) % (2**32)
            np.random.seed(text_hash)
            embeddings.append(np.random.randn(384))  # 384 = MiniLM dimension
        return np.array(embeddings)

    embeddings = _embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings


def _compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (N, D) array of embeddings

    Returns:
        (N, N) cosine similarity matrix
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms

    # Cosine similarity = dot product of normalized vectors
    similarity = normalized @ normalized.T
    return similarity


def _normalize_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """
    Normalize kernel matrix so trace equals 1.

    This transforms the similarity matrix into a valid density matrix
    for von Neumann entropy computation.

    Args:
        K: (N, N) similarity matrix

    Returns:
        (N, N) normalized matrix with trace = 1
    """
    trace = np.trace(K)
    if trace == 0:
        # Degenerate case: return uniform distribution
        n = K.shape[0]
        return np.eye(n) / n
    return K / trace


def _compute_von_neumann_entropy(K: np.ndarray) -> float:
    """
    Compute von Neumann entropy of a density matrix.

    H(K) = -sum(lambda_i * log(lambda_i)) for eigenvalues lambda_i

    Args:
        K: (N, N) normalized density matrix (trace = 1)

    Returns:
        von Neumann entropy (non-negative real number)
    """
    # Get eigenvalues (K is symmetric, so use eigh)
    eigenvalues = np.linalg.eigvalsh(K)

    # Filter out non-positive eigenvalues (numerical noise)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    # Compute entropy: H = -sum(lambda * log(lambda))
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    return float(entropy)


def compute_semantic_uncertainty(texts: List[str]) -> float:
    """
    Compute semantic uncertainty using Kernel Language Entropy.

    This is the main public interface for the uncertainty calculator.

    Args:
        texts: List of N textual outputs to analyze for semantic consistency

    Returns:
        Uncertainty score in [0, 1]:
        - 0.0 = All texts are semantically identical (no uncertainty)
        - 1.0 = Texts are maximally diverse (maximum uncertainty)

    Example:
        >>> texts = ["Pneumonia detected", "Pneumonia observed", "Lung infection"]
        >>> uncertainty = compute_semantic_uncertainty(texts)
        >>> if uncertainty < 0.3:
        ...     print("Consensus reached")
    """
    if not texts:
        return 1.0  # Empty = maximum uncertainty (conservative)

    if len(texts) == 1:
        return 0.0  # Single text = no uncertainty

    # Step 1: Get embeddings
    embeddings = _get_embeddings(texts)

    # Step 2: Compute cosine similarity matrix
    K = _compute_cosine_similarity_matrix(embeddings)

    # Step 3: Normalize so trace = 1
    K_normalized = _normalize_kernel_matrix(K)

    # Step 4: Compute von Neumann entropy
    entropy = _compute_von_neumann_entropy(K_normalized)

    # Step 5: Normalize entropy to [0, 1]
    # Max entropy for N items = log(N)
    max_entropy = np.log(len(texts))
    if max_entropy == 0:
        return 0.0

    uncertainty = entropy / max_entropy

    # Clamp to [0, 1] (safety for numerical issues)
    uncertainty = max(0.0, min(1.0, uncertainty))

    return uncertainty


def compute_semantic_uncertainty_with_details(
    texts: List[str]
) -> dict:
    """
    Compute semantic uncertainty with full details for debugging/auditing.

    Returns dict with:
    - uncertainty: float in [0, 1]
    - entropy: raw von Neumann entropy
    - max_entropy: theoretical maximum
    - similarity_matrix: pairwise cosine similarities
    - num_samples: number of texts analyzed
    """
    if not texts:
        return {
            "uncertainty": 1.0,
            "entropy": 0.0,
            "max_entropy": 0.0,
            "similarity_matrix": [],
            "num_samples": 0
        }

    if len(texts) == 1:
        return {
            "uncertainty": 0.0,
            "entropy": 0.0,
            "max_entropy": 0.0,
            "similarity_matrix": [[1.0]],
            "num_samples": 1
        }

    embeddings = _get_embeddings(texts)
    K = _compute_cosine_similarity_matrix(embeddings)
    K_normalized = _normalize_kernel_matrix(K)
    entropy = _compute_von_neumann_entropy(K_normalized)
    max_entropy = np.log(len(texts))
    uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
    uncertainty = max(0.0, min(1.0, uncertainty))

    return {
        "uncertainty": uncertainty,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "similarity_matrix": K.tolist(),
        "num_samples": len(texts)
    }
