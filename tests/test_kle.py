"""
Unit tests for Kernel Language Entropy (KLE) uncertainty module.

Tests the compute_semantic_uncertainty function with various inputs.
"""

import sys
sys.path.insert(0, '.')

# pytest is optional - tests can run standalone
try:
    import pytest
except ImportError:
    pytest = None

from uncertainty.kle import (
    compute_semantic_uncertainty,
    compute_semantic_uncertainty_with_details,
    _compute_cosine_similarity_matrix,
    _normalize_kernel_matrix,
    _compute_von_neumann_entropy,
)
import numpy as np


class TestKLEBasics:
    """Basic functionality tests."""

    def test_empty_list_returns_max_uncertainty(self):
        """Empty input should return maximum uncertainty (conservative)."""
        result = compute_semantic_uncertainty([])
        assert result == 1.0

    def test_single_text_returns_zero_uncertainty(self):
        """Single text should have zero uncertainty."""
        result = compute_semantic_uncertainty(["Pneumonia detected"])
        assert result == 0.0

    def test_identical_texts_low_uncertainty(self):
        """Identical texts should have very low uncertainty.
        
        Note: In mock mode, this uses seeded random embeddings where identical
        inputs produce identical embeddings, so uncertainty should still be ~0.
        """
        texts = [
            "Community-acquired pneumonia",
            "Community-acquired pneumonia",
            "Community-acquired pneumonia",
        ]
        result = compute_semantic_uncertainty(texts)
        # In mock mode with seeded random: identical inputs = same seed = same embeddings = low entropy
        # The hash-based seed means identical text lists produce identical embeddings
        assert result < 0.5, f"Expected low uncertainty for identical texts, got {result}"

    def test_similar_texts_moderate_uncertainty(self):
        """Semantically similar texts should have low-moderate uncertainty."""
        texts = [
            "Pneumonia detected in right lower lobe",
            "Right lower lobe shows pneumonia",
            "Lung infection consistent with pneumonia",
        ]
        result = compute_semantic_uncertainty(texts)
        # Similar semantic content should cluster
        assert result < 0.5, f"Expected moderate uncertainty for similar texts, got {result}"

    def test_diverse_texts_high_uncertainty(self):
        """Completely different texts should have high uncertainty."""
        texts = [
            "Pneumonia detected",
            "No abnormality detected",
            "Consider cardiac evaluation",
            "Patient history suggests diabetes",
            "Recommend orthopedic consultation",
        ]
        result = compute_semantic_uncertainty(texts)
        # Very different content = high entropy
        assert result > 0.3, f"Expected high uncertainty for diverse texts, got {result}"

    def test_uncertainty_bounded_zero_one(self):
        """Uncertainty should always be in [0, 1]."""
        test_cases = [
            ["a", "b", "c", "d", "e"],
            ["same"] * 10,
            ["slightly different", "slightly different text"],
        ]
        for texts in test_cases:
            result = compute_semantic_uncertainty(texts)
            assert 0.0 <= result <= 1.0, f"Uncertainty {result} out of bounds for {texts}"


class TestKLEMathematics:
    """Tests for the underlying mathematical operations."""

    def test_cosine_similarity_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        embeddings = np.array([[1, 0, 0], [1, 0, 0]])
        sim = _compute_cosine_similarity_matrix(embeddings)
        assert np.allclose(sim, [[1, 1], [1, 1]])

    def test_cosine_similarity_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0."""
        embeddings = np.array([[1, 0], [0, 1]])
        sim = _compute_cosine_similarity_matrix(embeddings)
        assert np.allclose(sim, [[1, 0], [0, 1]])

    def test_normalize_kernel_trace_one(self):
        """Normalized kernel should have trace = 1."""
        K = np.array([[2, 0.5], [0.5, 3]])
        K_norm = _normalize_kernel_matrix(K)
        assert np.isclose(np.trace(K_norm), 1.0)

    def test_von_neumann_entropy_uniform(self):
        """Entropy of uniform distribution should be log(N)."""
        N = 4
        K = np.eye(N) / N  # Uniform distribution
        entropy = _compute_von_neumann_entropy(K)
        expected = np.log(N)
        assert np.isclose(entropy, expected, atol=1e-6)

    def test_von_neumann_entropy_pure_state(self):
        """Entropy of pure state (single eigenvalue = 1) should be 0."""
        K = np.zeros((3, 3))
        K[0, 0] = 1.0  # Pure state
        entropy = _compute_von_neumann_entropy(K)
        assert np.isclose(entropy, 0.0, atol=1e-10)


class TestKLEWithDetails:
    """Tests for the detailed output function."""

    def test_details_contains_required_fields(self):
        """Detailed output should contain all expected fields."""
        texts = ["text1", "text2", "text3"]
        result = compute_semantic_uncertainty_with_details(texts)
        
        assert "uncertainty" in result
        assert "entropy" in result
        assert "max_entropy" in result
        assert "similarity_matrix" in result
        assert "num_samples" in result

    def test_details_num_samples_correct(self):
        """num_samples should match input length."""
        texts = ["a", "b", "c", "d"]
        result = compute_semantic_uncertainty_with_details(texts)
        assert result["num_samples"] == 4

    def test_details_similarity_matrix_shape(self):
        """Similarity matrix should be N x N."""
        texts = ["a", "b", "c"]
        result = compute_semantic_uncertainty_with_details(texts)
        sim = result["similarity_matrix"]
        assert len(sim) == 3
        assert len(sim[0]) == 3


class TestKLEMockMode:
    """Tests that work in mock mode (no model loading)."""

    def test_deterministic_in_mock_mode(self):
        """Same inputs should give same outputs in mock mode."""
        from app.config import settings
        original = settings.MOCK_MODELS
        settings.MOCK_MODELS = True
        
        try:
            texts = ["test1", "test2", "test3"]
            result1 = compute_semantic_uncertainty(texts)
            result2 = compute_semantic_uncertainty(texts)
            # Due to seeded random, should be deterministic
            assert result1 == result2
        finally:
            settings.MOCK_MODELS = original


if __name__ == "__main__":
    # Run basic tests
    print("Running KLE unit tests...")
    
    # Basic tests
    test = TestKLEBasics()
    test.test_empty_list_returns_max_uncertainty()
    print("✓ Empty list returns max uncertainty")
    
    test.test_single_text_returns_zero_uncertainty()
    print("✓ Single text returns zero uncertainty")
    
    test.test_identical_texts_low_uncertainty()
    print("✓ Identical texts have low uncertainty")
    
    # Math tests
    math_test = TestKLEMathematics()
    math_test.test_cosine_similarity_identical_vectors()
    print("✓ Cosine similarity for identical vectors")
    
    math_test.test_normalize_kernel_trace_one()
    print("✓ Kernel normalization (trace = 1)")
    
    math_test.test_von_neumann_entropy_uniform()
    print("✓ Von Neumann entropy (uniform)")
    
    print("\n✅ All basic tests passed!")
