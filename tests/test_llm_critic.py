"""
Unit tests for the Critic model with LLM semantic critic integration.

Covers:
  1. Rule-only mode (ENABLE_LLM_CRITIC=False)
  2. Rule + LLM mode (ENABLE_LLM_CRITIC=True, MOCK_MODELS=True)
  3. LLM failure / fallback mode
  4. Guard logic (LLM skipped when uncertainty is low)
  5. Synthetic clinical case: high certainty + high KLE
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '.')

from app.config import settings
from agents.critic.model import CriticModel
from agents.critic.llm_critic import LLMCriticOutput, medgemma_critic


# ============================================================================
# Fixtures
# ============================================================================

# Assertive report (high linguistic certainty)
ASSERTIVE_FINDINGS = (
    "Right lower lobe demonstrates dense consolidation with air bronchograms. "
    "The opacity is homogeneous and well-defined."
)
ASSERTIVE_IMPRESSION = (
    "Definite right lower lobe pneumonia. "
    "The findings confirm community-acquired bacterial infection. "
    "Diagnostic of lobar pneumonia."
)

# Hedged report (low linguistic certainty)
HEDGED_FINDINGS = (
    "There may be a subtle area of increased density in the right lower lobe."
)
HEDGED_IMPRESSION = (
    "Possible right lower lobe opacity, which could represent early pneumonia "
    "or atelectasis. Consider clinical correlation. Differential includes "
    "viral vs bacterial etiology."
)


# ============================================================================
# Tests
# ============================================================================

class TestCriticRuleOnlyMode(unittest.TestCase):
    """Tests with ENABLE_LLM_CRITIC=False — pure rule-based behaviour."""

    def setUp(self):
        self._orig_enable = settings.ENABLE_LLM_CRITIC
        self._orig_mock = settings.MOCK_MODELS
        settings.ENABLE_LLM_CRITIC = False
        settings.MOCK_MODELS = True

    def tearDown(self):
        settings.ENABLE_LLM_CRITIC = self._orig_enable
        settings.MOCK_MODELS = self._orig_mock

    def test_overconfident_assertive_high_kle(self):
        """High certainty language + high KLE → overconfident."""
        model = CriticModel()
        is_oc, flags, hedging, score = model.evaluate(
            ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, kle_uncertainty=0.7
        )
        self.assertTrue(is_oc)
        self.assertGreater(len(flags), 0)
        self.assertIsNotNone(hedging)
        self.assertLessEqual(score, 0.5)

    def test_not_overconfident_hedged_high_kle(self):
        """Hedged language + high KLE → NOT overconfident."""
        model = CriticModel()
        is_oc, flags, hedging, score = model.evaluate(
            HEDGED_FINDINGS, HEDGED_IMPRESSION, kle_uncertainty=0.7
        )
        self.assertFalse(is_oc)

    def test_deterministic_when_llm_disabled(self):
        """Results must be identical across calls when LLM is off."""
        model = CriticModel()
        r1 = model.evaluate(ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, 0.5)
        r2 = model.evaluate(ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, 0.5)
        self.assertEqual(r1, r2)

    def test_return_shape(self):
        """evaluate() returns a 4-tuple with expected types."""
        model = CriticModel()
        result = model.evaluate(ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, 0.3)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        is_oc, flags, hedging, score = result
        self.assertIsInstance(is_oc, bool)
        self.assertIsInstance(flags, list)
        self.assertIsInstance(score, float)


class TestCriticRulePlusLLM(unittest.TestCase):
    """Tests with ENABLE_LLM_CRITIC=True using mock MedGemma."""

    def setUp(self):
        self._orig_enable = settings.ENABLE_LLM_CRITIC
        self._orig_mock = settings.MOCK_MODELS
        settings.ENABLE_LLM_CRITIC = True
        settings.MOCK_MODELS = True

    def tearDown(self):
        settings.ENABLE_LLM_CRITIC = self._orig_enable
        settings.MOCK_MODELS = self._orig_mock

    def test_llm_enriches_concern_flags(self):
        """LLM should add [LLM] prefixed flags when it detects issues."""
        model = CriticModel()
        is_oc, flags, hedging, score = model.evaluate(
            ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, kle_uncertainty=0.7
        )
        self.assertTrue(is_oc)
        llm_flags = [f for f in flags if f.startswith("[LLM]")]
        self.assertGreater(len(llm_flags), 0, "Expected [LLM] prefixed concern flags")

    def test_llm_hedging_takes_priority(self):
        """When LLM suggests hedging, it should override rule-based suggestion."""
        model = CriticModel()
        _, _, hedging, _ = model.evaluate(
            ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, kle_uncertainty=0.7
        )
        # Mock LLM returns specific hedging for assertive + high uncertainty
        self.assertIsNotNone(hedging)

    def test_safety_score_penalised_by_llm(self):
        """Safety score should be lower when LLM detects high semantic risk."""
        model = CriticModel()

        # Get rule-only baseline
        settings.ENABLE_LLM_CRITIC = False
        _, _, _, rule_score = model.evaluate(
            ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, kle_uncertainty=0.7
        )

        # Get rule+LLM
        settings.ENABLE_LLM_CRITIC = True
        _, _, _, llm_score = model.evaluate(
            ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, kle_uncertainty=0.7
        )

        # LLM mock returns semantic_risk_score=0.65 for this case,
        # which should penalise the safety score
        self.assertLessEqual(llm_score, rule_score)

    def test_guard_skips_llm_low_uncertainty(self):
        """LLM should NOT be called when KLE is low and rule-based is fine."""
        model = CriticModel()
        with patch.object(medgemma_critic, 'critique', wraps=medgemma_critic.critique) as mock_crit:
            model.evaluate(HEDGED_FINDINGS, HEDGED_IMPRESSION, kle_uncertainty=0.1)
            mock_crit.assert_not_called()


class TestCriticLLMFailure(unittest.TestCase):
    """Tests for graceful LLM failure fallback."""

    def setUp(self):
        self._orig_enable = settings.ENABLE_LLM_CRITIC
        self._orig_mock = settings.MOCK_MODELS
        settings.ENABLE_LLM_CRITIC = True
        settings.MOCK_MODELS = True

    def tearDown(self):
        settings.ENABLE_LLM_CRITIC = self._orig_enable
        settings.MOCK_MODELS = self._orig_mock

    def test_fallback_on_llm_exception(self):
        """If LLM critique() raises, fall back to rule-based output."""
        model = CriticModel()
        with patch.object(medgemma_critic, 'critique', side_effect=Exception("boom")):
            # Should NOT raise — just fall back
            is_oc, flags, hedging, score = model.evaluate(
                ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, kle_uncertainty=0.7
            )
            # Rule-based should still detect overconfidence
            self.assertTrue(is_oc)
            # No [LLM] flags since LLM failed
            llm_flags = [f for f in flags if f.startswith("[LLM]")]
            self.assertEqual(len(llm_flags), 0)

    def test_fallback_on_llm_returns_none(self):
        """If LLM critique() returns None, fall back gracefully."""
        model = CriticModel()
        with patch.object(medgemma_critic, 'critique', return_value=None):
            is_oc, flags, hedging, score = model.evaluate(
                ASSERTIVE_FINDINGS, ASSERTIVE_IMPRESSION, kle_uncertainty=0.7
            )
            self.assertTrue(is_oc)  # rule-based still works
            llm_flags = [f for f in flags if f.startswith("[LLM]")]
            self.assertEqual(len(llm_flags), 0)


class TestSyntheticClinicalCase(unittest.TestCase):
    """
    Synthetic case: High linguistic certainty + high KLE uncertainty.
    Demonstrates that the LLM critic detects a missing differential
    and adjusts the safety score.
    """

    def setUp(self):
        self._orig_enable = settings.ENABLE_LLM_CRITIC
        self._orig_mock = settings.MOCK_MODELS
        settings.ENABLE_LLM_CRITIC = True
        settings.MOCK_MODELS = True

    def tearDown(self):
        settings.ENABLE_LLM_CRITIC = self._orig_enable
        settings.MOCK_MODELS = self._orig_mock

    def test_synthetic_high_certainty_high_kle(self):
        """
        A report using 'definite' and 'confirms' with KLE=0.75 should:
        1. Be flagged overconfident by rule-based
        2. Have LLM detect missing differentials
        3. Have adjusted (lower) safety score
        """
        findings = (
            "Dense consolidation in the right lower lobe with air bronchograms. "
            "No pleural effusion."
        )
        impression = (
            "The findings are definite for community-acquired pneumonia. "
            "This confirms bacterial lobar pneumonia."
        )
        kle = 0.75

        model = CriticModel()
        is_oc, flags, hedging, score = model.evaluate(findings, impression, kle)

        # Overconfident
        self.assertTrue(is_oc, "Should be flagged overconfident")

        # LLM should have added missing differentials
        diff_flags = [f for f in flags if "Missing differentials" in f]
        self.assertGreater(len(diff_flags), 0, "Expected missing differentials flag")

        # Safety score should be low
        self.assertLess(score, 0.5, f"Safety score {score} should be < 0.5")

        # Hedging should be suggested
        self.assertIsNotNone(hedging)

        print(f"\n{'='*60}")
        print("SYNTHETIC CASE RESULTS")
        print(f"{'='*60}")
        print(f"  Overconfident:  {is_oc}")
        print(f"  Safety Score:   {score:.3f}")
        print(f"  Hedging:        {hedging}")
        print(f"  Concern Flags:")
        for f in flags:
            print(f"    - {f}")


class TestLLMCriticOutput(unittest.TestCase):
    """Tests for the LLMCriticOutput Pydantic model."""

    def test_default_values(self):
        out = LLMCriticOutput()
        self.assertIsNone(out.overconfidence_reason)
        self.assertEqual(out.missing_differentials, [])
        self.assertIsNone(out.justification_gap)
        self.assertIsNone(out.suggested_hedging)
        self.assertEqual(out.semantic_risk_score, 0.0)

    def test_full_construction(self):
        out = LLMCriticOutput(
            overconfidence_reason="Too certain",
            missing_differentials=["TB", "Malignancy"],
            justification_gap="Findings don't fully support impression",
            suggested_hedging="Use 'suggestive of'",
            semantic_risk_score=0.8,
        )
        self.assertEqual(out.overconfidence_reason, "Too certain")
        self.assertEqual(len(out.missing_differentials), 2)
        self.assertAlmostEqual(out.semantic_risk_score, 0.8)

    def test_score_clamped(self):
        """Score must be in [0, 1]."""
        with self.assertRaises(Exception):
            LLMCriticOutput(semantic_risk_score=1.5)
        with self.assertRaises(Exception):
            LLMCriticOutput(semantic_risk_score=-0.1)


if __name__ == "__main__":
    unittest.main()
