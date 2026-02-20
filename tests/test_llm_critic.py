"""
Test Critic with Historical Past Mistakes Memory
"""

import numpy as np
from types import SimpleNamespace

from agents.critic.agent import critic_node
from graph.state import VerifaiState
from app.config import settings

# Ensure history memory is enabled
settings.ENABLE_PAST_MISTAKES_MEMORY = True
settings.ENABLE_LLM_CRITIC = False  # Start with rule-based + history only


def build_mock_state():
    """
    Build a high-risk pneumonia-style case that should
    retrieve similar historical mistakes.
    """

    # Simulate radiologist output (overconfident pneumonia)
    radiologist_output = SimpleNamespace(
        findings="Patchy opacity in the right lower lobe.",
        impression="Definite pneumonia in the right lower lobe."
    )

    # Simulate CheXbert output
    chexbert_output = SimpleNamespace(
        labels={"Pneumonia": "present"}
    )

    # Build VerifaiState
    state = VerifaiState()
    state["radiologist_output"] = radiologist_output
    state["radiologist_kle_uncertainty"] = 0.55  # high uncertainty
    state["chexbert_output"] = chexbert_output
    state["historian_output"] = None
    state["literature_output"] = None

    return state


def test_critic_with_history():
    print("\n=== Running Critic Historical Memory Test ===")

    state = build_mock_state()

    result = critic_node(state)

    critic_output = result["critic_output"]

    print("\n--- Critic Output ---")
    print("Overconfident:", critic_output.is_overconfident)
    print("Safety Score:", critic_output.safety_score)
    print("Similar Mistakes Count:", critic_output.similar_mistakes_count)
    print("Historical Risk Level:", critic_output.historical_risk_level)
    print("Concern Flags:")
    for flag in critic_output.concern_flags:
        print(" -", flag)

    print("\nTrace:")
    for t in result["trace"]:
        print(" -", t)


if __name__ == "__main__":
    test_critic_with_history()
