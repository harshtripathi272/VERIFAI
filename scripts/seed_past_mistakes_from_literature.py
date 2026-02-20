"""
Test Critic Historical Memory (Aligned with Seeded Literature Cases)
"""

import numpy as np
from types import SimpleNamespace
from sentence_transformers import SentenceTransformer

from agents.critic.agent import critic_node
from graph.state import VerifaiState
from app.config import settings

# Enable historical memory
settings.ENABLE_PAST_MISTAKES_MEMORY = True
settings.ENABLE_LLM_CRITIC = False

# Use same embedding model as seeding
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_mock_state():
    """
    Pneumonia-style case similar to seeded mistake:
    'Early pneumonia misinterpreted as atelectasis'
    """

    findings = (
        "Patchy opacity in the right lower lobe with overlapping densities."
    )

    impression = (
        "Likely pneumonia in the right lower lobe."
    )

    radiologist_output = SimpleNamespace(
        findings=findings,
        impression=impression
    )

    chexbert_output = SimpleNamespace(
        labels={"Pneumonia": "present"}
    )

    state = VerifaiState()
    state["radiologist_output"] = radiologist_output
    state["radiologist_kle_uncertainty"] = 0.55  # within normal test range
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

    print("\nConcern Flags:")
    for flag in critic_output.concern_flags:
        print(" -", flag)

    print("\nTrace:")
    for t in result["trace"]:
        print(" -", t)


if __name__ == "__main__":
    test_critic_with_history()
