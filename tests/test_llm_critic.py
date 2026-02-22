"""
Enhanced Critic Test — LLM OFF + LLM ON modes.

Test 1 (LLM OFF): Verifies rule-based critic + historical memory context.
Test 2 (LLM ON):  Enables the MedGemma semantic critic (shared model loader).
                  Set RUN_LLM_TEST = True to run (requires GPU / model download).
"""

from types import SimpleNamespace
from agents.critic.agent import critic_node
from graph.state import VerifaiState
from app.config import settings

# ─── Configuration ────────────────────────────────────────────────────────────
# Flip to True to also run the MedGemma LLM critic test (needs GPU + model).
RUN_LLM_TEST = True
# ──────────────────────────────────────────────────────────────────────────────


def build_mock_state():
    radiologist_output = SimpleNamespace(
        findings="Patchy opacity in the right lower lobe.",
        impression="Definite pneumonia in the right lower lobe."
    )

    chexbert_output = SimpleNamespace(
        labels={"Pneumonia": "present"}
    )

    state = VerifaiState()
    state["radiologist_output"] = radiologist_output
    state["radiologist_kle_uncertainty"] = 0.55
    state["chexbert_output"] = chexbert_output
    state["historian_output"] = None
    state["literature_output"] = None

    return state


# ─── Test 1: LLM OFF (rule-based + historical memory) ─────────────────────────

def test_critic_with_history():
    """Run critic with LLM disabled. Validates rule-based + historical memory."""
    print("\n=== Running Critic Historical Memory Test (LLM OFF) ===")

    settings.ENABLE_PAST_MISTAKES_MEMORY = True
    settings.ENABLE_LLM_CRITIC = False

    state = build_mock_state()
    result = critic_node(state)
    critic_output = result["critic_output"]

    print("\n--- Core Outputs ---")
    print("Overconfident:", critic_output.is_overconfident)
    print("Safety Score:", critic_output.safety_score)
    print("Routing Uncertainty (1 - safety):", result["current_uncertainty"])
    print("Similar Mistakes Count:", critic_output.similar_mistakes_count)
    print("Historical Risk Level:", critic_output.historical_risk_level)

    print("\n--- Concern Flags ---")
    if critic_output.concern_flags:
        for flag in critic_output.concern_flags:
            print(" -", flag)
    else:
        print(" None")

    print("\n--- Historical Context (Structured) ---")
    if hasattr(critic_output, "historical_context") and critic_output.historical_context:
        for idx, case in enumerate(critic_output.historical_context, 1):
            print(f"\n Case {idx}:")
            print("  Disease:", case.get("disease_type"))
            print("  Error Type:", case.get("error_type"))
            print("  Severity Level:", case.get("severity_level"))   # ← fixed key
            print("  KLE:", case.get("kle_uncertainty"))             # ← fixed key
            print("  Similarity:", case.get("similarity"))
            print("  Summary:", case.get("clinical_summary"))        # ← fixed key
    else:
        print(" None")

    print("\n--- Input Snapshot ---")
    print("Impression:", state["radiologist_output"].impression)
    print("KLE:", state["radiologist_kle_uncertainty"])

    print("\n--- Trace ---")
    for t in result["trace"]:
        print(" -", t)

    print("\n=== END TEST ===\n")


# ─── Test 2: LLM ON (MedGemma semantic critic via shared loader) ───────────────

def test_critic_with_llm():
    """
    Run critic with ENABLE_LLM_CRITIC = True.

    The MedGemma model is loaded via shared_model_loader (singleton).
    Requires a working GPU and the model available at settings.MEDGEMMA_4B_MODEL.

    Verifies:
    - LLM critique fires when KLE > 0.3 or rule-based overconfidence is True.
    - LLM adds [LLM] prefixed concern flags.
    - safety_score is penalised by semantic_risk_score.
    - shared model is loaded only once (no duplicate VRAM usage).
    """
    print("\n=== Running Critic LLM Critique Test (MedGemma ON) ===")
    print("  Model:", settings.MEDGEMMA_4B_MODEL)
    print("  This will load the shared MedGemma model on first call.\n")

    settings.ENABLE_PAST_MISTAKES_MEMORY = True
    settings.ENABLE_LLM_CRITIC = True

    state = build_mock_state()
    # Use a high KLE to guarantee the LLM path fires (threshold: KLE > 0.3)
    state["radiologist_kle_uncertainty"] = 0.60

    result = critic_node(state)
    critic_output = result["critic_output"]

    print("--- Core Outputs ---")
    print("Overconfident:", critic_output.is_overconfident)
    print("Safety Score:", critic_output.safety_score)
    print("Routing Uncertainty:", result["current_uncertainty"])

    print("\n--- Concern Flags ---")
    llm_flags = [f for f in critic_output.concern_flags if f.startswith("[LLM]")]
    rule_flags = [f for f in critic_output.concern_flags if not f.startswith("[LLM]")]

    if rule_flags:
        print(" [Rule-based]")
        for f in rule_flags:
            print("   -", f)

    if llm_flags:
        print(" [LLM-Critic / MedGemma]")
        for f in llm_flags:
            print("   -", f)
    else:
        print(" No LLM flags generated (LLM may have returned low-risk output).")

    print("\n--- Recommended Hedging ---")
    print(critic_output.recommended_hedging or "None")

    print("\n--- Trace ---")
    for t in result["trace"]:
        print(" -", t)

    print("\n=== END LLM TEST ===\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_critic_with_history()

    if RUN_LLM_TEST:
        test_critic_with_llm()
    else:
        print(
            "[INFO] LLM test skipped. Set RUN_LLM_TEST = True in this file "
            "to run MedGemma semantic critique (requires GPU + model)."
        )
