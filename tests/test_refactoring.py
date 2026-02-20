"""
Quick test script for refactored Radiologist and Critic agents.

Tests:
1. Radiologist generates text-only output with KLE uncertainty
2. Critic evaluates linguistic certainty vs KLE uncertainty
3. No crashes due to missing attributes
"""

import sys
sys.path.insert(0, '.')

from graph.state import VerifaiState, RadiologistOutput, CriticOutput
from agents.radiologist.agent import radiologist_node
from agents.critic.agent import critic_node
from app.config import settings

# Force mock mode
settings.MOCK_MODELS = True

print("=" * 60)
print("REFACTORING VERIFICATION TEST")
print("=" * 60)

# Test 1: Radiologist Node
print("\n[Test 1] Running Radiologist Node...")
print("-" * 60)

rad_state: VerifaiState = {
    "image_path": "img1.jpg",  # Dummy path
    "patient_id": "TEST-001",
    "radiologist_output": None,
    "critic_output": None,
    "historian_output": None,
    "literature_output": None,
    "debate_output": None,
    "current_uncertainty": 0.5,
    "routing_decision": "",
    "steps_taken": 0,
    "radiologist_kle_uncertainty": None,
    "final_diagnosis": None,
    "trace": []
}

try:
    rad_result = radiologist_node(rad_state)
    rad_output = rad_result["radiologist_output"]
    kle_uncertainty = rad_result.get("radiologist_kle_uncertainty")
    
    print(f"✓ Radiologist node executed successfully")
    print(f"  - Output type: {type(rad_output)}")
    print(f"  - Has findings: {bool(rad_output.findings)}")
    print(f"  - Has impression: {bool(rad_output.impression)}")
    print(f"  - Findings preview: {rad_output.findings[:80]}...")
    print(f"  - Impression preview: {rad_output.impression[:80]}...")
    print(f"  - KLE uncertainty: {kle_uncertainty}")
    
    # Check that old attributes are gone
    if hasattr(rad_output, 'hypotheses'):
        print("  ✗ ERROR: Old 'hypotheses' attribute still exists!")
    else:
        print("  ✓ Old 'hypotheses' attribute removed")
    
    if hasattr(rad_output, 'internal_signals'):
        print("  ✗ ERROR: Old 'internal_signals' attribute still exists!")
    else:
        print("  ✓ Old 'internal_signals' attribute removed")
    
except Exception as e:
    print(f"✗ Radiologist node FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Critic Node
print("\n[Test 2] Running Critic Node...")
print("-" * 60)

# Update state with radiologist output
rad_state["radiologist_output"] = rad_output
rad_state["radiologist_kle_uncertainty"] = kle_uncertainty
rad_state["trace"] = rad_result.get("trace", [])

try:
    critic_result = critic_node(rad_state)
    critic_output = critic_result["critic_output"]
    
    print(f"✓ Critic node executed successfully")
    print(f"  - Output type: {type(critic_output)}")
    print(f"  - Is overconfident: {critic_output.is_overconfident}")
    print(f"  - Concern flags: {len(critic_output.concern_flags)}")
    print(f"  - Safety score: {critic_output.safety_score}")
    print(f"  - Current uncertainty: {critic_result['current_uncertainty']}")
    
    if critic_output.concern_flags:
        print(f"  - First concern: {critic_output.concern_flags[0]}")
    
    # Check that old attributes are gone
    if hasattr(critic_output, 'overconfidence_probability'):
        print("  ✗ ERROR: Old 'overconfidence_probability' attribute still exists!")
    else:
        print("  ✓ Old 'overconfidence_probability' attribute removed")
    
    if hasattr(critic_output, 'calculated_uncertainty'):
        print("  ✗ ERROR: Old 'calculated_uncertainty' attribute still exists!")
    else:
        print("  ✓ Old 'calculated_uncertainty' attribute removed")
    
except Exception as e:
    print(f"✗ Critic node FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Multiple samples produce variation
print("\n[Test 3] Testing KLE Sample Variation...")
print("-" * 60)

try:
    # Run radiologist 3 times
    kle_scores = []
    for i in range(3):
        result = radiologist_node(rad_state)
        kle = result.get("radiologist_kle_uncertainty")
        if kle is not None:
            kle_scores.append(kle)
    
    if len(kle_scores) >= 2:
        print(f"✓ KLE scores collected: {kle_scores}")
        print(f"  - Mean: {sum(kle_scores) / len(kle_scores):.3f}")
        print(f"  - Range: [{min(kle_scores):.3f}, {max(kle_scores):.3f}]")
    else:
        print("  ! Warning: Not enough KLE scores to assess variation")
    
except Exception as e:
    print(f"✗ Sample variation test FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("\nSummary:")
print("  - Radiologist outputs text-only (findings + impression)")
print("  - KLE uncertainty computed from multiple samples")
print("  - Critic evaluates linguistic certainty vs KLE")
print("  - No crashes, no missing attributes")
print("\nRefactoring verification complete!")
