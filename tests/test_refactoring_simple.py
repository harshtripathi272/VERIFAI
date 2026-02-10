"""
Simplified inline test for refactored agents.
"""

import sys
sys.path.insert(0, '.')

print("=" * 60)
print("REFACTORING VERIFICATION - SIMPLE TEST")
print("=" * 60)

# Test imports
print("\n[1] Testing imports...")
try:
    from graph.state import RadiologistOutput, CriticOutput
    from agents.radiologist.model import generate_findings
    from uncertainty.kle import compute_semantic_uncertainty
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test RadiologistOutput structure
print("\n[2] Testing RadiologistOutput structure...")
try:
    test_output = RadiologistOutput(
        findings="Test findings text",
        impression="Test impression text"
    )
    print(f"✓ RadiologistOutput created successfully")
    print(f"  - findings: {test_output.findings[:30]}...")
    print(f"  - impression: {test_output.impression[:30]}...")
    
    # Check old attributes are gone
    assert not hasattr(test_output, 'hypotheses'), "Old 'hypotheses' still exists!"
    assert not hasattr(test_output, 'internal_signals'), "Old 'internal_signals' still exists!"
    print("✓ Old attributes removed")
except Exception as e:
    print(f"✗ RadiologistOutput test failed: {e}")
    sys.exit(1)

# Test CriticOutput structure
print("\n[3] Testing CriticOutput structure...")
try:
    test_critic_output = CriticOutput(
        is_overconfident=False,
        concern_flags=["test concern"],
        recommended_hedging=None,
        safety_score=0.8
    )
    print(f"✓ CriticOutput created successfully")
    print(f"  - is_overconfident: {test_critic_output.is_overconfident}")
    print(f"  - safety_score: {test_critic_output.safety_score}")
    
    # Check old attributes are gone
    assert not hasattr(test_critic_output, 'overconfidence_probability'), "Old 'overconfidence_probability' still exists!"
    assert not hasattr(test_critic_output, 'calculated_uncertainty'), "Old 'calculated_uncertainty' still exists!"
    print("✓ Old attributes removed")
except Exception as e:
    print(f"✗ CriticOutput test failed: {e}")
    sys.exit(1)

# Test Model Generation
print("\n[4] Testing model generation...")
try:
    from app.config import settings
    settings.MOCK_MODELS = True
    
    result = generate_findings(None, {})
    print(f"✓ generate_findings executed")
    print(f"  - Type: {type(result)}")
    print(f"  - Has 'findings': {'findings' in result}")
    print(f"  - Has 'impression': {'impression' in result}")
    print(f"  - Findings preview: {result['findings'][:50]}...")
    print(f"  - Impression preview: {result['impression'][:50]}...")
except Exception as e:
    print(f"✗ Model generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test KLE
print("\n[5] Testing KLE uncertainty...")
try:
    samples = [
        "Community-acquired pneumonia",
        "Bacterial pneumonia, likely community-acquired",
        "Pneumonia consistent with bacterial etiology"
    ]
    uncertainty = compute_semantic_uncertainty(samples)
    print(f"✓ KLE computed successfully")
    print(f"  - Uncertainty: {uncertainty:.3f}")
    print(f"  - In range [0,1]: {0.0 <= uncertainty <= 1.0}")
except Exception as e:
    print(f"✗ KLE test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL VERIFICATION TESTS PASSED")
print("=" * 60)
print("\nRefactoring Summary:")
print("  ✓ RadiologistOutput uses text-only format")
print("  ✓ CriticOutput uses new safety-based schema")
print("  ✓ Model generation returns findings+impression text")
print("  ✓ KLE uncertainty computation works")
print("  ✓ Old attributes (hypotheses, internal_signals) removed")
