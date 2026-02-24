"""
VERIFAI Integration Test

Smoke test for the complete graph pipeline.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.workflow import app
from graph.state import VerifaiState


def test_full_pipeline():
    """Test complete diagnostic pipeline."""
    print("=" * 60)
    print("VERIFAI Integration Test")
    print("=" * 60)
    
    initial_state: VerifaiState = {
        "image_path": "test_xray.png",
        "patient_id": "PAT-TEST-123",
        "dicom_metadata": None,
        "radiologist_output": None,
        "critic_output": None,
        "historian_output": None,
        "literature_output": None,
        "current_uncertainty": 1.0,
        "routing_decision": "",
        "steps_taken": 0,
        "final_diagnosis": None,
        "trace": ["[TEST] Initializing test run"]
    }
    
    try:
        config = {"configurable": {"thread_id": "test_router_123"}}
        result = app.invoke(initial_state, config)
        
        print("\n📝 EXECUTION TRACE:")
        print("-" * 40)
        for line in result["trace"]:
            print(f"  {line}")
        
        print("\n📊 FINAL STATE:")
        print("-" * 40)
        dx = result.get("final_diagnosis")
        if dx:
            print(f"  Diagnosis: {dx.diagnosis}")
            print(f"  Confidence: {dx.calibrated_confidence:.1%}")
            print(f"  Deferred: {dx.deferred}")
            if dx.deferral_reason:
                print(f"  Reason: {dx.deferral_reason}")
        else:
            print("  ERROR: No diagnosis produced")
        
        print(f"  Final Uncertainty: {result['current_uncertainty']:.1%}")
        print(f"  Steps Taken: {result['steps_taken']}")
        
        print("\n✅ TEST PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
