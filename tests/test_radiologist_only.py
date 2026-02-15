"""
Simple test script for Radiologist agent only.
Tests the vision-language model inference with the fixed hook mechanism.
"""

from agents.radiologist.model import generate_findings
import json

def test_radiologist():
    print("=" * 70)
    print("RADIOLOGIST AGENT TEST")
    print("=" * 70)
    
    # Test input
    image_path = "./img1.jpg"
    view = "AP"
    
    print(f"\nInput:")
    print(f"  Image: {image_path}")
    print(f"  View: {view}")
    
    print("\n" + "-" * 70)
    print("Running Radiologist Agent...")
    print("-" * 70 + "\n")
    
    try:
        # Call the radiologist agent
        result = generate_findings(image_path, view=view)
        
        print("\n" + "=" * 70)
        print("RADIOLOGIST OUTPUT")
        print("=" * 70)
        print(f"\nFindings:\n{result.get('findings', 'N/A')}")
        print(f"\nImpression:\n{result.get('impression', 'N/A')}")
        
        # Pretty print full result
        print("\n" + "=" * 70)
        print("FULL RESULT (JSON)")
        print("=" * 70)
        print(json.dumps(result, indent=2))
        
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR OCCURRED")
        print("=" * 70)
        print(f"\nException type: {type(e).__name__}")
        print(f"Exception message: {e}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        print("\n✗ Test failed!")
        return False
    
    return True


if __name__ == "__main__":
    success = test_radiologist()
    exit(0 if success else 1)
