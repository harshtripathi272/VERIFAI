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
    image_path = ["../dataset/med/official_data_iccv_final/files/p10/p10010440/s56908581/0cadb1ed-80bd62aa-8d4563e1-2289ab1f-5be0b197.jpg","../dataset/med/official_data_iccv_final/files/p10/p10010440/s56908581/e0ceccb1-efe6919f-2b3c8cd2-c087f0b0-3d3adc66.jpg"]
    view = ["AP","LATERAL"]
    
    
    
    print("\n" + "-" * 70)
    print("Running Radiologist Agent...")
    print("-" * 70 + "\n")
    
    try:
        # Call the radiologist agent
        result = generate_findings(image_paths=image_path, views=view)
        
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
