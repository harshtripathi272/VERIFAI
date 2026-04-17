"""
Test script for Literature Agent only.
Tests the RAG-based literature retrieval with MedGemma using shared model loader.
"""

import sys
import os
# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.literature.agent import literature_agent_node
from graph.state import VerifaiState, RadiologistOutput, CheXbertOutput
import json


def test_literature_agent():
    print("=" * 70)
    print("LITERATURE AGENT TEST - RAG RETRIEVAL WITH MEDGEMMA")
    print("=" * 70)
    
    # Create sample radiologist output (what literature agent would receive)
    radiologist_output = RadiologistOutput(
        findings="""
        Patchy airspace opacities noted in the right lower lobe with associated
        air bronchograms. The cardiac silhouette is within normal limits.
        No pleural effusion or pneumothorax. The bony thorax is intact.
        Mild prominence of the pulmonary vasculature.
        """,
        impression="""
        Right lower lobe consolidation, most consistent with pneumonia.
        Consider clinical correlation with patient symptoms and laboratory findings.
        """
    )
    
    # Create sample CheXbert output (structured pathology labels)
    chexbert_output = CheXbertOutput(
        labels={
            "Consolidation": "present",
            "Pneumonia": "uncertain",
            "Cardiomegaly": "absent",
            "Edema": "absent",
            "Pleural Effusion": "absent"
        },
        confidence=0.85
    )
    
    # Create sample historian output (clinical context)
    # Mock simple histogram output structure
    historian_output_dict = {
        "clinical_summary": "Patient presents with fever, cough, and shortness of breath for 3 days. History of hypertension.",
        "supporting_facts": [
            {"description": "Elevated WBC count (15,000/uL)", "resource_type": "Observation"},
            {"description": "Fever 38.5°C", "resource_type": "Observation"}
        ],
        "confidence_adjustment": 0.1
    }
    
    # Create state with all required inputs
    state = VerifaiState(
        image_path="/path/to/chest_xray.jpg",
        patient_id="test-patient-lit-001",
        radiologist_output=radiologist_output,
        chexbert_output=chexbert_output,
        historian_output=historian_output_dict  # Add historian context
    )
    
    print("\nInput State:")
    print(f"  Patient ID: {state.get('patient_id')}")
    print(f"  Image Path: {state.get('image_path')}")
    print(f"\nRadiologist Findings Preview: {radiologist_output.findings[:100]}...")
    print(f"\nRadiologist Impression: {radiologist_output.impression[:100]}...")
    print(f"\nCheXbert Labels: {chexbert_output.labels}")
    print(f"\nHistorian Summary: {historian_output_dict['clinical_summary'][:100]}...")
    
    print("\n" + "-" * 70)
    print("Step 1: Calling Literature Agent...")
    print("         This will load shared MedGemma model if not already loaded")
    print("         Expected to search PubMed/PMC/Semantic Scholar")
    print("-" * 70 + "\n")
    
    try:
        # Call literature agent node
        result = literature_agent_node(state)
        
        literature_output = result.get("literature_output")
        trace = result.get("trace", [])
        
        print("\n" + "=" * 70)
        print("LITERATURE AGENT OUTPUT")
        print("=" * 70)
        
        if isinstance(literature_output, str):
            # String summary output
            print(f"\nLiterature Summary ({len(literature_output)} chars):")
            print(literature_output)
        else:
            # Structured output
            print("\nLiterature Output (structured):")
            print(json.dumps(literature_output, indent=2, default=str))
        
        print("\n" + "-" * 70)
        print("TRACE ENTRIES")
        print("-" * 70)
        for entry in trace:
            print(f"  - {entry}")
        
        print("\n✓ Test completed successfully!")
        print(f"\nLiterature output size: {len(str(literature_output))} chars")
        
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
    success = test_literature_agent()
    exit(0 if success else 1)
