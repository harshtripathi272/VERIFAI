"""
Test script for Evidence Gathering Node.
Tests parallel execution of Historian + Literature agents.
This is a key node in the workflow that gathers clinical context before debate.
"""

import sys
import os
# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.workflow import evidence_gathering_node
from graph.state import VerifaiState, RadiologistOutput, CheXbertOutput
import json
import time


def test_evidence_gathering():
    print("=" * 70)
    print("EVIDENCE GATHERING NODE TEST - PARALLEL HISTORIAN + LITERATURE")
    print("=" * 70)
    
    # Create sample radiologist output (findings + impression)
    radiologist_output = RadiologistOutput(
        findings="""
        Bilateral interstitial infiltrates with ground-glass opacities.
        Patchy consolidation in the right middle lobe. Cardiac silhouette 
        is mildly enlarged. Small bilateral pleural effusions. No pneumothorax.
        Calcified granulomas in both lung apices, likely old granulomatous disease.
        """,
        impression="""
        1. Bilateral ground-glass opacities and consolidation, suspicious for 
           multifocal pneumonia vs. atypical infection vs. early ARDS.
        2. Mild cardiomegaly with small bilateral pleural effusions.
        3. Recommend correlation with clinical findings and consider CT chest
           if diagnosis remains uncertain.
        """
    )
    
    # Create sample CheXbert output
    chexbert_output = CheXbertOutput(
        labels={
            "Consolidation": "present",
            "Edema": "uncertain",
            "Pleural Effusion": "present",
            "Cardiomegaly": "present",
            "Pneumonia": "uncertain",
            "Atelectasis": "absent"
        },
        confidence=0.78
    )
    
    # Create state with all required inputs
    state = VerifaiState(
        image_path="/path/to/chest_xray_bilateral.jpg",
        patient_id="test-patient-ev-001",
        radiologist_output=radiologist_output,
        chexbert_output=chexbert_output,
        _session_id="test-evidence-session-001"
    )
    
    print("\nInput State:")
    print(f"  Patient ID: {state.patient_id}")
    print(f"  Image Path: {state.image_path}")
    print(f"  Session ID: {state._session_id}")
    
    print(f"\nRadiologist Findings ({len(radiologist_output.findings)} chars):")
    print(f"  Preview: {radiologist_output.findings[:150]}...")
    
    print(f"\nRadiologist Impression ({len(radiologist_output.impression)} chars):")
    print(f"  Preview: {radiologist_output.impression[:150]}...")
    
    print(f"\nCheXbert Labels ({len(chexbert_output.labels)} pathologies):")
    for label, status in chexbert_output.labels.items():
        print(f"  - {label}: {status}")
    
    print("\n" + "=" * 70)
    print("PIPELINE ARCHITECTURE")
    print("=" * 70)
    print("""
    Evidence Gathering Node executes BOTH agents in PARALLEL:
    
    ┌─────────────────────────────────────────┐
    │   Evidence Gathering Node               │
    │                                         │
    │   ┌──────────────┐   ┌──────────────┐  │
    │   │  Historian   │   │  Literature  │  │
    │   │   Agent      │   │    Agent     │  │
    │   │              │   │              │  │
    │   │ • FHIR Data  │   │ • PubMed     │  │
    │   │ • Clinical   │   │ • PMC        │  │
    │   │   History    │   │ • SemScholar │  │
    │   │ • MedGemma   │   │ • MedGemma   │  │
    │   │   Reasoning  │   │   RAG        │  │
    │   └──────────────┘   └──────────────┘  │
    │          ↓                   ↓          │
    │   ┌─────────────────────────────────┐  │
    │   │    Merged Output State         │  │
    │   └─────────────────────────────────┘  │
    └─────────────────────────────────────────┘
    
    Both agents share the same MedGemma model instance (singleton).
    ThreadPoolExecutor ensures parallel execution with timeout handling.
    """)
    
    print("\n" + "-" * 70)
    print("Step 1: Running Evidence Gathering (Parallel Execution)...")
    print("         This will execute both Historian and Literature")
    print("         Timeout: 30s per agent")
    print("-" * 70 + "\n")
    
    start_time = time.time()
    
    try:
        # Call evidence gathering node (runs historian + literature in parallel)
        result = evidence_gathering_node(state)
        
        elapsed_time = time.time() - start_time
        
        historian_output = result.get("historian_output")
        literature_output = result.get("literature_output")
        trace = result.get("trace", [])
        
        print("\n" + "=" * 70)
        print("EVIDENCE GATHERING RESULTS")
        print("=" * 70)
        print(f"\nExecution Time: {elapsed_time:.2f}s")
        
        # Historian output
        print("\n" + "-" * 70)
        print("HISTORIAN OUTPUT")
        print("-" * 70)
        if historian_output:
            if isinstance(historian_output, dict):
                print("\nHistorian Result (dict):")
                print(json.dumps(historian_output, indent=2, default=str))
            else:
                print(f"\nHistorian Result: {historian_output}")
            print(f"\nHistorian output size: {len(str(historian_output))} chars")
        else:
            print("\n⚠ No historian output (may have failed or timed out)")
        
        # Literature output
        print("\n" + "-" * 70)
        print("LITERATURE OUTPUT")
        print("-" * 70)
        if literature_output:
            if isinstance(literature_output, str):
                print(f"\nLiterature Summary ({len(literature_output)} chars):")
                print(literature_output)
            else:
                print("\nLiterature Result (structured):")
                print(json.dumps(literature_output, indent=2, default=str))
            print(f"\nLiterature output size: {len(str(literature_output))} chars")
        else:
            print("\n⚠ No literature output (may have failed or timed out)")
        
        # Trace entries
        print("\n" + "-" * 70)
        print("TRACE ENTRIES")
        print("-" * 70)
        for entry in trace:
            print(f"  - {entry}")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        success_count = sum([
            1 if historian_output else 0,
            1 if literature_output else 0
        ])
        print(f"\nAgents Successful: {success_count}/2")
        print(f"Historian: {'✓' if historian_output else '✗'}")
        print(f"Literature: {'✓' if literature_output else '✗'}")
        print(f"Total Execution Time: {elapsed_time:.2f}s")
        
        if success_count == 2:
            print("\n✓ Test completed successfully! Both agents ran in parallel.")
        elif success_count == 1:
            print("\n⚠ Partial success: One agent failed or timed out.")
        else:
            print("\n✗ Both agents failed!")
            return False
        
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
    success = test_evidence_gathering()
    exit(0 if success else 1)
