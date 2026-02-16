"""
VERIFAI Doctor Feedback Example

Complete end-to-end example showing:
1. Normal diagnosis workflow
2. Doctor reviews and rejects
3. System reprocesses with feedback
4. Results are compared and linked

This is a demonstration script showing how to integrate
the doctor feedback loop into your application.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def example_normal_workflow():
    """Example 1: Run normal diagnostic workflow."""
    print("\n" + "="*70)
    print("STEP 1: NORMAL DIAGNOSTIC WORKFLOW")
    print("="*70 + "\n")
    
    from graph.workflow import app
    from graph.state import VerifaiState
    
    # Create initial state
    state = VerifaiState(
        image_path="example_xray.jpg",
        patient_id="PATIENT_001",
        _session_id="example-session-001"
    )
    
    print("Running workflow...")
    print("  - Radiologist analyzing image...")
    print("  - CheXbert labeling pathologies...")
    print("  - Gathering evidence (FHIR + Literature)...")
    print("  - Critic evaluating...")
    print("  - Debate reconciling...")
    
    # In real usage, you would call:
    # result = app.invoke(state)
    
    # For this example, we'll simulate the result
    print("\n✅ Workflow completed!")
    print("\nResults:")
    print("  Diagnosis: Pneumonia, right lower lobe")
    print("  Confidence: 78%")
    print("  Session ID: example-session-001")
    
    return "example-session-001", "Pneumonia, right lower lobe", 0.78


def example_doctor_review(session_id, original_diagnosis, original_confidence):
    """Example 2: Doctor reviews and provides feedback."""
    print("\n" + "="*70)
    print("STEP 2: DOCTOR REVIEWS DIAGNOSIS")
    print("="*70 + "\n")
    
    print(f"Doctor reviewing session: {session_id}")
    print(f"  Original diagnosis: {original_diagnosis}")
    print(f"  Original confidence: {original_confidence:.1%}")
    print("\n❌ Doctor disagrees with diagnosis!\n")
    
    from agents.feedback import capture_doctor_feedback
    
    # Doctor's assessment
    doctor_notes = """
    This diagnosis is incorrect. The image shows bilateral pleural effusion,
    not pneumonia. The opacity in the right lower lobe is due to fluid
    accumulation, not consolidation. The patient's history of CHF strongly
    supports a diagnosis of cardiogenic pleural effusion rather than
    infectious pneumonia. Additionally, there are bilateral costophrenic
    angle blunting consistent with effusion.
    """
    
    correct_diagnosis = "Bilateral pleural effusion, likely cardiogenic"
    
    reasons = [
        "missed_finding",  # Missed the pleural effusion
        "incorrect_primary_diagnosis"  # Called it pneumonia instead
    ]
    
    print("Doctor providing detailed feedback:")
    print(f"  Notes: {doctor_notes.strip()[:100]}...")
    print(f"  Correct diagnosis: {correct_diagnosis}")
    print(f"  Issues: {', '.join(reasons)}")
    
    # Capture feedback
    # In real usage:
    # feedback_id = capture_doctor_feedback(
    #     session_id=session_id,
    #     feedback_type="rejection",
    #     doctor_notes=doctor_notes.strip(),
    #     correct_diagnosis=correct_diagnosis,
    #     rejection_reasons=reasons,
    #     doctor_id="dr_smith"
    # )
    
    # Simulated
    feedback_id = 1001
    
    print(f"\n✅ Feedback captured!")
    print(f"  Feedback ID: {feedback_id}")
    print(f"  Type: rejection")
    print(f"  Stored with full workflow context")
    
    return feedback_id


def example_reprocess_with_feedback(feedback_id, original_session_id):
    """Example 3: Reprocess workflow with doctor's feedback."""
    print("\n" + "="*70)
    print("STEP 3: REPROCESS WITH DOCTOR FEEDBACK")
    print("="*70 + "\n")
    
    from agents.feedback import (
        prepare_feedback_for_reprocessing,
        create_feedback_enhanced_state,
        link_feedback_reprocessing_result
    )
    from graph.workflow import app
    
    print(f"Loading feedback: {feedback_id}")
    
    # In real usage:
    # feedback_input = prepare_feedback_for_reprocessing(feedback_id)
    
    print("✅ Feedback loaded with preserved context:")
    print("  - Radiologist findings (preserved)")
    print("  - CheXbert labels (preserved)")
    print("  - FHIR clinical history (preserved)")
    print("  - Literature evidence (preserved)")
    print("  + Doctor's feedback notes")
    
    # Create enhanced state
    print("\nCreating feedback-enhanced state...")
    
    # In real usage:
    # new_state = create_feedback_enhanced_state(
    #     feedback_input=feedback_input,
    #     image_path="example_xray.jpg",
    #     patient_id="PATIENT_001"
    # )
    
    print("✅ State created:")
    print("  - is_feedback_iteration = TRUE")
    print("  - doctor_feedback injected")
    print("  - Will skip to critic (no re-analysis)")
    
    print("\nReprocessing workflow...")
    print("  - [SKIP] Radiologist (using cached)")
    print("  - [SKIP] CheXbert (using cached)")
    print("  - [SKIP] Evidence gathering (using cached)")
    print("  - Critic evaluating WITH doctor feedback...")
    print("  - Debate reconciling with new context...")
    
    # In real usage:
    # result = app.invoke(new_state)
    
    # Simulated result
    new_session_id = "example-session-002"
    new_diagnosis = "Bilateral pleural effusion, likely cardiogenic"
    new_confidence = 0.85
    
    print("\n✅ Reprocessing completed!")
    print(f"  New session ID: {new_session_id}")
    print(f"  New diagnosis: {new_diagnosis}")
    print(f"  New confidence: {new_confidence:.1%}")
    
    # Link results
    print("\nLinking reprocessing result to feedback...")
    
    # In real usage:
    # link_feedback_reprocessing_result(
    #     feedback_id=feedback_id,
    #     new_session_id=new_session_id,
    #     final_diagnosis=new_diagnosis,
    #     final_confidence=new_confidence
    # )
    
    print("✅ Results linked!")
    
    return new_session_id, new_diagnosis, new_confidence


def example_compare_results(original_dx, original_conf, new_dx, new_conf):
    """Example 4: Compare before and after."""
    print("\n" + "="*70)
    print("STEP 4: COMPARE RESULTS")
    print("="*70 + "\n")
    
    print("BEFORE (Original):")
    print(f"  Diagnosis: {original_dx}")
    print(f"  Confidence: {original_conf:.1%}")
    print(f"  Status: ❌ Rejected by doctor")
    
    print("\nAFTER (Reprocessed with feedback):")
    print(f"  Diagnosis: {new_dx}")
    print(f"  Confidence: {new_conf:.1%}")
    print(f"  Status: ✅ Aligned with doctor's assessment")
    
    improvement = new_conf - original_conf
    print(f"\nConfidence improvement: {improvement:+.1%}")
    
    print("\n📊 Impact:")
    print(f"  • Skipped image re-analysis (60% faster)")
    print(f"  • Preserved evidence context")
    print(f"  • Incorporated expert knowledge")
    print(f"  • Created audit trail")


def main():
    """Run complete example."""
    print("\n" + "="*70)
    print("VERIFAI DOCTOR FEEDBACK LOOP - COMPLETE EXAMPLE")
    print("="*70)
    
    print("\nThis example demonstrates:")
    print("  1. Normal diagnostic workflow")
    print("  2. Doctor review and rejection")
    print("  3. Reprocessing with feedback")
    print("  4. Comparison of results")
    
    print("\nNote: This is a simulation. In production, you would:")
    print("  - Actually run the workflow with real images")
    print("  - Store data in Supabase database")
    print("  - Integrate with your UI for doctor feedback")
    
    input("\nPress Enter to start...")
    
    # Step 1: Normal workflow
    session_id, original_dx, original_conf = example_normal_workflow()
    input("\nPress Enter to continue to doctor review...")
    
    # Step 2: Doctor review
    feedback_id = example_doctor_review(session_id, original_dx, original_conf)
    input("\nPress Enter to continue to reprocessing...")
    
    # Step 3: Reprocess with feedback
    new_session_id, new_dx, new_conf = example_reprocess_with_feedback(
        feedback_id, session_id
    )
    input("\nPress Enter to see comparison...")
    
    # Step 4: Compare results
    example_compare_results(original_dx, original_conf, new_dx, new_conf)
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETED")
    print("="*70)
    
    print("\n📚 To use in production:")
    print("  1. Set up Supabase (see DOCTOR_FEEDBACK_AND_CLOUD_DB_GUIDE.md)")
    print("  2. Configure .env with credentials")
    print("  3. Run: python setup_helper.py check-db")
    print("  4. Integrate feedback UI in your application")
    print("  5. Use the actual functions (not simulated)")
    
    print("\n💡 Quick Reference:")
    print("  - QUICK_REFERENCE.md for code snippets")
    print("  - FEEDBACK_FLOW_DIAGRAM.md for visual guide")
    print("  - IMPLEMENTATION_SUMMARY.md for overview")
    
    print("\n✅ Ready to enhance your radiology AI with expert feedback!\n")


if __name__ == "__main__":
    main()
