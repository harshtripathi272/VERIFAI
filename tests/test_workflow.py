"""Quick test of the VERIFAI LangGraph workflow."""

from graph.workflow import app as verifai_graph

# Initialize state
initial_state = {
    "image_path": "./img1.jpg",
    "view": "AP",
    "patient_id": None,
    "radiologist_output": None,
    "critic_output": None,
    "historian_output": None,
    "literature_output": None,
    "current_uncertainty": 1.0,
    "routing_decision": "",
    "steps_taken": 0,
    "final_diagnosis": None,
    "trace": ["[TEST] Initialization"],
}

# Run graph
result = verifai_graph.invoke(initial_state)

# Print results
print("=" * 60)
print("VERIFAI WORKFLOW TEST")
print("=" * 60)

print("\n--- TRACE ---")
for entry in result["trace"]:
    print(entry)

print("\n--- RESULT ---")
final_dx = result.get("final_diagnosis")
if final_dx:
    print(f"Diagnosis: {final_dx.diagnosis}")
    print(f"Confidence: {final_dx.calibrated_confidence:.0%}")
    print(f"Deferred: {final_dx.deferred}")
    if final_dx.deferral_reason:
        print(f"Reason: {final_dx.deferral_reason}")
else:
    print("No final diagnosis")

print(f"\nFinal Uncertainty: {result['current_uncertainty']:.2%}")
print(f"Steps Taken: {result['steps_taken']}")
print(f"\nHistorian invoked: {result.get('historian_output') is not None}")
print(f"Literature invoked: {result.get('literature_output') is not None}")
