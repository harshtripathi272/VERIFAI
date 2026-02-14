"""
Test script for the Debate Workflow

Tests the new debate mechanism between Critic and Evidence Team (Historian + Literature).
"""

import sys
sys.path.insert(0, '.')

from graph.state import (
    VerifaiState, RadiologistOutput, CriticOutput, HistorianOutput,
    HistorianFact
)
from agents.debate.agent import DebateOrchestrator, debate_node


def create_mock_state():
    """Create a mock state for testing."""
    
    # Mock radiologist output
    # Mock radiologist output
    radiologist_output = RadiologistOutput(
        findings="FINDINGS:\nRight lower lobe consolidation with air bronchograms. Severity: High.",
        impression="IMPRESSION:\nLobar consolidation pattern consistent with bacterial pneumonia. Differential includes atypical pneumonia."
    )
    
    # Mock critic output (moderate overconfidence concern)
    critic_output = CriticOutput(
        is_overconfident=False,
        concern_flags=["Moderate entropy detected", "Single-view limitation"],
        recommended_hedging="Consider adding 'likely' qualifier to impression",
        safety_score=0.65
    )
    
    # Mock historian output (supporting evidence)
    historian_output = HistorianOutput(
        supporting_facts=[
            HistorianFact(
                fact_type="supporting",
                description="Patient has fever (38.5°C) and productive cough for 3 days",
                fhir_resource_id="obs-123",
                fhir_resource_type="Observation"
            ),
            HistorianFact(
                fact_type="supporting",
                description="Elevated WBC count (15,000/μL)",
                fhir_resource_id="obs-456",
                fhir_resource_type="Observation"
            )
        ],
        contradicting_facts=[],
        confidence_adjustment=0.10,
        clinical_summary="65yo male with fever, cough, elevated WBC. History of COPD."
    )
    
    # Mock literature output (string format from optimized agent)
    literature_output = """Found 5 relevant studies:

1. [HIGH] Smith et al. (2024): Radiographic Patterns in Community-Acquired Pneumonia
   Lobar consolidation with air bronchograms demonstrates 94% specificity for bacterial pneumonia...

2. [HIGH] Chen et al. (2024): Pneumonia Outcomes in Diabetic Patients
   Diabetic patients show 2.3x increased mortality in community-acquired pneumonia...

3. [MEDIUM] Brown et al. (2023): Differentiating Atelectasis from Consolidation
   Volume loss and mediastinal shift favor atelectasis over pneumonic consolidation..."""
    
    return {
        "image_path": "test_image.dcm",
        "patient_id": "patient-001",
        "dicom_metadata": {},
        "radiologist_output": radiologist_output,
        "critic_output": critic_output,
        "historian_output": historian_output,
        "literature_output": literature_output,
        "current_uncertainty": 0.42,
        "routing_decision": "",
        "steps_taken": 0,
        "final_diagnosis": None,
        "debate_output": None,
        "trace": []
    }


def test_debate_orchestrator():
    """Test the DebateOrchestrator directly."""
    print("=" * 60)
    print("TEST: DebateOrchestrator")
    print("=" * 60)
    
    state = create_mock_state()
    
    orchestrator = DebateOrchestrator(max_rounds=3, consensus_threshold=0.15)
    
    result = orchestrator.run_debate(
        radiologist_output=state["radiologist_output"],
        critic_output=state["critic_output"],
        historian_output=state["historian_output"],
        literature_output=state["literature_output"]
    )
    
    print(f"\n📊 Debate Results:")
    print(f"   Rounds completed: {len(result.rounds)}")
    print(f"   Consensus reached: {result.final_consensus}")
    print(f"   Consensus diagnosis: {result.consensus_diagnosis}")
    print(f"   Consensus confidence: {result.consensus_confidence:.2%}")
    print(f"   Escalate to Chief: {result.escalate_to_chief}")
    print(f"   Total confidence adjustment: {result.total_confidence_adjustment:+.2%}")
    print(f"   Summary: {result.debate_summary}")
    
    print(f"\n📝 Debate Rounds:")
    for round in result.rounds:
        print(f"\n   Round {round.round_number}:")
        print(f"   - Critic: {round.critic_challenge.argument[:80]}...")
        print(f"   - Historian: {round.historian_response.argument[:80]}...")
        print(f"   - Literature: {round.literature_response.argument[:80]}...")
        print(f"   - Consensus: {round.round_consensus or 'Not reached'}")
        print(f"   - Confidence Δ: {round.confidence_delta:+.2%}")
    
    return result


def test_debate_node():
    """Test the debate_node function (LangGraph integration)."""
    print("\n" + "=" * 60)
    print("TEST: debate_node (LangGraph Integration)")
    print("=" * 60)
    
    state = create_mock_state()
    
    result = debate_node(state)
    
    print(f"\n📊 Node Output:")
    print(f"   Routing decision: {result['routing_decision']}")
    print(f"   New uncertainty: {result['current_uncertainty']:.2%}")
    print(f"   Trace entries: {len(result['trace'])}")
    
    for entry in result['trace']:
        print(f"   - {entry}")
    
    debate_output = result['debate_output']
    print(f"\n   Debate consensus: {debate_output.final_consensus}")
    print(f"   Debate confidence: {debate_output.consensus_confidence:.2%}")
    
    return result


def test_workflow_integration():
    """Test the full workflow with debate."""
    print("\n" + "=" * 60)
    print("TEST: Full Workflow Integration")
    print("=" * 60)
    
    try:
        from graph.workflow import app, build_workflow
        
        print("\n✅ Workflow imported successfully")
        print("   Flow: Radiologist → Critic → Evidence Gathering → Debate → Finalize/Chief")
        
        # Just verify the graph structure
        workflow = build_workflow()
        print(f"   Nodes: {list(workflow.nodes.keys())}")
        
    except Exception as e:
        print(f"\n⚠️  Workflow import failed (expected if langgraph not installed): {e}")


if __name__ == "__main__":
    print("\n🔬 VERIFAI Debate System Test\n")
    
    # Test 1: Orchestrator
    debate_result = test_debate_orchestrator()
    
    # Test 2: Node
    node_result = test_debate_node()
    
    # Test 3: Workflow
    test_workflow_integration()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    
    # Summary
    print("\n📋 Summary:")
    print(f"   - Debate reached consensus: {debate_result.final_consensus}")
    print(f"   - Final confidence: {debate_result.consensus_confidence:.2%}")
    print(f"   - Routing decision: {node_result['routing_decision']}")
