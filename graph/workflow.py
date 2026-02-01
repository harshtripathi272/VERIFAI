"""
VERIFAI LangGraph Workflow

Defines the complete multi-agent DAG with uncertainty-gated routing.
"""

from langgraph.graph import StateGraph, START, END

from graph.state import VerifaiState, FinalDiagnosis
from graph.router import router_node, route_conditional_edge

# Import agent nodes
from agents.radiologist.agent import radiologist_node
from agents.critic.agent import critic_node
from agents.historian.agent import historian_node
from agents.literature.agent import literature_node
from agents.chief.agent import chief_node


def finalize_node(state: VerifaiState) -> dict:
    """
    Finalize node: creates final diagnosis from accumulated evidence.
    
    Called when uncertainty is low enough for direct diagnosis
    without Chief Orchestrator escalation.
    """
    rad = state.get("radiologist_output")
    hist = state.get("historian_output")
    lit = state.get("literature_output")
    uncertainty = state["current_uncertainty"]
    
    if not rad or not rad.hypotheses:
        return {
            "final_diagnosis": FinalDiagnosis(
                diagnosis=None,
                calibrated_confidence=0.0,
                deferred=True,
                deferral_reason="No diagnostic findings available"
            ),
            "trace": ["FINALIZE: No findings to finalize"]
        }
    
    # Start with radiologist's top hypothesis
    top_dx = rad.hypotheses[0]
    confidence = top_dx.confidence
    
    # Apply adjustments from historian
    if hist:
        confidence += hist.confidence_adjustment
    
    # Boost confidence if literature supports
    if lit and lit.overall_evidence_strength in ["medium", "high"]:
        confidence += 0.05 if lit.overall_evidence_strength == "medium" else 0.10
    
    # Clamp confidence
    confidence = max(0.0, min(0.99, confidence))
    
    final = FinalDiagnosis(
        diagnosis=top_dx.diagnosis,
        calibrated_confidence=confidence,
        deferred=False,
        explanation=f"Based on {len(rad.findings)} visual findings with supporting context.",
        recommended_next_steps=["Confirm with clinical correlation", "Consider follow-up imaging if symptoms persist"]
    )
    
    trace_entry = f"FINALIZE: {final.diagnosis} (confidence={confidence:.2%})"
    
    return {
        "final_diagnosis": final,
        "trace": [trace_entry]
    }


def build_workflow() -> StateGraph:
    """
    Constructs the VERIFAI LangGraph DAG.
    
    Flow:
    START → Radiologist → Critic → Router →┬→ Historian → Critic (loop)
                                           ├→ Literature → Critic (loop)
                                           ├→ Chief → END
                                           └→ Finalize → END
    """
    graph = StateGraph(VerifaiState)
    
    # === Add Nodes ===
    graph.add_node("radiologist", radiologist_node)
    graph.add_node("critic", critic_node)
    graph.add_node("router", router_node)
    graph.add_node("historian", historian_node)
    graph.add_node("literature", literature_node)
    graph.add_node("chief", chief_node)
    graph.add_node("finalize", finalize_node)
    
    # === Define Edges ===
    
    # Entry: START → Radiologist
    graph.add_edge(START, "radiologist")
    
    # Radiologist → Critic (always evaluate uncertainty)
    graph.add_edge("radiologist", "critic")
    
    # Critic → Router (decide next step)
    graph.add_edge("critic", "router")
    
    # Router → Conditional edges based on routing_decision
    graph.add_conditional_edges(
        "router",
        route_conditional_edge,
        {
            "historian": "historian",
            "literature": "literature", 
            "chief": "chief",
            "finalize": "finalize"
        }
    )
    
    # After Historian → back to Critic for re-evaluation
    graph.add_edge("historian", "critic")
    
    # After Literature → back to Critic for re-evaluation
    graph.add_edge("literature", "critic")
    
    # Chief → END (final arbitration complete)
    graph.add_edge("chief", END)
    
    # Finalize → END
    graph.add_edge("finalize", END)
    
    return graph


# Compile graph for execution
workflow = build_workflow()
app = workflow.compile()
