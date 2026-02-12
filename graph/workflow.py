"""
VERIFAI LangGraph Workflow

Defines the complete multi-agent DAG with debate-based consensus.

NEW FLOW:
START → Radiologist → Critic → [Historian + Literature (parallel)] → Debate → Chief/Finalize → END
"""

from langgraph.graph import StateGraph, START, END
from concurrent.futures import ThreadPoolExecutor, wait

from graph.state import VerifaiState, FinalDiagnosis
from app.config import settings

# Import agent nodes
from agents.radiologist.agent import radiologist_node
from agents.critic.agent import critic_node
from agents.historian.agent import historian_node
from agents.literature.agent import literature_agent_node as literature_node  # Fix: correct function name
from agents.debate.agent import debate_node
from agents.chief.agent import chief_node


def evidence_gathering_node(state: VerifaiState) -> dict:
    """
    Parallel execution of Historian and Literature agents.
    
    Both agents ALWAYS run to gather complete evidence before debate.
    This is faster than sequential execution and provides richer context.
    """
    results = {}
    trace_entries = []
    
    # Check if parallel execution is enabled
    use_parallel = getattr(settings, 'USE_PARALLEL_AGENTS', True)
    
    if use_parallel:
        # Run both agents in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            historian_future = executor.submit(historian_node, state)
            literature_future = executor.submit(literature_node, state)
            
            # Wait for both to complete
            try:
                historian_result = historian_future.result(timeout=30)
                results["historian_output"] = historian_result.get("historian_output")
                trace_entries.extend(historian_result.get("trace", []))
            except Exception as e:
                trace_entries.append(f"EVIDENCE_GATHER: Historian failed - {str(e)[:50]}")
            
            try:
                literature_result = literature_future.result(timeout=30)
                results["literature_output"] = literature_result.get("literature_output")
                trace_entries.extend(literature_result.get("trace", []))
            except Exception as e:
                trace_entries.append(f"EVIDENCE_GATHER: Literature failed - {str(e)[:50]}")
    else:
        # Sequential execution (fallback)
        try:
            historian_result = historian_node(state)
            results["historian_output"] = historian_result.get("historian_output")
            trace_entries.extend(historian_result.get("trace", []))
        except Exception as e:
            trace_entries.append(f"EVIDENCE_GATHER: Historian failed - {str(e)[:50]}")
        
        try:
            literature_result = literature_node(state)
            results["literature_output"] = literature_result.get("literature_output")
            trace_entries.extend(literature_result.get("trace", []))
        except Exception as e:
            trace_entries.append(f"EVIDENCE_GATHER: Literature failed - {str(e)[:50]}")
    
    trace_entries.insert(0, "EVIDENCE_GATHER: Historian + Literature executed" + 
                        (" (parallel)" if use_parallel else " (sequential)"))
    
    results["trace"] = trace_entries
    return results


def finalize_node(state: VerifaiState) -> dict:
    """
    Finalize node: creates final diagnosis from debate consensus.
    
    Uses debate output for calibrated confidence.
    """
    rad = state.get("radiologist_output")
    debate = state.get("debate_output")
    hist = state.get("historian_output")
    lit = state.get("literature_output")
    
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
    
    # Use debate consensus if available
    if debate and debate.final_consensus:
        final = FinalDiagnosis(
            diagnosis=debate.consensus_diagnosis,
            calibrated_confidence=debate.consensus_confidence,
            deferred=False,
            explanation=f"Consensus reached through {len(debate.rounds)}-round debate. {debate.debate_summary}",
            recommended_next_steps=["Confirm with clinical correlation", "Consider follow-up imaging if symptoms persist"]
        )
        trace_entry = f"FINALIZE: {final.diagnosis} (confidence={final.calibrated_confidence:.2%}) via debate consensus"
    else:
        # Fallback to original logic
        top_dx = rad.hypotheses[0]
        confidence = top_dx.confidence
        
        if hist:
            confidence += hist.confidence_adjustment
        
        if lit and hasattr(lit, 'overall_evidence_strength') and lit.overall_evidence_strength in ["medium", "high"]:
            confidence += 0.05 if lit.overall_evidence_strength == "medium" else 0.10
        
        # Apply debate adjustment if available
        if debate:
            confidence += debate.total_confidence_adjustment
        
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


def route_after_debate(state: VerifaiState) -> str:
    """
    Route based on debate outcome.
    
    - Consensus reached → finalize
    - No consensus → chief
    """
    debate = state.get("debate_output")
    
    if debate and debate.final_consensus:
        return "finalize"
    elif debate and debate.escalate_to_chief:
        return "chief"
    else:
        # Default to finalize if no debate output
        return "finalize"


def build_workflow() -> StateGraph:
    """
    Constructs the VERIFAI LangGraph DAG with debate mechanism.
    
    NEW Flow:
    START → Radiologist → Critic → Evidence Gathering (Hist + Lit parallel) → Debate →┬→ Finalize → END
                                                                                       └→ Chief → END
    """
    graph = StateGraph(VerifaiState)
    
    # === Add Nodes ===
    graph.add_node("radiologist", radiologist_node)
    graph.add_node("critic", critic_node)
    graph.add_node("evidence_gathering", evidence_gathering_node)  # NEW: Parallel Hist + Lit
    graph.add_node("debate", debate_node)  # NEW: Debate mechanism
    graph.add_node("chief", chief_node)
    graph.add_node("finalize", finalize_node)
    
    # === Define Edges ===
    
    # Entry: START → Radiologist
    graph.add_edge(START, "radiologist")
    
    # Radiologist → Critic (always evaluate first)
    graph.add_edge("radiologist", "critic")
    
    # Critic → Evidence Gathering (ALWAYS run Historian + Literature)
    graph.add_edge("critic", "evidence_gathering")
    
    # Evidence Gathering → Debate
    graph.add_edge("evidence_gathering", "debate")
    
    # Debate → Conditional: Finalize or Chief
    graph.add_conditional_edges(
        "debate",
        route_after_debate,
        {
            "finalize": "finalize",
            "chief": "chief"
        }
    )
    
    # Chief → END (final arbitration complete)
    graph.add_edge("chief", END)
    
    # Finalize → END
    graph.add_edge("finalize", END)
    
    return graph


# === LEGACY WORKFLOW (for backward compatibility) ===

# def build_legacy_workflow() -> StateGraph:
#     """
#     Original workflow with uncertainty-gated routing.
#     Use this if you prefer the old behavior.
#     """
#     from graph.router import router_node, route_conditional_edge
    
#     graph = StateGraph(VerifaiState)
    
#     graph.add_node("radiologist", radiologist_node)
#     graph.add_node("critic", critic_node)
#     graph.add_node("router", router_node)
#     graph.add_node("historian", historian_node)
#     graph.add_node("literature", literature_node)
#     graph.add_node("chief", chief_node)
#     graph.add_node("finalize", finalize_node)
    
#     graph.add_edge(START, "radiologist")
#     graph.add_edge("radiologist", "critic")
#     graph.add_edge("critic", "router")
    
#     graph.add_conditional_edges(
#         "router",
#         route_conditional_edge,
#         {
#             "historian": "historian",
#             "literature": "literature", 
#             "chief": "chief",
#             "finalize": "finalize"
#         }
#     )
    
#     graph.add_edge("historian", "critic")
#     graph.add_edge("literature", "critic")
#     graph.add_edge("chief", END)
#     graph.add_edge("finalize", END)
    
#     return graph


# === Compile Workflows ===

# Use debate workflow by default
workflow = build_workflow()
app = workflow.compile()

# # Legacy workflow available if needed
# legacy_workflow = build_legacy_workflow()
# legacy_app = legacy_workflow.compile()
