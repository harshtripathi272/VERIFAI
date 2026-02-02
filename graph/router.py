"""
VERIFAI Uncertainty-Gated Router

Determines next agent based on uncertainty thresholds.
"""

from graph.state import VerifaiState
from app.config import settings


def compute_routing_decision(state: VerifaiState) -> str:
    """
    Core routing logic based on uncertainty thresholds.
    
    Routing Rules:
    - U < 0.30: Direct to finalize (low uncertainty)
    - 0.30 <= U < 0.40: Invoke Historian (if not done)
    - 0.40 <= U < 0.50: Invoke Literature (if not done)  
    - U >= 0.50: Escalate to Chief Orchestrator
    
    Returns:
        str: Next node name ("historian", "literature", "chief", "finalize")
    """
    uncertainty = state["current_uncertainty"]
    steps = state.get("steps_taken", 0)
    
    # Safety: prevent infinite loops
    if steps >= settings.MAX_ROUTING_STEPS:
        return "chief"
    
    # High uncertainty -> Chief
    if uncertainty >= settings.THRESHOLD_CHIEF:
        return "chief"
    
    # Medium-high uncertainty -> Literature (if not already invoked)
    if uncertainty >= settings.THRESHOLD_LITERATURE:
        if state.get("literature_output") is None:
            return "literature"
        # Already have literature, but still uncertain -> Chief
        return "chief"
    
    # Medium uncertainty -> Historian (if not already invoked)
    if uncertainty >= settings.THRESHOLD_HISTORIAN:
        if state.get("historian_output") is None:
            return "historian"
        # Already have historian context -> finalize
        return "finalize"
    
    # Low uncertainty -> direct finalize
    return "finalize"


def router_node(state: VerifaiState) -> dict:
    """
    Router node for LangGraph.
    
    Computes routing decision and updates state.
    """
    decision = compute_routing_decision(state)
    steps = state.get("steps_taken", 0)
    uncertainty = state["current_uncertainty"]
    
    trace_entry = (
        f"ROUTER: U={uncertainty:.2%}, Steps={steps+1} → {decision.upper()}"
    )
    
    return {
        "routing_decision": decision,
        "steps_taken": steps + 1,
        "trace": [trace_entry]
    }


def route_conditional_edge(state: VerifaiState) -> str:
    """
    Conditional edge function for LangGraph.
    
    Returns the routing decision for graph traversal.
    """
    return state.get("routing_decision", "finalize")
