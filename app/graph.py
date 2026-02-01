"""
VERIFAI LangGraph Workflow Definition

Defines the complete graph structure with uncertainty-gated routing.
"""

from langgraph.graph import StateGraph, START, END

from app.state import VerifaiState
from app.agents import (
    radiologist_node,
    critic_node,
    historian_node,
    literature_node,
    chief_node,
    router_node,
    finalize_node,
)


def route_after_critic(state: VerifaiState) -> str:
    """
    Conditional edge: routes based on uncertainty after critic evaluation.

    This is the uncertainty-gated routing logic:
    - U < 0.30: Direct to diagnosis
    - 0.30 ≤ U < 0.40: Invoke Historian (if not already done)
    - 0.40 ≤ U < 0.50: Invoke Literature (if not already done)
    - U ≥ 0.50: Escalate to Chief
    """
    decision = state.get("routing_decision", "diagnose")

    if decision == "historian":
        return "historian"
    elif decision == "literature":
        return "literature"
    elif decision == "chief":
        return "chief"
    else:
        return "finalize"


def build_graph() -> StateGraph:
    """
    Constructs the VERIFAI LangGraph.

    Flow:
    ┌─────────┐     ┌────────────┐     ┌────────┐     ┌────────────────┐
    │  START  │────►│ Radiologist│────►│ Critic │────►│     Router     │
    └─────────┘     └────────────┘     └────────┘     └───────┬────────┘
                                                              │
                    ┌─────────────────────────────────────────┼─────────────────────────┐
                    │                    │                    │                         │
                    ▼                    ▼                    ▼                         ▼
             ┌────────────┐      ┌────────────┐       ┌─────────────┐           ┌───────────┐
             │  Historian │      │ Literature │       │    Chief    │           │  Finalize │
             └─────┬──────┘      └──────┬─────┘       └──────┬──────┘           └─────┬─────┘
                   │                    │                    │                        │
                   ▼                    ▼                    ▼                        ▼
             ┌────────────┐      ┌────────────┐          ┌───────┐                ┌───────┐
             │   Critic   │      │   Critic   │          │  END  │                │  END  │
             │  (Re-eval) │      │  (Re-eval) │          └───────┘                └───────┘
             └─────┬──────┘      └──────┬─────┘
                   │                    │
                   └────────────────────┴────────► Router (loop back)
    """
    # Initialize the graph with our state type
    graph = StateGraph(VerifaiState)

    # --- Add Nodes ---
    graph.add_node("radiologist", radiologist_node)
    graph.add_node("critic", critic_node)
    graph.add_node("router", router_node)
    graph.add_node("historian", historian_node)
    graph.add_node("literature", literature_node)
    graph.add_node("chief", chief_node)
    graph.add_node("finalize", finalize_node)

    # --- Define Edges ---

    # Entry: START -> Radiologist
    graph.add_edge(START, "radiologist")

    # Radiologist -> Critic
    graph.add_edge("radiologist", "critic")

    # Critic -> Router (decides next step)
    graph.add_edge("critic", "router")

    # Router -> Conditional edges based on routing_decision
    graph.add_conditional_edges(
        "router",
        route_after_critic,
        {
            "historian": "historian",
            "literature": "literature",
            "chief": "chief",
            "finalize": "finalize",
        },
    )

    # After Historian -> back to Critic for re-evaluation
    graph.add_edge("historian", "critic")

    # After Literature -> back to Critic for re-evaluation
    graph.add_edge("literature", "critic")

    # Chief -> END (final arbitration complete)
    graph.add_edge("chief", END)

    # Finalize -> END (direct diagnosis complete)
    graph.add_edge("finalize", END)

    return graph


def compile_graph():
    """
    Compiles the graph for execution.
    """
    graph = build_graph()
    return graph.compile()


# Pre-compiled graph for import
verifai_graph = compile_graph()
