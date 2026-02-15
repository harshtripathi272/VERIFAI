"""
VERIFAI LangGraph Workflow

Defines the complete multi-agent DAG with debate-based consensus.
All agent invocations are logged to the SQL database automatically.

NEW FLOW:
START → Radiologist → CheXbert → Evidence Gathering (Hist + Lit parallel) → Critic → Debate → Chief/Finalize → END
"""

import uuid
from langgraph.graph import StateGraph, START, END
from concurrent.futures import ThreadPoolExecutor, wait

from graph.state import VerifaiState, FinalDiagnosis
from app.config import settings
from db.logger import AgentLogger

# Import agent nodes
from agents.radiologist.agent import radiologist_node
from agents.chexbert.agent import chexbert_node  # NEW: Structured pathology labeling
from agents.critic.agent import critic_node
from agents.historian.agent import historian_node
from agents.literature.agent import literature_agent_node as literature_node
from agents.debate.agent import debate_node
from agents.chief.agent import chief_node


# =============================================================================
# THREAD-LOCAL LOGGER REGISTRY (one logger per session)
# =============================================================================
import threading
_logger_registry: dict[str, AgentLogger] = {}
_registry_lock = threading.Lock()


def _get_or_create_logger(state: VerifaiState) -> AgentLogger:
    """Get or create a logger for the current workflow session."""
    session_id = state.get("_session_id")
    
    if session_id and session_id in _logger_registry:
        return _logger_registry[session_id]
    
    # Create new session
    session_id = session_id or str(uuid.uuid4())
    logger = AgentLogger(
        session_id=session_id,
        image_path=state.get("image_path", ""),
        patient_id=state.get("patient_id"),
        workflow_type="debate"
    )
    
    with _registry_lock:
        _logger_registry[session_id] = logger
    
    return logger


def _cleanup_logger(session_id: str):
    """Remove logger from registry after session completes."""
    with _registry_lock:
        _logger_registry.pop(session_id, None)


# =============================================================================
# LOGGED AGENT NODE WRAPPERS
# =============================================================================

def logged_radiologist_node(state: VerifaiState) -> dict:
    """Radiologist node with automatic DB logging."""
    logger = _get_or_create_logger(state)
    result = radiologist_node(state)
    try:
        logger.log_radiologist(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log radiologist: {e}")
    return result


def logged_critic_node(state: VerifaiState) -> dict:
    """Critic node with automatic DB logging."""
    logger = _get_or_create_logger(state)
    result = critic_node(state)
    try:
        logger.log_critic(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log critic: {e}")
    return result


def logged_evidence_gathering_node(state: VerifaiState) -> dict:
    """Evidence gathering node with automatic DB logging."""
    logger = _get_or_create_logger(state)
    result = evidence_gathering_node(state)
    try:
        logger.log_evidence_gathering(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log evidence_gathering: {e}")
    return result


def logged_debate_node(state: VerifaiState) -> dict:
    """Debate node with automatic DB logging."""
    logger = _get_or_create_logger(state)
    result = debate_node(state)
    try:
        logger.log_debate(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log debate: {e}")
    return result


def logged_chief_node(state: VerifaiState) -> dict:
    """Chief node with automatic DB logging + session completion."""
    logger = _get_or_create_logger(state)
    result = chief_node(state)
    try:
        logger.log_chief(state, result)
        final_dx = result.get("final_diagnosis")
        if final_dx:
            logger.complete_session(final_diagnosis=final_dx)
        _cleanup_logger(logger.session_id)
    except Exception as e:
        print(f"[DB LOG] Failed to log chief: {e}")
    return result


def logged_finalize_node(state: VerifaiState) -> dict:
    """Finalize node with automatic DB logging + session completion."""
    logger = _get_or_create_logger(state)
    result = finalize_node(state)
    try:
        logger.log_finalize(state, result)
        final_dx = result.get("final_diagnosis")
        if final_dx:
            logger.complete_session(final_diagnosis=final_dx)
        _cleanup_logger(logger.session_id)
    except Exception as e:
        print(f"[DB LOG] Failed to log finalize: {e}")
    return result



# EVIDENCE GATHERING (unchanged logic)
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
    RadiologistOutput is now plain text (findings + impression),
    so we rely on debate consensus or impression text for diagnosis.
    """
    rad = state.get("radiologist_output")
    debate = state.get("debate_output")
    hist = state.get("historian_output")
    lit = state.get("literature_output")
    kle_uncertainty = state.get("radiologist_kle_uncertainty", 0.5)
    
    if not rad or not rad.impression:
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
        # Fallback: use impression text as diagnosis, KLE-based confidence
        # Base confidence = 1.0 - KLE uncertainty (higher uncertainty = lower confidence)
        confidence = max(0.1, 1.0 - kle_uncertainty)
        
        if hist:
            confidence += hist.confidence_adjustment
        
        if lit and hasattr(lit, 'overall_evidence_strength') and lit.overall_evidence_strength in ["medium", "high"]:
            confidence += 0.05 if lit.overall_evidence_strength == "medium" else 0.10
        
        # Apply debate adjustment if available
        if debate:
            confidence += debate.total_confidence_adjustment
        
        confidence = max(0.0, min(0.99, confidence))
        
        # Use impression as the diagnosis text
        impression_preview = rad.impression[:200] if len(rad.impression) > 200 else rad.impression
        
        final = FinalDiagnosis(
            diagnosis=impression_preview,
            calibrated_confidence=confidence,
            deferred=False,
            explanation=f"Based on radiologist findings with KLE uncertainty={kle_uncertainty:.3f}. {rad.findings[:100]}...",
            recommended_next_steps=["Confirm with clinical correlation", "Consider follow-up imaging if symptoms persist"]
        )
        trace_entry = f"FINALIZE: {impression_preview[:80]}... (confidence={confidence:.2%})"
    
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
    All nodes are wrapped with automatic SQL logging.
    
    UPDATED Flow (Sequential Reasoning Depth):
    START → Radiologist → Evidence Gathering (Hist + Lit parallel) → Critic → Debate →┬→ Finalize → END
                                                                                        └→ Chief → END
    
    CRITICAL: Evidence gathering MUST complete before Critic evaluation.
    This ensures Critic evaluates fully enriched diagnostic context:
    - Imaging findings (Radiologist)
    - Clinical history (Historian FHIR data)
    - Literature evidence (Literature citations)
    - Epistemic uncertainty (KLE score)
    """
    graph = StateGraph(VerifaiState)
    
    # === Add Logged Nodes ===
    graph.add_node("radiologist", logged_radiologist_node)
    graph.add_node("evidence_gathering", logged_evidence_gathering_node)  # Parallel Hist + Lit
    graph.add_node("chexbert", chexbert_node)
    graph.add_node("critic", logged_critic_node)
    graph.add_node("debate", logged_debate_node)
    graph.add_node("chief", logged_chief_node)
    graph.add_node("finalize", logged_finalize_node)
    
    # === Define Edges ==
    
    # Entry: START → Radiologist
    graph.add_edge(START, "radiologist")
    
    # NEW: Radiologist → CheXbert (label findings immediately)
    graph.add_edge("radiologist", "chexbert")
    
    # NEW: CheXbert → Evidence Gathering (gather context with structured labels)
    graph.add_edge("chexbert", "evidence_gathering")
    
    # NEW: Evidence Gathering → Critic (evaluate WITH full context)
    graph.add_edge("evidence_gathering", "critic")
    
    # Critic → Debate (adversarial reconciliation with enriched context)
    graph.add_edge("critic", "debate")
    
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
