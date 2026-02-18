"""
VERIFAI LangGraph Workflow

Defines the complete multi-agent DAG with debate-based consensus.
All agent invocations are logged to the SQL database automatically.

NEW FLOW:
START → Radiologist → CheXbert → Evidence Gathering (Hist + Lit parallel) → Critic → Debate → Chief/Finalize → END
"""

import uuid
from typing import Any
from langgraph.graph import StateGraph, START, END
from concurrent.futures import ThreadPoolExecutor, wait

from graph.state import VerifaiState, FinalDiagnosis
from app.config import settings
from db.adapter import get_logger  # Unified database adapter (SQLite or Supabase)


# Import agent nodes
from agents.radiologist.agent import radiologist_node
from agents.chexbert.agent import chexbert_node
from agents.critic.agent import critic_node
from agents.historian.agent import historian_node
from agents.literature.agent import literature_agent_node as literature_node
from agents.debate.agent import debate_node
from agents.validator import validator_node, initialize_validator_tools  # Validator: runs after debate always
from agents.feedback.agent import feedback_node  # Doctor feedback processing


# =============================================================================
# THREAD-LOCAL LOGGER REGISTRY (one logger per session)
# =============================================================================
import threading
_logger_registry: dict[str, Any] = {}
_registry_lock = threading.Lock()


def _get_or_create_logger(state: VerifaiState):
    """Get or create a logger for the current workflow session."""
    session_id = state.get("_session_id")
    
    if session_id and session_id in _logger_registry:
        return _logger_registry[session_id]
    
    # Create new session
    session_id = session_id or str(uuid.uuid4())
    logger = get_logger(  # NEW: Uses adapter to select SQLite or Supabase
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
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Radiologist Node")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = radiologist_node(state)
    print(f"[WORKFLOW] Radiologist completed - Generated {len(result.get('radiologist_output', {}).findings or '')} chars of findings")
    try:
        logger.log_radiologist(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log radiologist: {e}")
    return result


def logged_critic_node(state: VerifaiState) -> dict:
    """Critic node with automatic DB logging."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Critic Node")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = critic_node(state)
    critic_output = result.get('critic_output')
    if critic_output:
        print(f"[WORKFLOW] Critic completed - Risk Score: {critic_output.final_risk_score:.2f}")
    try:
        logger.log_critic(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log critic: {e}")
    return result


def logged_evidence_gathering_node(state: VerifaiState) -> dict:
    """Evidence gathering node with automatic DB logging."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Evidence Gathering (Historian + Literature in parallel)")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = evidence_gathering_node(state)
    print(f"[WORKFLOW] Evidence gathering completed")
    print(f"  - Historian output: {'✓' if result.get('historian_output') else '✗'}")
    print(f"  - Literature output: {'✓' if result.get('literature_output') else '✗'}")
    try:
        logger.log_evidence_gathering(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log evidence_gathering: {e}")
    return result


def logged_debate_node(state: VerifaiState) -> dict:
    """Debate node with automatic DB logging."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Debate Node")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = debate_node(state)
    debate_output = result.get('debate_output')
    if debate_output:
        print(f"[WORKFLOW] Debate completed - Rounds: {len(debate_output.rounds)}, Consensus: {debate_output.final_consensus}")
    try:
        logger.log_debate(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log debate: {e}")
    return result


def logged_validator_node(state: VerifaiState) -> dict:
    """Validator node — runs after debate in BOTH scenarios (consensus + max-rounds exceeded)."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Validator Node")
    debate = state.get("debate_output")
    if debate and debate.final_consensus:
        print("[WORKFLOW] Validator mode: CONSENSUS VALIDATION")
    else:
        print("[WORKFLOW] Validator mode: ESCALATION (max rounds exceeded)")
    print("="*60)
    result = validator_node(state)
    recommendation = (result.get("validator_output") or {}).get("recommendation", "FINALIZE")
    print(f"[WORKFLOW] Validator completed - Recommendation: {recommendation}")
    return result


def logged_finalize_node(state: VerifaiState) -> dict:
    """Finalize node with automatic DB logging + session completion."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Finalize Node")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = finalize_node(state)
    final_dx = result.get("final_diagnosis")
    if final_dx:
        print(f"[WORKFLOW] Finalize completed - Diagnosis: {final_dx.diagnosis[:50] if final_dx.diagnosis else 'None'}... Confidence: {final_dx.calibrated_confidence:.2%}")
    try:
        logger.log_finalize(state, result)
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
    Finalize node: builds the FinalDiagnosis from debate + validator signals.

    Validator recommendation effects:
    - FINALIZE              → full confidence, no changes
    - FINALIZE_LOW_CONFIDENCE → confidence capped at 0.65, note added
    - FLAG_FOR_HUMAN        → deferred=True, deferral_reason set
    """
    rad = state.get("radiologist_output")
    debate = state.get("debate_output")
    hist = state.get("historian_output")
    lit = state.get("literature_output")
    kle_uncertainty = state.get("radiologist_kle_uncertainty", 0.5)
    validator_out = state.get("validator_output") or {}
    recommendation = validator_out.get("recommendation", "FINALIZE")
    validator_explanation = validator_out.get("explanation", "")

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

    # ── FLAG_FOR_HUMAN: validator says evidence is weak / critical rule violated ──
    if recommendation == "FLAG_FOR_HUMAN":
        return {
            "final_diagnosis": FinalDiagnosis(
                diagnosis=debate.consensus_diagnosis if (debate and debate.final_consensus) else rad.impression[:200],
                calibrated_confidence=0.0,
                deferred=True,
                deferral_reason=f"Validator flagged for human review: {validator_explanation}",
                recommended_next_steps=[
                    "Manual radiologist review required",
                    "Check validator flags: " + str(validator_out.get("rules", {}).get("triggered_rule_names", [])),
                    "Review retrieved historical cases in validator_output"
                ]
            ),
            "trace": [f"FINALIZE: DEFERRED — Validator flagged for human review ({validator_explanation})"]
        }

    # ── Build base confidence ─────────────────────────────────────────────────
    if debate and debate.final_consensus:
        diagnosis_text = debate.consensus_diagnosis
        confidence = debate.consensus_confidence
        base_explanation = f"Consensus reached through {len(debate.rounds)}-round debate. {debate.debate_summary}"
    else:
        diagnosis_text = rad.impression[:200]
        confidence = max(0.1, 1.0 - kle_uncertainty)
        if hist:
            confidence += hist.confidence_adjustment
        if lit and hasattr(lit, "overall_evidence_strength") and lit.overall_evidence_strength in ["medium", "high"]:
            confidence += 0.05 if lit.overall_evidence_strength == "medium" else 0.10
        if debate:
            confidence += debate.total_confidence_adjustment
        confidence = max(0.0, min(0.99, confidence))
        base_explanation = f"No debate consensus after {len(debate.rounds) if debate else 0} rounds. Based on radiologist impression with KLE={kle_uncertainty:.3f}."

    # ── FINALIZE_LOW_CONFIDENCE: cap at 0.65 ─────────────────────────────────
    if recommendation == "FINALIZE_LOW_CONFIDENCE":
        confidence = min(confidence, 0.65)
        base_explanation += f" Validator confidence reduced: {validator_explanation}"

    final = FinalDiagnosis(
        diagnosis=diagnosis_text,
        calibrated_confidence=confidence,
        deferred=False,
        explanation=base_explanation,
        recommended_next_steps=[
            "Confirm with clinical correlation",
            "Consider follow-up imaging if symptoms persist"
        ]
    )

    trace_entry = f"FINALIZE: {diagnosis_text[:80] if diagnosis_text else 'None'}... (confidence={confidence:.2%}, validator={recommendation})"
    return {
        "final_diagnosis": final,
        "trace": [trace_entry]
    }


def route_after_debate(state: VerifaiState) -> str:
    """
    After debate, ALWAYS go to validator — regardless of whether
    consensus was reached or max rounds exceeded.

    Scenario 1: Debate reached consensus → Validator validates it.
    Scenario 2: Debate hit max rounds without consensus → Validator escalates with evidence.
    """
    return "validator"


def route_after_validator(state: VerifaiState) -> str:
    """
    After validator, always proceed to finalize.
    The validator_output.recommendation field (FINALIZE / FINALIZE_LOW_CONFIDENCE /
    FLAG_FOR_HUMAN) is stored in state and consumed by finalize_node.
    """
    return "finalize"


def should_start_from_critic(state: VerifaiState) -> str:
    """
    Route decision for feedback-driven reprocessing.
    
    - If is_feedback_iteration=True → go directly to critic (skip radiologist/chexbert/evidence)
    - Otherwise → normal flow starting from radiologist
    
    This allows doctor feedback to restart the workflow from critic
    with all the original context preserved.
    """
    is_feedback = state.get("is_feedback_iteration", False)
    
    if is_feedback:
        return "critic_feedback"  # Special path for feedback iteration
    else:
        return "radiologist"  # Normal path


def build_workflow() -> StateGraph:
    """
    Constructs the VERIFAI LangGraph DAG with debate + validator mechanism.
    All nodes are wrapped with automatic SQL logging.

    NORMAL Flow:
    START → Radiologist → CheXbert → Evidence Gathering (Hist+Lit parallel)
          → Critic → Debate → Validator → Finalize → END

    Validator runs in BOTH debate outcomes:
      ✅ Consensus reached   → Validator validates the consensus
      ⚠️ Max rounds exceeded → Validator escalates with evidence

    FEEDBACK Flow (doctor rejects diagnosis):
    START → [routing] → Critic (with feedback context) → Debate → Validator → Finalize → END

    No Chief node — Validator is the final arbitration layer.
    """
    graph = StateGraph(VerifaiState)

    # === Nodes ===
    graph.add_node("radiologist", logged_radiologist_node)
    graph.add_node("chexbert", chexbert_node)
    graph.add_node("evidence_gathering", logged_evidence_gathering_node)
    graph.add_node("critic", logged_critic_node)
    graph.add_node("critic_feedback", logged_critic_node)  # Same logic, different entry point
    graph.add_node("debate", logged_debate_node)
    graph.add_node("validator", logged_validator_node)     # NEW: always runs after debate
    graph.add_node("finalize", logged_finalize_node)

    # === Edges ===

    # START → Conditional: normal flow vs feedback iteration
    graph.add_conditional_edges(
        START,
        should_start_from_critic,
        {
            "radiologist": "radiologist",
            "critic_feedback": "critic_feedback"
        }
    )

    # NORMAL FLOW
    graph.add_edge("radiologist", "chexbert")
    graph.add_edge("chexbert", "evidence_gathering")
    graph.add_edge("evidence_gathering", "critic")

    # FEEDBACK FLOW (skips evidence gathering, uses preserved context)
    graph.add_edge("critic_feedback", "debate")

    # Critic → Debate
    graph.add_edge("critic", "debate")

    # Debate → Validator (ALWAYS — both consensus and no-consensus paths)
    graph.add_conditional_edges(
        "debate",
        route_after_debate,
        {"validator": "validator"}
    )

    # Validator → Finalize (recommendation stored in state, consumed by finalize_node)
    graph.add_conditional_edges(
        "validator",
        route_after_validator,
        {"finalize": "finalize"}
    )

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
