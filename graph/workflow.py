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
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from concurrent.futures import ThreadPoolExecutor, wait

from graph.state import VerifaiState, FinalDiagnosis
from app.config import settings
from db.adapter import get_logger  # Unified database adapter (SQLite or Supabase)
from uncertainty.muc import (
    compute_ig,
    compute_historian_uncertainty, compute_historian_alignment,
    compute_literature_uncertainty, compute_literature_alignment,
    compute_debate_ds_fusion,
    compute_validator_uncertainty, compute_validator_alignment,
    compute_critic_uncertainty, compute_critic_alignment,
)


# Import agent nodes
from agents.radiologist.agent import radiologist_node
from agents.chexbert.agent import chexbert_node
from agents.critic.agent import critic_node
from agents.historian.agent import historian_node
from agents.literature.agent import literature_agent_node as literature_node
from agents.debate.agent import debate_node
from agents.validator import validator_node, initialize_validator_tools  # Validator: runs after debate always
from agents.feedback.agent import feedback_node  # Doctor feedback processing
from graph.state import DoctorFeedback


# THREAD-LOCAL LOGGER REGISTRY (one logger per session)

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
    u_in = state.get("current_uncertainty", 0.50)
    print(f"  ⤷ Uncertainty IN  : {u_in:.2%}")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = radiologist_node(state)
    u_out = result.get("current_uncertainty", u_in)
    delta = u_out - u_in
    print(f"[WORKFLOW] Radiologist completed - Generated {len(result.get('radiologist_output', {}).findings or '')} chars of findings")
    print(f"  ⤷ Uncertainty OUT : {u_out:.2%}  (Δ = {delta:+.3f})")
    try:
        logger.log_radiologist(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log radiologist: {e}")
    return result


def logged_chexbert_node(state: VerifaiState) -> dict:
    """CheXbert node with automatic DB logging and MUC uncertainty tracking."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting CheXbert Node")
    u_in = state.get("current_uncertainty", 0.50)
    print(f"  ⤷ Uncertainty IN  : {u_in:.2%}")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = chexbert_node(state)
    chex_output = result.get('chexbert_output')
    u_out = result.get("current_uncertainty", u_in)
    delta = u_out - u_in
    if chex_output:
        num_present = sum(1 for s in chex_output.labels.values() if s == "present")
        num_uncertain = sum(1 for s in chex_output.labels.values() if s == "uncertain")
        print(f"[WORKFLOW] CheXbert completed")
        print(f"  Present      : {num_present}")
        print(f"  Uncertain    : {num_uncertain}")
        if chex_output.labels:
            labels_str = ", ".join([f"{c} ({s})" for c, s in chex_output.labels.items()])
            print(f"  Labels       : {labels_str}")
    else:
        print("[WORKFLOW] CheXbert completed - No output")
    print(f"  ⤷ Uncertainty OUT : {u_out:.2%}  (Δ = {delta:+.3f})")
    # Log trace entries from agent
    for t in result.get("trace", []):
        if "MUC" in t:
            print(f"  {t}")
    return result


def logged_critic_node(state: VerifaiState) -> dict:
    """Critic node with automatic DB logging."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Critic Node")
    u_in = state.get("current_uncertainty", 0.50)
    print(f"  ⤷ Uncertainty IN  : {u_in:.2%}")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = critic_node(state)
    critic_output = result.get('critic_output')
    u_out = result.get("current_uncertainty", u_in)
    delta = u_out - u_in
    if critic_output:
        print(f"[WORKFLOW] Critic completed")
        print(f"  Safety Score   : {critic_output.safety_score:.3f}")
        print(f"  Overconfident  : {critic_output.is_overconfident}")
        print(f"  Historical Risk: {critic_output.historical_risk_level.upper()}")
        print(f"  Similar Mistakes: {critic_output.similar_mistakes_count}")
        if critic_output.concern_flags:
            print(f"  Concern Flags  :")
            for flag in critic_output.concern_flags[:5]:  # Top 5
                print(f"    - {flag}")
        if critic_output.recommended_hedging:
            print(f"  Hedging        : {critic_output.recommended_hedging[:120]}")
        if critic_output.historical_context:
            print(f"  Historical Context ({len(critic_output.historical_context)} matched cases):")
            for ctx in critic_output.historical_context[:3]:
                print(f"    [{ctx.get('disease_type','?')}] err={ctx.get('error_type','?')} sim={ctx.get('similarity',0):.3f}")
    else:
        print("[WORKFLOW] Critic completed - No output")
    print(f"  ⤷ Uncertainty OUT : {u_out:.2%}  (Δ = {delta:+.3f})")
    try:
        logger.log_critic(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log critic: {e}")
    return result


def logged_evidence_gathering_node(state: VerifaiState) -> dict:
    """Evidence gathering node with automatic DB logging."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Evidence Gathering (Historian + Literature in parallel)")
    u_in = state.get("current_uncertainty", 0.50)
    print(f"  ⤷ Uncertainty IN  : {u_in:.2%}")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = evidence_gathering_node(state)
    # === MUC: Compute IG for Historian ===
    u_current = u_in
    hist = result.get('historian_output')
    if hist:
        supporting_count = len(hist.supporting_facts)
        contradicting_count = len(hist.contradicting_facts)
        hist_unc = compute_historian_uncertainty(supporting_count, contradicting_count)
        hist_align = compute_historian_alignment(
            supporting_count, contradicting_count, hist.confidence_adjustment
        )
        hist_ig = compute_ig(
            agent_name="historian",
            agent_uncertainty=hist_unc,
            alignment_score=hist_align,
            system_uncertainty=u_current,
        )
        u_current = hist_ig.system_uncertainty_after
        print(f"  Historian:")
        print(f"    Supporting/Contradicting : {supporting_count}/{contradicting_count}")
        print(f"    Confidence Adjustment    : {hist.confidence_adjustment:+.3f}")
        print(f"    MUC: unc={hist_unc:.3f}, align={hist_align:.3f}, IG={hist_ig.information_gain:.4f}")
        if hist.supporting_facts:
            for f in hist.supporting_facts[:2]:
                print(f"      + [{f.fhir_resource_type}] {f.description[:100]}")
        if hist.contradicting_facts:
            for f in hist.contradicting_facts[:2]:
                print(f"      - [{f.fhir_resource_type}] {f.description[:100]}")
    else:
        print(f"  Historian: No output (patient_id may be missing or no FHIR data)")

    # === MUC: Compute IG for Literature ===
    lit = result.get('literature_output')
    if lit:
        if isinstance(lit, str):
            # Literature is a synthesis string — parse keywords for evidence strength
            lit_lower = lit.lower()
            if any(kw in lit_lower for kw in ["strongly support", "high evidence", "robust evidence"]):
                ev_strength = "high"
            elif any(kw in lit_lower for kw in ["contradicts", "does not support", "inconsistent"]):
                ev_strength = "low"
            elif any(kw in lit_lower for kw in ["limited evidence", "weak evidence", "insufficient"]):
                ev_strength = "low"
            else:
                ev_strength = "medium"
            has_contra = any(kw in lit_lower for kw in ["contradicts", "argues against", "inconsistent"])
            lit_unc = compute_literature_uncertainty(
                citation_count=lit_lower.count("doi:") + lit_lower.count("pmid") + 1,
                evidence_strength=ev_strength,
            )
            lit_align = compute_literature_alignment(
                evidence_strength=ev_strength,
                has_contradicting_differentials=has_contra,
                synthesis_text=lit,
            )
        else:
            # Structured literature output
            citations = getattr(lit, 'citations', [])
            ev_strength = getattr(lit, 'overall_evidence_strength', 'medium')
            high_count = sum(1 for c in citations if getattr(c, 'strength', '') == 'high')
            high_ratio = high_count / len(citations) if citations else 0.0
            lit_unc = compute_literature_uncertainty(
                citation_count=len(citations),
                evidence_strength=ev_strength,
                high_strength_ratio=high_ratio,
            )
            lit_align = compute_literature_alignment(
                evidence_strength=ev_strength,
            )

        lit_ig = compute_ig(
            agent_name="literature",
            agent_uncertainty=lit_unc,
            alignment_score=lit_align,
            system_uncertainty=u_current,
        )
        u_current = lit_ig.system_uncertainty_after
        print(f"  Literature: [{ev_strength.upper()}]")
        print(f"    MUC: unc={lit_unc:.3f}, align={lit_align:.3f}, IG={lit_ig.information_gain:.4f}")
    else:
        print(f"  Literature: No output")

    # Store updated uncertainty in result
    result["current_uncertainty"] = u_current
    print(f"  Uncertainty OUT : {u_current:.2%}  (Δ = {u_current - u_in:+.3f})")

    try:
        logger.log_evidence_gathering(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log evidence_gathering: {e}")
    return result


def logged_debate_node(state: VerifaiState) -> dict:
    """Debate node with automatic DB logging."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Debate Node")
    u_in = state.get("current_uncertainty", 0.50)
    print(f"  ⤷ Uncertainty IN  : {u_in:.2%}")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = debate_node(state)
    debate_output = result.get('debate_output')
    u_out = result.get("current_uncertainty", u_in)
    delta = u_out - u_in
    if debate_output:
        print(f"\n[WORKFLOW] Debate completed")
        print(f"  Rounds       : {len(debate_output.rounds)}")
        print(f"  Consensus    : {'YES' if debate_output.final_consensus else 'NO'}")
        print(f"  Confidence   : {debate_output.consensus_confidence:.2%}")
        print(f"  Total Δ      : {debate_output.total_confidence_adjustment:+.3f}")
    print(f"  ⤷ Uncertainty OUT : {u_out:.2%}  (Δ = {delta:+.3f})")
    try:
        logger.log_debate(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log debate: {e}")
    return result


def logged_validator_node(state: VerifaiState) -> dict:
    """Validator node — runs after debate in BOTH scenarios (consensus + max-rounds exceeded)."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Validator Node")
    u_in = state.get("current_uncertainty", 0.50)
    print(f"  ⤷ Uncertainty IN  : {u_in:.2%}")
    debate = state.get("debate_output")
    if debate and debate.final_consensus:
        print("[WORKFLOW] Validator mode: CONSENSUS VALIDATION")
    else:
        print("[WORKFLOW] Validator mode: ESCALATION (max rounds exceeded)")
    print("="*60)
    result = validator_node(state)
    # === MUC: Compute Validator IG ===
    vout = result.get("validator_output") or {}
    entity = vout.get('entity_matching', {})
    rules = vout.get('rules', {})
    retrieval = vout.get('retrieval', {})
    recommendation = vout.get('recommendation', 'FINALIZE')

    entity_f1 = entity.get('entity_f1')
    if isinstance(entity_f1, str):
        try:
            entity_f1 = float(entity_f1)
        except (ValueError, TypeError):
            entity_f1 = None

    val_unc = compute_validator_uncertainty(
        entity_f1=entity_f1 if entity_f1 is not None else 0.5,
        has_critical_flags=rules.get('has_critical_flag', False),
        flag_count=rules.get('flag_count', 0),
        retrieval_agrees=retrieval.get('agrees_with_chexbert', True) if retrieval and not retrieval.get('error') else True,
    )
    val_align = compute_validator_alignment(
        recommendation=recommendation,
        entity_f1=entity_f1 if entity_f1 is not None else 0.5
    )
    val_ig = compute_ig(
        agent_name="validator",
        agent_uncertainty=val_unc,
        alignment_score=val_align,
        system_uncertainty=u_in,
    )

    print(f"  Recommendation : {recommendation}")
    print(f"  MUC: unc={val_unc:.3f}, align={val_align:.3f}, IG={val_ig.information_gain:.4f}")
    print(f"  Uncertainty OUT : {val_ig.system_uncertainty_after:.2%}  (Δ = {val_ig.system_uncertainty_after - u_in:+.3f})")

    # Store updated uncertainty
    result["current_uncertainty"] = val_ig.system_uncertainty_after
    return result


def logged_finalize_node(state: VerifaiState) -> dict:
    """Finalize node with automatic DB logging + session completion."""
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Finalize Node")
    u_in = state.get("current_uncertainty", 0.50)
    print(f"  ⤷ Uncertainty IN  : {u_in:.2%}")
    print("="*60)
    logger = _get_or_create_logger(state)
    result = finalize_node(state)
    final_dx = result.get("final_diagnosis")
    if final_dx:
        final_u = max(0.01, 1.0 - final_dx.calibrated_confidence)
        print(f"[WORKFLOW] Finalize completed")
        print(f"  Diagnosis    : {final_dx.diagnosis[:80] if final_dx.diagnosis else 'None'}")
        print(f"  Confidence   : {final_dx.calibrated_confidence:.2%}")
        print(f"  Deferred     : {final_dx.deferred}")
        print(f"  ⤷ Final Uncertainty : {final_u:.2%}")
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
            # NOTE: 300s timeout because both agents share the GPU inference lock
            # and must execute sequentially - literature can take 60-120s alone.
            try:
                historian_result = historian_future.result(timeout=300)
                results["historian_output"] = historian_result.get("historian_output")
                trace_entries.extend(historian_result.get("trace", []))
            except Exception as e:
                trace_entries.append(f"EVIDENCE_GATHER: Historian failed - {str(e)[:100]}")
            
            try:
                literature_result = literature_future.result(timeout=300)
                results["literature_output"] = literature_result.get("literature_output")
                trace_entries.extend(literature_result.get("trace", []))
            except Exception as e:
                trace_entries.append(f"EVIDENCE_GATHER: Literature failed - {str(e)[:100]}")
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
    uncertainty = state.get("current_uncertainty", 0.5)
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
        confidence = max(0.1, 1.0 - uncertainty)
        if hist:
            confidence += hist.confidence_adjustment
        if lit and hasattr(lit, "overall_evidence_strength") and lit.overall_evidence_strength in ["medium", "high"]:
            confidence += 0.05 if lit.overall_evidence_strength == "medium" else 0.10
        if debate:
            confidence += debate.total_confidence_adjustment
        confidence = max(0.0, min(0.99, confidence))
        base_explanation = f"No debate consensus after {len(debate.rounds) if debate else 0} rounds. Based on radiologist impression with uncertainty={uncertainty:.3f}."

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


def human_review_node(state: VerifaiState) -> dict:
    """
    Human-in-the-Loop Node.
    Gathers the outputs from the workflow (Diagnosis, Citations, Heatmaps, Context)
    and yields them to the human via a LangGraph interrupt().
    If the human rejects and provides context, we loop back to the Critic.
    """
    print("\n" + "="*60)
    print("[WORKFLOW] Starting Human Review Node")
    print("="*60)

    final = state.get("final_diagnosis")
    rad = state.get("radiologist_output")
    lit = state.get("literature_output")
    hist = state.get("historian_output")
    session_id = state.get("_session_id", "unknown_session")

    # Compile presentation data
    data_to_human = {
        "session_id": session_id,
        "diagnosis": final.diagnosis if final else None,
        "confidence": final.calibrated_confidence if final else 0.0,
        "deferred": final.deferred if final else False,
        "explanation": final.explanation if final else "",
        "heatmap_paths": rad.heatmap_paths if rad and getattr(rad, "heatmap_paths", None) else {},
    }

    if lit and hasattr(lit, "citations"):
        data_to_human["literature_citations"] = [{"title": c.title, "url": c.url} for c in lit.citations]
    elif isinstance(lit, str):
         data_to_human["literature_summary"] = lit

    if hist and hasattr(hist, "supporting_facts"):
         data_to_human["historical_supporting"] = [f.description for f in hist.supporting_facts]
         data_to_human["historical_contradicting"] = [f.description for f in getattr(hist, "contradicting_facts", [])]

    # Interrupt the graph and wait for human response
    print("[WORKFLOW] Halting for Human Review...")
    response = interrupt(data_to_human)

    # Response schema expected from human:
    # {"action": "approve" | "reject", "feedback": str, "correct_diagnosis": str (optional)}
    
    if not isinstance(response, dict):
        response = {"action": "approve", "feedback": str(response)}

    action = response.get("action", "approve").lower()
    
    if action == "reject":
        print("[WORKFLOW] Human rejected the diagnosis. Restarting workflow via Critic...")
        
        # Build DoctorFeedback for reprocessing
        df = DoctorFeedback(
            feedback_id=1, # Mock ID, normally generated by DB
            original_session_id=session_id,
            feedback_type='rejection',
            doctor_notes=response.get("feedback", "Human rejected the diagnosis without specifics."),
            correct_diagnosis=response.get("correct_diagnosis", None)
        )
        
        return {
             "doctor_feedback": df,
             "is_feedback_iteration": True,
             "trace": [f"HUMAN REVIEW: Rejected diagnosis. Feedback: {df.doctor_notes[:100]}"]
        }
    else:
        print("[WORKFLOW] Human approved the diagnosis.")
        # If approved, just pass through and end
        return {
            "trace": ["HUMAN REVIEW: Approved diagnosis."]
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


def route_after_human_review(state: VerifaiState) -> str:
    """
    Routes based on the result of the human review.
    If the human rejected (is_feedback_iteration=True), loop back to Critic.
    Otherwise, END workflow.
    """
    if state.get("is_feedback_iteration", False):
         return "critic_feedback"
    return "end"


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
    graph.add_node("chexbert", logged_chexbert_node)
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

    # Finalize → Human Review
    graph.add_edge("finalize", "human_review")
    graph.add_node("human_review", human_review_node)

    # Human Review → Conditional END or Critic Feedback
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
             "end": END,
             "critic_feedback": "critic_feedback"
        }
    )

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

# Use debate workflow by default with memory saver for check-pointing
memory = MemorySaver()
workflow = build_workflow()
app = workflow.compile(checkpointer=memory)
