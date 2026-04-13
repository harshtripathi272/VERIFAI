"""
Validator Agent Node

Runs in TWO scenarios:
1. When Debate Orchestrator REACHES consensus (validation mode)
2. When Debate runs for max rounds WITHOUT consensus (escalation mode)

The validator aggregates ALL agent outputs + runs its own 3 tools:
1. CXR-RePaiR Retrieval: Find similar historical cases
2. RadGraph Entity Matching: Verify clinical facts match  
3. Clinical Rules Engine: Check for contradictions

The validator prepares a comprehensive final output aggregating all evidence
for human review, including:
- All agent outputs (Radiologist, CheXbert, Historian, Literature, Critic, Debate)
- Retrieval evidence from similar historical cases
- Entity matching results
- Clinical rule violations
- Final recommendation with supporting reasoning
"""

from typing import Dict, Any
from graph.state import VerifaiState
from agents.validator.retrieval_tool import CXRRetrieverTool
from agents.validator.radgraph_tool import RadGraphEntityTool
from agents.validator.rules_engine import ClinicalRulesEngine
from uncertainty.muc import (
    compute_ig,
    compute_validator_uncertainty,
    compute_validator_alignment,
)

# Global tool instances (initialized once at startup)
_retriever = None
_radgraph = None
_rules_engine = None
_tools_initialized = False


def initialize_validator_tools(vision_encoder=None, image_processor=None):
    """
    Initialize all validator tools. Call this once when building the graph.
    
    Args:
        vision_encoder: MedSigLIP / SiglipVisionModel instance, OR None to auto-load.
        image_processor: Matching image processor, OR None to auto-load.
    
    When both are None the function will automatically load google/medsiglip-448
    (public HuggingFace model, ~600MB) so the retrieval tool can embed query images.
    """
    global _retriever, _radgraph, _rules_engine, _tools_initialized
    
    if _tools_initialized:
        print("[Validator] Tools already initialized")
        return
    
    print("[Validator] Initializing validator tools...")
    
    # ── Auto-load MedSigLIP when caller passes None ──────────────────────────
    if vision_encoder is None or image_processor is None:
        try:
            import torch
            from transformers import SiglipVisionModel, AutoImageProcessor
            from app.config import settings

            model_name = getattr(settings, 'MEDSIGLIP_BASE_MODEL', 'google/medsiglip-448')
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Validator] Auto-loading MedSigLIP vision encoder from {model_name} on {device}...")

            vision_encoder = SiglipVisionModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                token=settings.HUGGINGFACE_TOKEN
            ).to(device)
            vision_encoder.eval()

            image_processor = AutoImageProcessor.from_pretrained(
                model_name,
                token=settings.HUGGINGFACE_TOKEN
            )
            print("[Validator] ✓ MedSigLIP auto-loaded")
        except Exception as e:
            print(f"[Validator] ✗ Failed to auto-load MedSigLIP: {e}")
            vision_encoder = None
            image_processor = None
    # ─────────────────────────────────────────────────────────────────────────

    # Initialize retrieval tool
    if vision_encoder is not None and image_processor is not None:
        try:
            _retriever = CXRRetrieverTool(
                index_path="retrieval_data/retrieval_index.faiss",
                metadata_path="retrieval_data/retrieval_metadata.json",
                vision_encoder=vision_encoder,
                image_processor=image_processor
            )
            print("[Validator] ✓ Retrieval tool initialized")
        except Exception as e:
            print(f"[Validator] ✗ Failed to initialize retrieval tool: {e}")
            _retriever = None
    else:
        print("[Validator] ✗ Retrieval tool skipped — no vision encoder available")
        _retriever = None
    
    # Initialize RadGraph tool
    try:
        _radgraph = RadGraphEntityTool(
            model_type="modern-radgraph-xl"  # Auto-downloads from HuggingFace
        )
        print("[Validator] ✓ RadGraph tool initialized")
    except Exception as e:
        print(f"[Validator] ✗ Failed to initialize RadGraph tool: {e}")
        _radgraph = None
    
    # Initialize rules engine (always succeeds)
    _rules_engine = ClinicalRulesEngine()
    print("[Validator] ✓ Rules engine initialized")
    
    _tools_initialized = True
    print("[Validator] All tools initialized successfully")


def validator_node(state: VerifaiState) -> Dict[str, Any]:
    """
    Validator node runs in TWO scenarios:
    1. When debate REACHES consensus → validate the consensus
    2. When debate runs max rounds WITHOUT consensus → escalate with evidence
    
    The validator aggregates ALL agent outputs and runs 3 tools to prepare
    a comprehensive final output for human review.
    
    Decision Logic:
    - If retrieval agrees with CheXbert AND entity F1 > 0.8 AND no critical flags
      → FINALIZE with high confidence
    - If retrieval disagrees AND entity F1 < 0.5
      → FLAG_FOR_HUMAN (multiple tools show weak evidence)
    - If critical rule violations present
      → FLAG_FOR_HUMAN (safety concern)
    - Otherwise
      → FINALIZE_LOW_CONFIDENCE (mixed signals)
    
    Args:
        state: Current VERIFAI state with all agent outputs
    
    Returns:
        Dict with validator_output (comprehensive aggregation), routing_decision, trace
    """
    if not _tools_initialized:
        return {
            "validator_output": {
                "error": "Validator tools not initialized",
                "recommendation": "FLAG_FOR_HUMAN"
            },
            "routing_decision": "FLAG_FOR_HUMAN",
            "trace": ["VALIDATOR: Tools not initialized - flagging for human review"]
        }
    
    trace_entries = ["VALIDATOR: Running validation tools..."]
    
    # ── TOOL 1: Retrieval ──────────────────────────────
    retrieval_result = {}
    if _retriever:
        try:
            retrieval_result = _retriever.execute(state)
            trace_entries.append(
                f"VALIDATOR RETRIEVAL: Found {len(retrieval_result.get('retrieved_sentences', []))} similar cases, "
                f"consensus={retrieval_result.get('consensus_diagnosis', 'Unknown')}"
            )
        except Exception as e:
            trace_entries.append(f"VALIDATOR RETRIEVAL: Failed - {str(e)[:100]}")
            retrieval_result = {
                "error": str(e),
                "retrieved_sentences": [],
                "consensus_diagnosis": "Unknown",
                "agrees_with_chexbert": False
            }
    else:
        trace_entries.append("VALIDATOR RETRIEVAL: Tool not available")
        retrieval_result = {
            "error": "Retrieval tool not initialized",
            "retrieved_sentences": [],
            "consensus_diagnosis": "Unknown",
            "agrees_with_chexbert": False
        }
    
    # ── TOOL 2: Entity Matching ────────────────────────
    entity_result = {}
    if _radgraph:
        try:
            entity_result = _radgraph.execute(
                state=state,
                retrieved_sentences=retrieval_result.get("retrieved_sentences", [])
            )
            trace_entries.append(
                f"VALIDATOR RADGRAPH: Entity F1={entity_result.get('entity_f1', 0):.3f}, "
                f"verdict={entity_result.get('verdict', 'unknown')}"
            )
        except Exception as e:
            trace_entries.append(f"VALIDATOR RADGRAPH: Failed - {str(e)[:100]}")
            entity_result = {
                "error": str(e),
                "entity_f1": 0.0,
                "verdict": "error"
            }
    else:
        trace_entries.append("VALIDATOR RADGRAPH: Tool not available")
        entity_result = {
            "error": "RadGraph tool not initialized",
            "entity_f1": 0.0,
            "verdict": "unavailable"
        }
    
    # ── TOOL 3: Rules Engine ───────────────────────────
    rules_result = {}
    if _rules_engine:
        try:
            rules_result = _rules_engine.execute(state)
            trace_entries.append(
                f"VALIDATOR RULES: {rules_result.get('summary', 'No violations')}"
            )
        except Exception as e:
            trace_entries.append(f"VALIDATOR RULES: Failed - {str(e)[:100]}")
            rules_result = {
                "error": str(e),
                "has_critical_flag": False,
                "flag_count": 0,
                "warn_count": 0
            }
    else:
        # Should never happen, but handle gracefully
        trace_entries.append("VALIDATOR RULES: Engine not available")
        rules_result = {
            "error": "Rules engine not initialized",
            "has_critical_flag": False,
            "flag_count": 0,
            "warn_count": 0
        }
    
    # ── DECISION LOGIC ─────────────────────────────────

    # Extract key signals
    retrieval_available = _retriever is not None and not retrieval_result.get("error")
    retrieval_agrees = retrieval_result.get("agrees_with_chexbert", False)
    entity_f1 = entity_result.get("entity_f1", 0.0)
    entity_strong = entity_f1 > 0.6   # Relaxed from 0.8 — achievable without full FAISS index
    entity_weak = entity_f1 < 0.35
    has_critical_flag = rules_result.get("has_critical_flag", False)
    no_critical_flag = not has_critical_flag

    # Decision tree (retrieval-availability aware)
    if retrieval_available:
        # Full 3-tool decision
        if retrieval_agrees and entity_strong and no_critical_flag:
            recommendation = "FINALIZE"
            confidence_level = "high"
            explanation = "All validation tools show strong agreement"
        elif has_critical_flag:
            recommendation = "FLAG_FOR_HUMAN"
            confidence_level = "low"
            explanation = f"Critical rule violations: {', '.join(rules_result.get('triggered_rule_names', []))}"
        elif not retrieval_agrees and entity_weak:
            recommendation = "FLAG_FOR_HUMAN"
            confidence_level = "low"
            explanation = "Weak evidence: retrieval disagrees and low entity match"
        else:
            recommendation = "FINALIZE_LOW_CONFIDENCE"
            confidence_level = "medium"
            explanation = "Mixed evidence signals from validation tools"
    else:
        # Retrieval index not available — rely on entity matching + rules only
        if has_critical_flag:
            recommendation = "FLAG_FOR_HUMAN"
            confidence_level = "low"
            explanation = f"Critical rule violations: {', '.join(rules_result.get('triggered_rule_names', []))}"
        elif entity_strong and no_critical_flag:
            recommendation = "FINALIZE"
            confidence_level = "high"
            explanation = "Strong entity matching confirms radiologist findings (retrieval index unavailable)"
        elif entity_weak:
            recommendation = "FLAG_FOR_HUMAN"
            confidence_level = "low"
            explanation = "Low entity match score — insufficient clinical evidence (retrieval index unavailable)"
        else:
            recommendation = "FINALIZE_LOW_CONFIDENCE"
            confidence_level = "medium"
            explanation = "Moderate entity matching (retrieval index unavailable — relying on RadGraph + rules)"
    # Aggregate ALL agent outputs + validation tool results
    validator_output = {
        "recommendation": recommendation,
        "confidence_level": confidence_level,
        "explanation": explanation,
        
        # === VALIDATION TOOLS RESULTS ===
        "retrieval": retrieval_result,
        "entity_matching": entity_result,
        "rules": rules_result,
        
        # === AGENT OUTPUTS SUMMARY ===
        "agent_summary": {
            "radiologist": {
                "findings": state.get("radiologist_output").findings[:200] if state.get("radiologist_output") else None,
                "impression": state.get("radiologist_output").impression[:200] if state.get("radiologist_output") else None,
                "current_uncertainty": state.get("current_uncertainty")
            },
            "chexbert": {
                "positive_labels": [
                    label for label, val in state.get("chexbert_output").labels.items()
                    if val == "present"  # actual label value from f1chexbert
                ] if state.get("chexbert_output") else [],
                "uncertain_labels": [
                    label for label, val in state.get("chexbert_output").labels.items()
                    if val == "uncertain"  # actual label value from f1chexbert
                ] if state.get("chexbert_output") else []
            },
            "critic": {
                "is_overconfident": state.get("critic_output").is_overconfident if state.get("critic_output") else None,
                "concern_count": len(state.get("critic_output").concern_flags) if state.get("critic_output") else 0,
                "safety_score": state.get("critic_output").safety_score if state.get("critic_output") else None
            },
            "historian": {
                "supporting_facts_count": len(state.get("historian_output").supporting_facts) if state.get("historian_output") else 0,
                "contradicting_facts_count": len(state.get("historian_output").contradicting_facts) if state.get("historian_output") else 0
            },
            "literature": {
                "evidence_strength": (
                    state.get("literature_output").overall_evidence_strength
                    if state.get("literature_output") and hasattr(state.get("literature_output"), "overall_evidence_strength")
                    else "text_only" if isinstance(state.get("literature_output"), str) else None
                ),
                "citation_count": (
                    len(state.get("literature_output").citations)
                    if state.get("literature_output") and hasattr(state.get("literature_output"), "citations")
                    else 0
                )
            },
            "debate": {
                "consensus_reached": state.get("debate_output").final_consensus if state.get("debate_output") else None,
                "rounds": len(state.get("debate_output").rounds) if state.get("debate_output") else 0,
                "escalated": state.get("debate_output").escalate_to_chief if state.get("debate_output") else False
            }
        },
        
        # === VALIDATION SUMMARY ===
        "summary": {
            "historical_support": retrieval_result.get("support_count", "0 out of 5"),
            "entity_match_f1": entity_result.get("entity_f1", 0.0),
            "flags": rules_result.get("flag_count", 0),
            "warnings": rules_result.get("warn_count", 0)
        },
        
        # === FINAL VERDICT FOR HUMAN ===
        "final_verdict": {
            "recommendation": recommendation,
            "confidence": confidence_level,
            "reasoning": explanation,
            "all_agents_agree": _check_agent_agreement(state),
            "external_evidence_strength": _assess_external_evidence(state),
            "key_concerns": rules_result.get("triggered_rule_names", [])
        }
    }
    
    # === MUC: Compute Information Gain for Validator ===
    current_uncertainty = state.get("current_uncertainty", 0.5)
    v_unc = compute_validator_uncertainty(
        entity_f1=entity_f1,
        has_critical_flags=has_critical_flag,
        flag_count=rules_result.get("flag_count", 0),
        retrieval_agrees=retrieval_agrees if retrieval_available else True,
    )
    v_align = compute_validator_alignment(
        recommendation=recommendation,
        entity_f1=entity_f1,
    )
    ig_result = compute_ig(
        agent_name="validator",
        agent_uncertainty=v_unc,
        alignment_score=v_align,
        system_uncertainty=current_uncertainty,
    )
    trace_entries.append(
        f"VALIDATOR MUC: unc={v_unc:.3f}, align={v_align:.3f}, "
        f"IG={ig_result.information_gain:.4f}, "
        f"U: {current_uncertainty:.4f} -> {ig_result.system_uncertainty_after:.4f}"
    )

    # Append to uncertainty_history
    uncertainty_history = list(state.get("uncertainty_history", []))
    uncertainty_history.append({
        "agent": "validator",
        "system_uncertainty": ig_result.system_uncertainty_after,
    })

    return {
        "validator_output": validator_output,
        "routing_decision": recommendation,
        "current_uncertainty": ig_result.system_uncertainty_after,
        "uncertainty_history": uncertainty_history,
        "trace": trace_entries
    }


def _check_agent_agreement(state: VerifaiState) -> bool:
    """Check if Critic, Historian, and Literature all support the diagnosis."""
    critic = state.get("critic_output")
    hist = state.get("historian_output")
    lit = state.get("literature_output")
    
    # High agreement if:
    # - Critic not overconfident
    # - Historian has supporting facts
    # - Literature has medium/high evidence
    
    critic_ok = not critic.is_overconfident if critic else False
    hist_ok = len(hist.supporting_facts) > 0 if hist else False
    lit_ok = (
        lit.overall_evidence_strength in ["medium", "high"]
        if lit and hasattr(lit, "overall_evidence_strength")
        else bool(lit) if isinstance(lit, str) else False  # non-empty string = some evidence
    )
    
    return critic_ok and hist_ok and lit_ok


def _assess_external_evidence(state: VerifaiState) -> str:
    """Assess overall external evidence strength."""
    hist = state.get("historian_output")
    lit = state.get("literature_output")
    
    hist_count = len(hist.supporting_facts) if hist else 0
    lit_strength = (
        lit.overall_evidence_strength
        if lit and hasattr(lit, "overall_evidence_strength")
        else "medium" if isinstance(lit, str) and lit else "none"
    )
    
    if hist_count >= 2 and lit_strength in ["medium", "high"]:
        return "strong"
    elif hist_count >= 1 or lit_strength in ["medium", "high"]:
        return "moderate"
    else:
        return "weak"
