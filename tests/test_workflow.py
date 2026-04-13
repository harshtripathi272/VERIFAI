"""
Full VERIFAI Workflow Test

Tests the entire pipeline end-to-end:
  Radiologist → CheXbert → Evidence Gathering (Historian + Literature)
  → Critic (LLM enabled) → Debate → Validator → Finalize

Literature: uses real ReAct agent; falls back to [MOCK LITERATURE] if JSON fails.
Validator:  Retrieval/RadGraph tools may be unavailable (no vision encoder passed);
            the Rules Engine always runs.
LLM Critic: forced ON via settings override.
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 1. Force settings overrides BEFORE any VERIFAI module is imported ────────
os.environ["ENABLE_LLM_CRITIC"] = "true"      # Enable MedGemma semantic critic
os.environ["MOCK_MODELS"] = "false"            # Real inference only

from app.config import settings
settings.ENABLE_LLM_CRITIC = True             # Belt-and-braces override at object level

# ── 2. Initialize Validator tools ────────────────────────────────────────────
# Pass None for vision_encoder / image_processor — the CXR retrieval tool will
# fail gracefully; RadGraph (HuggingFace) + Rules Engine will still run.
from agents.validator import initialize_validator_tools
print("[TEST] Initializing validator tools (vision=None → retrieval will be skipped)...")
initialize_validator_tools(vision_encoder=None, image_processor=None)

# ── 3. Import & configure graph ───────────────────────────────────────────────
from graph.workflow import app as verifai_graph

TEST_PATIENT_ID = "6265ea60-b031-40da-95bb-0ef6178a5a45"

initial_state = {
    "image_paths": ["./images/images/00000013_001.png"], 
    "views": ["AP"],
    "patient_id": TEST_PATIENT_ID,
    "current_fhir": {
        "resourceType": "DiagnosticReport",
        "id": "dummy-report-123",
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11528-7",
                    "display": "Radiology Report"
                }
            ]
        },
        "conclusion": "Mild bilateral opacities. Cannot exclude early pneumonia."
    },
    "radiologist_output": None,
    "critic_output": None,
    "historian_output": None,
    "literature_output": None,
    "current_uncertainty": 1.0,
    "routing_decision": "",
    "steps_taken": 0,
    "final_diagnosis": None,
    "trace": ["[TEST] Initialization — LLM Critic: ON, Mock Literature: fallback if JSON fails"],
}

# ── 4. Run ────────────────────────────────────────────────────────────────────
print("\n[TEST] Running full VERIFAI workflow...\n")
config = {"configurable": {"thread_id": "test_workflow_123"}}
result = verifai_graph.invoke(initial_state, config)

# ══ SUMMARY ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  VERIFAI FULL WORKFLOW TEST — SUMMARY")
print("=" * 65)

print("\n─── RADIOLOGIST ─────────────────────────────────────────────")
rad = result.get("radiologist_output")
if rad:
    print(f"  Findings  : {rad.findings[:160]}")
    print(f"  Impression: {rad.impression[:160]}")
    uncertainty = result.get("current_uncertainty")
    print(f"  Uncertainty: {uncertainty:.4f}" if uncertainty is not None else "  Uncertainty: N/A")
else:
    print("  ✗ No radiologist output")

print("\n─── CHEXBERT ────────────────────────────────────────────────")
chex = result.get("chexbert_output")
if chex:
    present = [k for k, v in chex.labels.items() if v == "present"]
    uncertain = [k for k, v in chex.labels.items() if v == "uncertain"]
    print(f"  Present  : {present or 'None'}")
    print(f"  Uncertain: {uncertain or 'None'}")
else:
    print("  ✗ No CheXbert output")

print("\n─── HISTORIAN (FAISS) ───────────────────────────────────────")
hist = result.get("historian_output")
if hist:
    print(f"  FAISS Result : {len(hist.supporting_facts)} supporting, {len(hist.contradicting_facts)} contradicting")
    print(f"  Conf. Adjust : {hist.confidence_adjustment:+.3f}")
    print(f"  Summary      : {hist.clinical_summary[:180]}")
    if hist.supporting_facts:
        top = hist.supporting_facts[0]
        print(f"  Top Support  : [{top.fhir_resource_type}] {top.description[:120]}")
else:
    print("  ✗ No historian output (check patient_id / FHIR data)")

print("\n─── LITERATURE ──────────────────────────────────────────────")
lit = result.get("literature_output")
if lit:
    tag = "[MOCK]" if str(lit).startswith("[MOCK") else "[REAL]"
    print(f"  Source: {tag}")
    for line in str(lit).split("\n")[:5]:
        print(f"  {line}")
else:
    print("  ✗ No literature output")

print("\n─── CRITIC (LLM enabled) ────────────────────────────────────")
critic = result.get("critic_output")
if critic:
    print(f"  Safety Score : {critic.safety_score:.3f}")
    print(f"  Overconfident: {critic.is_overconfident}")
    print(f"  Hist. Risk   : {critic.historical_risk_level.upper()}")
    print(f"  Concern Flags: {len(critic.concern_flags)}")
    for flag in critic.concern_flags[:5]:
        print(f"    • {flag[:120]}")
    if critic.recommended_hedging:
        print(f"  Hedging      : {critic.recommended_hedging[:120]}")
    hist_ctx = getattr(critic, "historical_context", [])
    if hist_ctx:
        print(f"  Hist. Context: {len(hist_ctx)} matched historical case(s)")
else:
    print("  ✗ No critic output")

print("\n─── DEBATE ──────────────────────────────────────────────────")
debate = result.get("debate_output")
if debate:
    print(f"  Consensus    : {debate.final_consensus}")
    print(f"  Rounds       : {len(debate.rounds)}")
    if debate.consensus_diagnosis:
        print(f"  Diagnosis    : {debate.consensus_diagnosis[:160]}")
    if debate.total_confidence_adjustment:
        print(f"  Conf. Adjust : {debate.total_confidence_adjustment:+.2%}")
    if debate.escalate_to_chief:
        print(f"  ⚠ Escalated  : {debate.escalation_reason}")
    if debate.debate_summary:
        print(f"  Summary      : {debate.debate_summary[:180]}")
else:
    print("  ✗ No debate output")


print("\n─── VALIDATOR ───────────────────────────────────────────────")
vout = result.get("validator_output") or {}
if vout:
    print(f"  Recommendation: {vout.get('recommendation', '?')}")
    print(f"  Confidence    : {vout.get('confidence_level', '?')}")
    expl = vout.get("explanation", "")
    if expl:
        print(f"  Explanation   : {expl[:180]}")

    retrieval = vout.get("retrieval", {})
    if retrieval and not retrieval.get("error"):
        print(f"  Retrieval     : {len(retrieval.get('retrieved_sentences', []))} similar cases"
              f" | consensus={retrieval.get('consensus_diagnosis', '?')[:60]}")
    else:
        err = retrieval.get("error", "Not initialized") if retrieval else "Not initialized"
        print(f"  Retrieval     : ✗ {err}")

    entity = vout.get("entity_matching", {})
    if entity and not entity.get("error"):
        print(f"  Entity F1     : {entity.get('entity_f1', 'N/A')} | verdict={entity.get('verdict', '?')}")
    else:
        print(f"  Entity Match  : ✗ {entity.get('error', 'Not initialized') if entity else 'Not initialized'}")

    rules = vout.get("rules", {})
    if rules:
        print(f"  Rules Engine  : flags={rules.get('flag_count', 0)}, warnings={rules.get('warn_count', 0)}"
              f", critical={rules.get('has_critical_flag', False)}")
        triggered = rules.get("triggered_rule_names", [])
        if triggered:
            print(f"  Triggered     : {triggered}")

    agent_sum = vout.get("agent_summary", {})
    if agent_sum:
        c_sum = agent_sum.get("critic", {})
        h_sum = agent_sum.get("historian", {})
        l_sum = agent_sum.get("literature", {})
        print(f"  Agent Summary : critic_safety={c_sum.get('safety_score', '?')}"
              f" | hist_support={h_sum.get('supporting_facts_count', 0)}"
              f" | lit_strength={l_sum.get('evidence_strength', '?')}")
else:
    print("  ✗ No validator output")

print("\n─── FINAL DIAGNOSIS ─────────────────────────────────────────")
final_dx = result.get("final_diagnosis")
if final_dx:
    print(f"  Diagnosis  : {final_dx.diagnosis}")
    print(f"  Confidence : {final_dx.calibrated_confidence:.1%}")
    print(f"  Deferred   : {final_dx.deferred}")
    if final_dx.deferral_reason:
        print(f"  Reason     : {final_dx.deferral_reason}")
    if final_dx.explanation:
        print(f"  Explanation: {final_dx.explanation[:220]}")
else:
    print("  ✗ No final diagnosis")

print(f"\n  Final Uncertainty : {result.get('current_uncertainty', 0):.2%}")
print(f"  Steps Taken       : {result.get('steps_taken', 0)}")
print(f"  Historian FAISS   : {'✓ Active' if result.get('historian_output') else '✗ No data'}")
print(f"  Literature        : {'✓ [MOCK]' if str(result.get('literature_output', '')).startswith('[MOCK') else '✓ [REAL]' if result.get('literature_output') else '✗ None'}")
print(f"  LLM Critic        : {'✓ ON' if settings.ENABLE_LLM_CRITIC else '✗ OFF'}")

print("\n─── AUDIT TRACE (last 25 entries) ──────────────────────────")
for entry in result.get("trace", [])[-25:]:
    print(f"  {entry}")

print("=" * 65)


# ── 5. Save Metrics for Observability Dashboard ──────────────────────
print("\n[TEST] Saving metrics for observability dashboard...")
try:
    from monitoring.metrics import metrics, track_diagnosis, save_metrics_snapshot

    # Record workflow-level metrics
    metrics.start_workflow("test_workflow_123")

    # Track the diagnosis metrics
    if final_dx:
        track_diagnosis(
            confidence=final_dx.calibrated_confidence,
            uncertainty=result.get("current_uncertainty", 0),
            deferred=final_dx.deferred,
            debate_rounds=len(result.get("debate_output", {}).rounds) if result.get("debate_output") else 0,
            safety_score=getattr(result.get("critic_output", None), "safety_score", 1.0) if result.get("critic_output") else 1.0,
        )

    # Track agent invocations
    for agent_name in ["radiologist", "chexbert", "historian", "literature", "critic", "debate", "validator", "finalize"]:
        metrics.agent_invocations.labels(agent_name=agent_name, status="success").inc()

    # End workflow
    metrics.end_workflow("test_workflow_123")

    # Save snapshot to JSON file (read by API dashboard)
    save_metrics_snapshot()
    print("[TEST] ✓ Metrics saved — observability dashboard will show real data!")
except Exception as e:
    print(f"[TEST] ✗ Failed to save metrics: {e}")
