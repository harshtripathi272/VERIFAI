"""
Test VERIFAI Database Logging System

Verifies:
1. Schema creation with all tables and indexes
2. Logging for every agent (radiologist, critic, historian, literature, debate, chief)
3. Full debate round-by-round logging with arguments
4. Session lifecycle (create → log → complete)
5. Query helpers (session summary, agent history, debate history, stats)
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from db.connection import init_db, get_db, DB_PATH
from db.logger import AgentLogger
from graph.state import (
    RadiologistOutput,
    CriticOutput, HistorianOutput, HistorianFact,
    LiteratureOutput, LiteratureCitation,
    DebateOutput, DebateRound, DebateArgument,
    FinalDiagnosis
)


def test_schema_creation():
    """Test that all tables and indexes are created."""
    print("\n" + "=" * 60)
    print("TEST 1: Schema Creation")
    print("=" * 60)

    # Remove old test DB if exists
    test_db = os.path.join(os.path.dirname(__file__), "verifai_logs.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    init_db()

    with get_db() as conn:
        # Count tables
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t['name'] for t in tables]
        print(f"  Tables created: {len(table_names)}")
        for t in table_names:
            print(f"    ✓ {t}")

        # Count indexes
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        print(f"\n  Indexes created: {len(indexes)}")
        for idx in indexes:
            print(f"    ✓ {idx['name']}")

    assert len(table_names) >= 12, f"Expected 12+ tables, got {len(table_names)}"
    assert len(indexes) >= 30, f"Expected 30+ indexes, got {len(indexes)}"
    print("\n  ✅ Schema creation PASSED")


def test_full_workflow_logging():
    """Test logging for every agent in a complete workflow."""
    print("\n" + "=" * 60)
    print("TEST 2: Full Workflow Logging")
    print("=" * 60)

    # Create logger (auto-creates session)
    logger = AgentLogger(
        image_path="test_images/chest_xray_001.png",
        patient_id="patient-123",
        workflow_type="debate"
    )
    session_id = logger.session_id
    print(f"  Session created: {session_id}")

    # --- Mock state ---
    state = {
        "image_path": "test_images/chest_xray_001.png",
        "patient_id": "patient-123",
        "radiologist_kle_uncertainty": 0.42,
    }

    # 1. RADIOLOGIST (plain-text findings + impression + KLE)
    rad_output = RadiologistOutput(
        findings="There is a dense consolidation in the right lower lobe with air bronchograms. A subtle ground-glass opacity is noted in the left upper lobe.",
        impression="Findings are most consistent with community-acquired pneumonia involving the right lower lobe. The left upper lobe opacity may represent early involvement or an atypical infection."
    )
    rad_result = {
        "radiologist_output": rad_output,
        "radiologist_kle_uncertainty": 0.42,
        "trace": ["RADIOLOGIST: Generated report from 5 samples", "RADIOLOGIST KLE: Epistemic uncertainty=0.420 (from 5 samples)"]
    }
    logger.log_radiologist(state, rad_result)
    print("  ✓ Radiologist logged (plain-text + KLE)")

    # 2. CRITIC (KLE-based overconfidence check)
    critic_output = CriticOutput(
        is_overconfident=False,
        concern_flags=["Moderate epistemic uncertainty (KLE=0.42)", "Impression uses assertive language despite uncertainty"],
        recommended_hedging="Consider adding 'likely' or 'suggestive of' to the impression",
        safety_score=0.72
    )
    critic_result = {
        "critic_output": critic_output,
        "trace": ["CRITIC: Safety=72.00%, Overconfident=NO, KLE=0.420, Concerns=2 [Context: FHIR+Literature]"]
    }
    logger.log_critic(state, critic_result)
    print("  ✓ Critic logged (KLE-based)")

    # 3. HISTORIAN
    hist_output = HistorianOutput(
        supporting_facts=[
            HistorianFact(fact_type="supporting", description="[CAP] Recent fever and cough noted", fhir_resource_id="Condition/123", fhir_resource_type="Condition"),
            HistorianFact(fact_type="supporting", description="[CAP] WBC elevated at 15.2k", fhir_resource_id="Observation/456", fhir_resource_type="Observation"),
        ],
        contradicting_facts=[
            HistorianFact(fact_type="contradicting", description="[CAP] No prior history of pneumonia", fhir_resource_id="Condition/789", fhir_resource_type="Condition"),
        ],
        confidence_adjustment=0.08,
        clinical_summary="Clinical history supports infectious etiology."
    )
    hist_result = {"historian_output": hist_output, "trace": ["HISTORIAN: CAP Δconfidence=+0.08"]}
    logger.log_historian(state, hist_result)
    print("  ✓ Historian logged")

    # 4. LITERATURE
    lit_output = LiteratureOutput(
        citations=[
            LiteratureCitation(pmid="38901234", title="CAP Imaging Patterns in Adults", authors=["Smith J", "Lee K"], journal="Radiology", year=2024, relevance_summary="Reviews typical imaging findings of CAP", evidence_strength="high", source="pubmed"),
            LiteratureCitation(pmid="38765432", title="AI-Assisted Pneumonia Detection", authors=["Patel R"], journal="JAMA", year=2023, relevance_summary="AI performance in pneumonia detection", evidence_strength="medium", source="semanticscholar"),
        ],
        overall_evidence_strength="high"
    )
    lit_result = {"literature_output": lit_output, "trace": ["LITERATURE_AGENT: 2 citations found"]}
    logger.log_literature(state, lit_result)
    print("  ✓ Literature logged")

    # 5. DEBATE (full rounds)
    debate_output = DebateOutput(
        rounds=[
            DebateRound(
                round_number=1,
                critic_challenge=DebateArgument(agent="critic", position="challenge", argument="Moderate overconfidence detected. Consider Pulmonary Edema.", confidence_impact=-0.05, evidence_refs=["KLE=0.42"]),
                historian_response=DebateArgument(agent="historian", position="support", argument="Clinical history supports diagnosis: Recent fever and cough noted; WBC elevated at 15.2k", confidence_impact=0.10, evidence_refs=["Condition/123", "Observation/456"]),
                literature_response=DebateArgument(agent="literature", position="support", argument="Strong literature support: Smith et al. (2024). Reviews typical imaging findings of CAP", confidence_impact=0.12, evidence_refs=["38901234", "38765432"]),
                round_consensus="reached",
                confidence_delta=0.17
            ),
        ],
        final_consensus=True,
        consensus_diagnosis="Community-Acquired Pneumonia",
        consensus_confidence=0.88,
        escalate_to_chief=False,
        debate_summary="Consensus reached in round 1. Final confidence: 88.00%",
        total_confidence_adjustment=0.17
    )
    debate_result = {
        "debate_output": debate_output,
        "trace": [
            "DEBATE: 1 rounds completed",
            "DEBATE: Consensus=YES",
            "DEBATE: Confidence adjustment=+17.00%"
        ]
    }
    logger.log_debate(state, debate_result)
    print("  ✓ Debate logged (1 round, 3 arguments)")

    # 6. FINALIZE
    final_dx = FinalDiagnosis(
        diagnosis="Community-Acquired Pneumonia",
        calibrated_confidence=0.88,
        deferred=False,
        explanation="Consensus reached through 1-round debate.",
        recommended_next_steps=["Confirm with clinical correlation", "Consider follow-up imaging"]
    )
    finalize_result = {"final_diagnosis": final_dx, "trace": ["FINALIZE: Community-Acquired Pneumonia (confidence=88.00%)"]}
    logger.log_finalize(state, finalize_result)
    print("  ✓ Finalize logged")

    # Complete session
    logger.complete_session(final_diagnosis=final_dx)
    print("  ✓ Session completed")

    print("\n  ✅ Full workflow logging PASSED")
    return session_id


def test_query_helpers(session_id: str):
    """Test all query helper methods."""
    print("\n" + "=" * 60)
    print("TEST 3: Query Helpers")
    print("=" * 60)

    # 1. Session summary
    summary = AgentLogger.get_session_summary(session_id)
    assert summary is not None, "Session summary should not be None"
    assert summary["session"]["status"] == "completed"
    assert summary["session"]["final_diagnosis"] == "Community-Acquired Pneumonia"
    assert summary["session"]["final_confidence"] == 0.88
    assert len(summary["invocations"]) >= 5, f"Expected 5+ invocations, got {len(summary['invocations'])}"
    assert len(summary["traces"]) >= 5, f"Expected 5+ traces, got {len(summary['traces'])}"
    assert summary["debate"]["summary"] is not None
    assert len(summary["debate"]["rounds"]) == 1
    assert len(summary["debate"]["rounds"][0]["arguments"]) == 3
    print(f"  ✓ Session summary: {len(summary['invocations'])} invocations, {len(summary['traces'])} traces")
    print(f"    Debate: {len(summary['debate']['rounds'])} rounds, {len(summary['debate']['rounds'][0]['arguments'])} arguments")

    # 2. List sessions
    sessions = AgentLogger.list_sessions(limit=10)
    assert len(sessions) >= 1
    print(f"  ✓ List sessions: {len(sessions)} found")

    # 3. Agent history
    for agent in ["radiologist", "critic", "historian", "literature", "debate", "finalize"]:
        history = AgentLogger.get_agent_history(agent)
        assert len(history) >= 1, f"Expected history for {agent}"
        print(f"  ✓ Agent history [{agent}]: {len(history)} invocations")

    # 4. Debate history
    debates = AgentLogger.get_debate_history(session_id=session_id)
    assert len(debates) >= 1
    assert len(debates[0]["arguments"]) == 3
    print(f"  ✓ Debate history: {len(debates)} debates, {len(debates[0]['arguments'])} arguments")

    # 5. Diagnosis stats
    stats = AgentLogger.get_diagnosis_stats()
    assert stats["total_sessions"] >= 1
    assert stats["completed"] >= 1
    print(f"  ✓ Stats: {stats['total_sessions']} sessions, avg_confidence={stats['avg_confidence']}")
    print(f"    Top diagnoses: {[d['final_diagnosis'] for d in stats['top_diagnoses']]}")
    if stats.get('debate_consensus_rate') is not None:
        print(f"    Debate consensus rate: {stats['debate_consensus_rate']:.0%}")

    print("\n  ✅ Query helpers PASSED")


def test_detail_verification(session_id: str):
    """Verify individual table contents."""
    print("\n" + "=" * 60)
    print("TEST 4: Detail Verification")
    print("=" * 60)

    with get_db() as conn:
        # Radiologist logs (plain text + KLE)
        rad_logs = conn.execute(
            "SELECT * FROM radiologist_logs WHERE session_id=?", (session_id,)
        ).fetchall()
        assert len(rad_logs) == 1, f"Expected 1 radiologist log, got {len(rad_logs)}"
        rad = rad_logs[0]
        assert rad['findings_text'] is not None and len(rad['findings_text']) > 0
        assert rad['impression_text'] is not None and len(rad['impression_text']) > 0
        assert rad['kle_uncertainty'] is not None
        print(f"  ✓ Radiologist log: KLE={rad['kle_uncertainty']:.3f}, samples={rad['num_samples']}")
        print(f"    Findings preview: {rad['findings_text'][:80]}...")
        print(f"    Impression preview: {rad['impression_text'][:80]}...")

        # Critic (KLE-based)
        critic = conn.execute(
            "SELECT * FROM critic_logs WHERE session_id=?", (session_id,)
        ).fetchall()
        assert len(critic) == 1
        c = critic[0]
        print(f"  ✓ Critic: overconfident={'YES' if c['is_overconfident'] else 'NO'}, safety={c['safety_score']:.2f}")
        concern_flags = json.loads(c['concern_flags'])
        print(f"    Concern flags: {len(concern_flags)}")
        for flag in concern_flags:
            print(f"      - {flag}")
        if c['recommended_hedging']:
            print(f"    Recommended hedging: {c['recommended_hedging'][:80]}")

        # Historian facts
        facts = conn.execute(
            "SELECT * FROM historian_facts WHERE session_id=?", (session_id,)
        ).fetchall()
        assert len(facts) == 3, f"Expected 3 facts, got {len(facts)}"
        supporting = [f for f in facts if f['fact_type'] == 'supporting']
        contradicting = [f for f in facts if f['fact_type'] == 'contradicting']
        print(f"  ✓ Historian facts: {len(supporting)} supporting, {len(contradicting)} contradicting")

        # Literature citations
        citations = conn.execute(
            "SELECT * FROM literature_citations WHERE session_id=?", (session_id,)
        ).fetchall()
        assert len(citations) == 2
        print(f"  ✓ Literature citations: {len(citations)}")
        for c in citations:
            print(f"    - PMID:{c['pmid']} [{c['evidence_strength']}] {c['title'][:50]}")

        # Debate arguments
        args = conn.execute(
            "SELECT * FROM debate_arguments WHERE session_id=? ORDER BY argument_id", (session_id,)
        ).fetchall()
        assert len(args) == 3
        print(f"  ✓ Debate arguments: {len(args)}")
        for a in args:
            print(f"    - [{a['agent']}] {a['position']}: impact={a['confidence_impact']:+.2f}")

        # Trace log
        traces = conn.execute(
            "SELECT * FROM trace_log WHERE session_id=? ORDER BY created_at", (session_id,)
        ).fetchall()
        print(f"  ✓ Trace entries: {len(traces)}")
        for t in traces:
            print(f"    [{t['agent_name']}] {t['entry'][:80]}")

    print("\n  ✅ Detail verification PASSED")


def print_db_size():
    """Print the database file size."""
    if os.path.exists(DB_PATH):
        size_kb = os.path.getsize(DB_PATH) / 1024
        print(f"\n  📁 Database file: {DB_PATH}")
        print(f"  📊 Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    print("=" * 60)
    print("  VERIFAI Database Logging System — Full Test Suite")
    print("=" * 60)

    test_schema_creation()
    session_id = test_full_workflow_logging()
    test_query_helpers(session_id)
    test_detail_verification(session_id)
    print_db_size()

    print("\n" + "=" * 60)
    print("  🎉 ALL TESTS PASSED — Database logging system is working!")
    print("=" * 60)
