"""
VERIFAI Agent Logger

Structured logging for all agent invocations, debates, and workflow sessions.
Every agent call is recorded with full input/output details.

Usage:
    logger = AgentLogger(session_id="abc-123")
    logger.log_radiologist(state, result)
    logger.log_critic(state, result)
    ...
    logger.complete_session(final_diagnosis)
"""

import json
import uuid
import time
from datetime import datetime
from typing import Optional, Any

from db.connection import get_db, init_db


class AgentLogger:
    """
    Centralized logger for all VERIFAI agent interactions.
    
    Creates a workflow session and logs each agent's invocation,
    outputs, and trace entries into structured SQL tables.
    """

    def __init__(self, session_id: str = None, image_paths: list = None, views: list = None, patient_id: str = None, workflow_type: str = "debate"):
        """
        Initialize logger and create a workflow session.
        
        Args:
            session_id: Unique session ID. Auto-generated if not provided.
            image_paths: List of paths to the input X-ray images.
            views: List of corresponding views.
            patient_id: Optional FHIR patient ID.
            workflow_type: 'debate' or 'legacy'.
        """
        # Ensure DB is initialized (idempotent)
        init_db()

        self.session_id = session_id or str(uuid.uuid4())
        
        # Normalize image_path to string if it's a list
        if isinstance(image_paths, list):
            self.image_path = ", ".join(image_paths)
        else:
            self.image_path = image_paths or ""
            
        self.patient_id = patient_id
        self.workflow_type = workflow_type
        self._agent_count = 0

        # Create session row
        with get_db() as conn:
            conn.execute(
                """INSERT INTO workflow_sessions 
                   (session_id, image_path, patient_id, workflow_type, status, started_at)
                   VALUES (?, ?, ?, ?, 'running', ?)""",
                (self.session_id, self.image_path, patient_id, workflow_type, datetime.utcnow().isoformat())
            )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _start_invocation(self, conn, agent_name: str, input_summary: str = None) -> int:
        """Record the start of an agent invocation. Returns invocation_id."""
        try:
            cursor = conn.execute(
                """INSERT INTO agent_invocations 
                   (session_id, agent_name, started_at, status, input_summary)
                   VALUES (?, ?, ?, 'running', ?)""",
                (self.session_id, agent_name, datetime.utcnow().isoformat(), input_summary)
            )
            self._agent_count += 1
            return cursor.lastrowid
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to start invocation for {agent_name}: {e}")
            return -1

    def _complete_invocation(self, conn, invocation_id: int, status: str,
                              output_summary: str = None, trace_entries: list = None,
                              error_message: str = None, started_at: float = None):
        """Mark an agent invocation as complete."""
        if invocation_id == -1:
            return
            
        try:
            duration_ms = int((time.time() - started_at) * 1000) if started_at else None
            conn.execute(
                """UPDATE agent_invocations 
                   SET completed_at=?, duration_ms=?, status=?, output_summary=?, 
                       trace_entries=?, error_message=?
                   WHERE invocation_id=?""",
                (datetime.utcnow().isoformat(), duration_ms, status,
                 output_summary, json.dumps(trace_entries or []),
                 error_message, invocation_id)
            )
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to complete invocation {invocation_id}: {e}")

    def _log_trace_entries(self, conn, agent_name: str, entries: list):
        """Insert trace entries into the flat trace_log table."""
        try:
            for entry in (entries or []):
                conn.execute(
                    "INSERT INTO trace_log (session_id, agent_name, entry) VALUES (?, ?, ?)",
                    (self.session_id, agent_name, entry)
                )
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to log trace entries for {agent_name}: {e}")

    def _safe_json(self, obj) -> str:
        """Safely serialize an object to JSON string."""
        try:
            if hasattr(obj, 'model_dump'):
                return json.dumps(obj.model_dump(), default=str)
            elif hasattr(obj, 'dict'):
                return json.dumps(obj.dict(), default=str)
            else:
                return json.dumps(obj, default=str)
        except Exception:
            return json.dumps(str(obj))

    # =========================================================================
    # RADIOLOGIST LOGGING
    # =========================================================================

    def log_radiologist(self, state: dict, result: dict):
        """
        Log radiologist agent output.
        
        Stores: plain-text findings, impression, and MUC uncertainty score.
        """
        t0 = time.time()
        rad_output = result.get("radiologist_output")
        # Read from MUC state key; fallback to legacy key for compatibility
        kle_uncertainty = result.get("current_uncertainty", result.get("radiologist_kle_uncertainty"))
        trace = result.get("trace", [])

        with get_db() as conn:
            inv_id = self._start_invocation(
                conn, "radiologist",
                input_summary=json.dumps({"image_paths": state.get("image_paths", [])})
            )

            try:
                if rad_output:
                    from app.config import settings
                    num_samples = getattr(settings, 'KLE_NUM_SAMPLES', 5)
                    
                    # Ensure image_path from state is also normalized
                    image_paths_state = state.get("image_paths", [])
                    if isinstance(image_paths_state, list):
                        image_paths_state = ", ".join(image_paths_state)

                    conn.execute(
                        """INSERT INTO radiologist_logs 
                           (session_id, invocation_id, image_path, findings_text,
                            impression_text, kle_uncertainty, num_samples)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (self.session_id, inv_id, image_paths_state,
                         rad_output.findings, rad_output.impression,
                         kle_uncertainty, num_samples)
                    )

                self._complete_invocation(conn, inv_id, "success",
                                           output_summary=self._safe_json(rad_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(conn, "radiologist", trace)

            except Exception as e:
                self._complete_invocation(conn, inv_id, "error",
                                           error_message=str(e), started_at=t0)
                raise

    # =========================================================================
    # CRITIC LOGGING
    # =========================================================================

    def log_critic(self, state: dict, result: dict):
        """
        Log critic agent output.
        
        Stores: overconfidence flag, safety score, concern flags, recommended hedging.
        """
        t0 = time.time()
        critic_output = result.get("critic_output")
        trace = result.get("trace", [])

        with get_db() as conn:
            inv_id = self._start_invocation(conn, "critic")

            try:
                if critic_output:
                    kle_input = state.get("current_uncertainty", state.get("radiologist_kle_uncertainty"))
                    conn.execute(
                        """INSERT INTO critic_logs 
                           (session_id, invocation_id, is_overconfident,
                            safety_score, concern_flags, recommended_hedging,
                            kle_uncertainty_input)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (self.session_id, inv_id,
                         1 if critic_output.is_overconfident else 0,
                         critic_output.safety_score,
                         json.dumps(critic_output.concern_flags),
                         critic_output.recommended_hedging,
                         kle_input)
                    )

                self._complete_invocation(conn, inv_id, "success",
                                           output_summary=self._safe_json(critic_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(conn, "critic", trace)

            except Exception as e:
                self._complete_invocation(conn, inv_id, "error",
                                           error_message=str(e), started_at=t0)
                raise

    # =========================================================================
    # HISTORIAN LOGGING
    # =========================================================================

    def log_historian(self, state: dict, result: dict):
        """
        Log historian agent output.
        
        Stores: supporting/contradicting facts, confidence adjustment, FHIR resource references.
        """
        t0 = time.time()
        hist_output = result.get("historian_output")
        trace = result.get("trace", [])

        with get_db() as conn:
            inv_id = self._start_invocation(
                conn, "historian",
                input_summary=json.dumps({"patient_id": state.get("patient_id", "")})
            )

            try:
                if hist_output:
                    cursor = conn.execute(
                        """INSERT INTO historian_logs 
                           (session_id, invocation_id, patient_id, confidence_adjustment,
                            clinical_summary, num_supporting, num_contradicting)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (self.session_id, inv_id, state.get("patient_id"),
                         hist_output.confidence_adjustment, hist_output.clinical_summary,
                         len(hist_output.supporting_facts), len(hist_output.contradicting_facts))
                    )
                    hist_log_id = cursor.lastrowid

                    # Supporting facts
                    for fact in hist_output.supporting_facts:
                        conn.execute(
                            """INSERT INTO historian_facts 
                               (historian_log_id, session_id, fact_type, description,
                                fhir_resource_id, fhir_resource_type)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (hist_log_id, self.session_id, fact.fact_type,
                             fact.description, fact.fhir_resource_id, fact.fhir_resource_type)
                        )

                    # Contradicting facts
                    for fact in hist_output.contradicting_facts:
                        conn.execute(
                            """INSERT INTO historian_facts 
                               (historian_log_id, session_id, fact_type, description,
                                fhir_resource_id, fhir_resource_type)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (hist_log_id, self.session_id, fact.fact_type,
                             fact.description, fact.fhir_resource_id, fact.fhir_resource_type)
                        )

                self._complete_invocation(conn, inv_id, "success",
                                           output_summary=self._safe_json(hist_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(conn, "historian", trace)

            except Exception as e:
                self._complete_invocation(conn, inv_id, "error",
                                           error_message=str(e), started_at=t0)
                raise

    # =========================================================================
    # LITERATURE LOGGING
    # =========================================================================

    def log_literature(self, state: dict, result: dict):
        """
        Log literature agent output.
        
        Handles both structured LiteratureOutput and raw string summary (fast mode).
        Stores: citations with PMID, evidence strength, source.
        """
        t0 = time.time()
        lit_output = result.get("literature_output")
        trace = result.get("trace", [])

        with get_db() as conn:
            inv_id = self._start_invocation(conn, "literature")

            try:
                if lit_output:
                    # Handle string output (fast parallel mode)
                    if isinstance(lit_output, str):
                        conn.execute(
                            """INSERT INTO literature_logs 
                               (session_id, invocation_id, overall_evidence_strength,
                                num_citations, raw_summary)
                               VALUES (?, ?, ?, ?, ?)""",
                            (self.session_id, inv_id, "unknown", 0, lit_output)
                        )
                    else:
                        # Structured LiteratureOutput
                        cursor = conn.execute(
                            """INSERT INTO literature_logs 
                               (session_id, invocation_id, overall_evidence_strength,
                                num_citations)
                               VALUES (?, ?, ?, ?)""",
                            (self.session_id, inv_id,
                             getattr(lit_output, 'overall_evidence_strength', 'low'),
                             len(getattr(lit_output, 'citations', [])))
                        )
                        lit_log_id = cursor.lastrowid

                        # Citations
                        for citation in getattr(lit_output, 'citations', []):
                            conn.execute(
                                """INSERT INTO literature_citations 
                                   (literature_log_id, session_id, pmid, title, authors,
                                    journal, year, relevance_summary, evidence_strength, source)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (lit_log_id, self.session_id, citation.pmid, citation.title,
                                 json.dumps(citation.authors), citation.journal, citation.year,
                                 citation.relevance_summary, citation.evidence_strength, citation.source)
                            )

                self._complete_invocation(conn, inv_id, "success",
                                           output_summary=self._safe_json(lit_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(conn, "literature", trace)

            except Exception as e:
                self._complete_invocation(conn, inv_id, "error",
                                           error_message=str(e), started_at=t0)
                raise

    # =========================================================================
    # DEBATE LOGGING — Full round-by-round with arguments
    # =========================================================================

    def log_debate(self, state: dict, result: dict):
        """
        Log the full debate process.
        
        Stores: every round, every argument from critic/historian/literature,
        consensus status, confidence deltas, escalation decisions.
        """
        t0 = time.time()
        debate_output = result.get("debate_output")
        trace = result.get("trace", [])

        with get_db() as conn:
            inv_id = self._start_invocation(conn, "debate")

            try:
                if debate_output:
                    # Main debate log
                    cursor = conn.execute(
                        """INSERT INTO debate_logs 
                           (session_id, invocation_id, num_rounds, final_consensus,
                            consensus_diagnosis, consensus_confidence, escalate_to_chief,
                            escalation_reason, debate_summary, total_confidence_adj)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (self.session_id, inv_id, len(debate_output.rounds),
                         1 if debate_output.final_consensus else 0,
                         debate_output.consensus_diagnosis, debate_output.consensus_confidence,
                         1 if debate_output.escalate_to_chief else 0,
                         debate_output.escalation_reason, debate_output.debate_summary,
                         debate_output.total_confidence_adjustment)
                    )
                    debate_log_id = cursor.lastrowid

                    # Each debate round
                    for rnd in debate_output.rounds:
                        round_cursor = conn.execute(
                            """INSERT INTO debate_rounds 
                               (debate_log_id, session_id, round_number, round_consensus, confidence_delta)
                               VALUES (?, ?, ?, ?, ?)""",
                            (debate_log_id, self.session_id, rnd.round_number,
                             rnd.round_consensus, rnd.confidence_delta)
                        )
                        round_id = round_cursor.lastrowid

                        # Critic challenge
                        if rnd.critic_challenge:
                            self._insert_debate_argument(conn, round_id, debate_log_id, rnd.critic_challenge)

                        # Historian response
                        if rnd.historian_response:
                            self._insert_debate_argument(conn, round_id, debate_log_id, rnd.historian_response)

                        # Literature response
                        if rnd.literature_response:
                            self._insert_debate_argument(conn, round_id, debate_log_id, rnd.literature_response)

                self._complete_invocation(conn, inv_id, "success",
                                           output_summary=self._safe_json(debate_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(conn, "debate", trace)

            except Exception as e:
                self._complete_invocation(conn, inv_id, "error",
                                           error_message=str(e), started_at=t0)
                raise

    def _insert_debate_argument(self, conn, round_id: int, debate_log_id: int, arg):
        """Insert a single debate argument."""
        conn.execute(
            """INSERT INTO debate_arguments 
               (round_id, debate_log_id, session_id, agent, position,
                argument, confidence_impact, evidence_refs)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (round_id, debate_log_id, self.session_id, arg.agent, arg.position,
             arg.argument, arg.confidence_impact, json.dumps(arg.evidence_refs))
        )

    # =========================================================================
    # CHIEF LOGGING
    # =========================================================================

    def log_chief(self, state: dict, result: dict):
        """
        Log chief orchestrator output.
        
        Stores: final arbitration, deferral decisions, recommended next steps.
        """
        t0 = time.time()
        final_dx = result.get("final_diagnosis")
        trace = result.get("trace", [])

        with get_db() as conn:
            inv_id = self._start_invocation(conn, "chief")

            try:
                if final_dx:
                    conn.execute(
                        """INSERT INTO chief_logs 
                           (session_id, invocation_id, diagnosis, calibrated_confidence,
                            was_deferred, deferral_reason, explanation, recommended_next_steps)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (self.session_id, inv_id, final_dx.diagnosis,
                         final_dx.calibrated_confidence, 1 if final_dx.deferred else 0,
                         final_dx.deferral_reason, final_dx.explanation,
                         json.dumps(final_dx.recommended_next_steps))
                    )

                self._complete_invocation(conn, inv_id, "success",
                                           output_summary=self._safe_json(final_dx),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(conn, "chief", trace)

            except Exception as e:
                self._complete_invocation(conn, inv_id, "error",
                                           error_message=str(e), started_at=t0)
                raise

    # =========================================================================
    # FINALIZE LOGGING
    # =========================================================================

    def log_finalize(self, state: dict, result: dict):
        """Log the finalize node output (same structure as chief but via consensus)."""
        t0 = time.time()
        final_dx = result.get("final_diagnosis")
        trace = result.get("trace", [])
        # Persist the full uncertainty cascade for the audit trail
        uncertainty_cascade = state.get("uncertainty_cascade") or state.get("uncertainty_history", [])

        with get_db() as conn:
            inv_id = self._start_invocation(conn, "finalize")

            try:
                if final_dx:
                    conn.execute(
                        """INSERT INTO chief_logs 
                           (session_id, invocation_id, diagnosis, calibrated_confidence,
                            was_deferred, deferral_reason, explanation, recommended_next_steps)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (self.session_id, inv_id, final_dx.diagnosis,
                         final_dx.calibrated_confidence, 1 if final_dx.deferred else 0,
                         final_dx.deferral_reason, final_dx.explanation,
                         json.dumps(final_dx.recommended_next_steps))
                    )

                self._complete_invocation(conn, inv_id, "success",
                                           output_summary=self._safe_json(final_dx),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(conn, "finalize", trace)

                # Write uncertainty cascade to workflow_sessions immediately (non-critical)
                if uncertainty_cascade:
                    try:
                        conn.execute(
                            "UPDATE workflow_sessions SET uncertainty_cascade=? WHERE session_id=?",
                            (json.dumps(uncertainty_cascade, default=str), self.session_id)
                        )
                    except Exception as uc_err:
                        print(f"[DB LOG] Could not write uncertainty_cascade: {uc_err}")

            except Exception as e:
                self._complete_invocation(conn, inv_id, "error",
                                           error_message=str(e), started_at=t0)
                raise

    # =========================================================================
    # EVIDENCE GATHERING (parallel historian + literature)
    # =========================================================================

    def log_evidence_gathering(self, state: dict, result: dict):
        """Log the evidence gathering node (combined historian + literature)."""
        trace = result.get("trace", [])

        with get_db() as conn:
            inv_id = self._start_invocation(conn, "evidence_gathering")
            self._complete_invocation(conn, inv_id, "success", trace_entries=trace, started_at=time.time())
            self._log_trace_entries(conn, "evidence_gathering", trace)

        # Also log individual agents if outputs are present
        if result.get("historian_output"):
            self.log_historian(state, result)
        if result.get("literature_output"):
            self.log_literature(state, result)

    # =========================================================================
    # SESSION LIFECYCLE
    # =========================================================================

    def complete_session(self, final_diagnosis=None, error_message: str = None):
        """
        Mark the workflow session as completed.
        
        Call this at the end of the pipeline with the final diagnosis.
        """
        with get_db() as conn:
            if final_diagnosis:
                conn.execute(
                    """UPDATE workflow_sessions 
                       SET status='completed', completed_at=?, final_diagnosis=?,
                           final_confidence=?, was_deferred=?, deferral_reason=?,
                           total_agents_invoked=?
                       WHERE session_id=?""",
                    (datetime.utcnow().isoformat(), final_diagnosis.diagnosis,
                     final_diagnosis.calibrated_confidence,
                     1 if final_diagnosis.deferred else 0,
                     final_diagnosis.deferral_reason,
                     self._agent_count, self.session_id)
                )
            elif error_message:
                conn.execute(
                    """UPDATE workflow_sessions 
                       SET status='failed', completed_at=?, error_message=?,
                           total_agents_invoked=?
                       WHERE session_id=?""",
                    (datetime.utcnow().isoformat(), error_message,
                     self._agent_count, self.session_id)
                )
            else:
                conn.execute(
                    """UPDATE workflow_sessions 
                       SET status='completed', completed_at=?, total_agents_invoked=?
                       WHERE session_id=?""",
                    (datetime.utcnow().isoformat(), self._agent_count, self.session_id)
                )

    def fail_session(self, error_message: str):
        """Mark the workflow session as failed."""
        self.complete_session(error_message=error_message)

    # =========================================================================
    # QUERY HELPERS — for retrieving logs
    # =========================================================================

    @staticmethod
    def get_session_summary(session_id: str) -> dict:
        """Get a full summary of a workflow session."""
        with get_db() as conn:
            session = conn.execute(
                "SELECT * FROM workflow_sessions WHERE session_id=?", (session_id,)
            ).fetchone()

            if not session:
                return None

            invocations = conn.execute(
                "SELECT * FROM agent_invocations WHERE session_id=? ORDER BY started_at",
                (session_id,)
            ).fetchall()

            traces = conn.execute(
                "SELECT * FROM trace_log WHERE session_id=? ORDER BY created_at",
                (session_id,)
            ).fetchall()

            debate = conn.execute(
                "SELECT * FROM debate_logs WHERE session_id=?", (session_id,)
            ).fetchone()

            debate_rounds = []
            if debate:
                rounds = conn.execute(
                    "SELECT * FROM debate_rounds WHERE debate_log_id=? ORDER BY round_number",
                    (debate['log_id'],)
                ).fetchall()
                for rnd in rounds:
                    args = conn.execute(
                        "SELECT * FROM debate_arguments WHERE round_id=? ORDER BY argument_id",
                        (rnd['round_id'],)
                    ).fetchall()
                    debate_rounds.append({
                        "round": dict(rnd),
                        "arguments": [dict(a) for a in args]
                    })

            return {
                "session": dict(session),
                "invocations": [dict(i) for i in invocations],
                "traces": [dict(t) for t in traces],
                "debate": {
                    "summary": dict(debate) if debate else None,
                    "rounds": debate_rounds
                }
            }

    @staticmethod
    def list_sessions(limit: int = 50, status: str = None, patient_id: str = None) -> list:
        """List workflow sessions with optional filters."""
        with get_db() as conn:
            query = "SELECT * FROM workflow_sessions WHERE 1=1"
            params = []

            if status:
                query += " AND status=?"
                params.append(status)
            if patient_id:
                query += " AND patient_id=?"
                params.append(patient_id)

            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def get_agent_history(agent_name: str, limit: int = 100) -> list:
        """Get invocation history for a specific agent."""
        with get_db() as conn:
            rows = conn.execute(
                """SELECT ai.*, ws.image_path, ws.patient_id 
                   FROM agent_invocations ai
                   JOIN workflow_sessions ws ON ai.session_id = ws.session_id
                   WHERE ai.agent_name=?
                   ORDER BY ai.started_at DESC LIMIT ?""",
                (agent_name, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def get_debate_history(session_id: str = None, limit: int = 50) -> list:
        """Get debate logs with full round details."""
        with get_db() as conn:
            if session_id:
                debates = conn.execute(
                    "SELECT * FROM debate_logs WHERE session_id=?", (session_id,)
                ).fetchall()
            else:
                debates = conn.execute(
                    "SELECT * FROM debate_logs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()

            results = []
            for debate in debates:
                rounds = conn.execute(
                    """SELECT dr.*, 
                        (SELECT COUNT(*) FROM debate_arguments WHERE round_id=dr.round_id) as num_arguments
                       FROM debate_rounds dr
                       WHERE dr.debate_log_id=? ORDER BY dr.round_number""",
                    (debate['log_id'],)
                ).fetchall()

                all_args = conn.execute(
                    "SELECT * FROM debate_arguments WHERE debate_log_id=? ORDER BY round_id, argument_id",
                    (debate['log_id'],)
                ).fetchall()

                results.append({
                    "debate": dict(debate),
                    "rounds": [dict(r) for r in rounds],
                    "arguments": [dict(a) for a in all_args]
                })

            return results

    @staticmethod
    def get_diagnosis_stats() -> dict:
        """Get aggregate statistics about diagnoses."""
        with get_db() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM workflow_sessions").fetchone()['cnt']
            completed = conn.execute("SELECT COUNT(*) as cnt FROM workflow_sessions WHERE status='completed'").fetchone()['cnt']
            failed = conn.execute("SELECT COUNT(*) as cnt FROM workflow_sessions WHERE status='failed'").fetchone()['cnt']
            deferred = conn.execute("SELECT COUNT(*) as cnt FROM workflow_sessions WHERE was_deferred=1").fetchone()['cnt']

            avg_confidence = conn.execute(
                "SELECT AVG(final_confidence) as avg_conf FROM workflow_sessions WHERE final_confidence IS NOT NULL"
            ).fetchone()['avg_conf']

            top_diagnoses = conn.execute(
                """SELECT final_diagnosis, COUNT(*) as cnt, AVG(final_confidence) as avg_conf
                   FROM workflow_sessions 
                   WHERE final_diagnosis IS NOT NULL
                   GROUP BY final_diagnosis ORDER BY cnt DESC LIMIT 10"""
            ).fetchall()

            avg_agents = conn.execute(
                "SELECT AVG(total_agents_invoked) as avg_agents FROM workflow_sessions WHERE status='completed'"
            ).fetchone()['avg_agents']

            debate_consensus_rate = conn.execute(
                """SELECT 
                     COUNT(CASE WHEN final_consensus=1 THEN 1 END) as consensus,
                     COUNT(*) as total
                   FROM debate_logs"""
            ).fetchone()

            return {
                "total_sessions": total,
                "completed": completed,
                "failed": failed,
                "deferred": deferred,
                "avg_confidence": round(avg_confidence, 4) if avg_confidence else None,
                "avg_agents_per_session": round(avg_agents, 1) if avg_agents else None,
                "debate_consensus_rate": (
                    round(debate_consensus_rate['consensus'] / max(debate_consensus_rate['total'], 1), 3)
                    if debate_consensus_rate else None
                ),
                "top_diagnoses": [dict(d) for d in top_diagnoses]
            }
