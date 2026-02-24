"""
VERIFAI Agent Logger - Supabase Edition

Cloud-based structured logging for all agent invocations using Supabase (PostgreSQL).
Replaces SQLite with scalable cloud storage for multi-user/production deployments.

Usage is identical to the original SQLite logger:
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

from db.supabase_connection import get_db, init_db, prepare_insert_data


class AgentLogger:
    """
    Centralized logger for all VERIFAI agent interactions using Supabase.
    
    Creates a workflow session and logs each agent's invocation,
    outputs, and trace entries into structured PostgreSQL tables.
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
        
        # Normalize image_path to string if it's a list (for DB compat)
        if isinstance(image_paths, list):
            self.image_path = ", ".join(image_paths)
        else:
            self.image_path = image_paths or ""
            
        self.patient_id = patient_id
        self.workflow_type = workflow_type
        self._agent_count = 0

        # Create session row in Supabase
        try:
            with get_db() as db:
                session_data = prepare_insert_data({
                    'session_id': self.session_id,
                    'image_path': self.image_path,
                    'patient_id': patient_id,
                    'workflow_type': workflow_type,
                    'status': 'running',
                    'started_at': datetime.utcnow().isoformat()
                })
                # Use upsert to avoid Unique Violation if called multiple times in one session
                db.table('workflow_sessions').upsert(session_data).execute()
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to initialize session in Supabase: {e}")

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _start_invocation(self, db, agent_name: str, input_summary: dict = None, is_feedback_iteration: bool = False) -> int:
        """Record the start of an agent invocation. Returns invocation_id."""
        try:
            data = prepare_insert_data({
                'session_id': self.session_id,
                'agent_name': agent_name,
                'started_at': datetime.utcnow().isoformat(),
                'status': 'running',
                'input_summary': input_summary,
                'is_feedback_iteration': is_feedback_iteration
            })
            
            result = db.table('agent_invocations').insert(data).execute()
            self._agent_count += 1
            return result.data[0]['invocation_id']
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to start invocation for {agent_name}: {e}")
            return -1

    def _complete_invocation(self, db, invocation_id: int, status: str,
                              output_summary: Any = None, trace_entries: list = None,
                              error_message: str = None, started_at: float = None):
        """Mark an agent invocation as complete."""
        if invocation_id == -1:
            return
            
        try:
            duration_ms = int((time.time() - started_at) * 1000) if started_at else None
            
            update_data = prepare_insert_data({
                'completed_at': datetime.utcnow().isoformat(),
                'duration_ms': duration_ms,
                'status': status,
                'output_summary': output_summary,
                'trace_entries': trace_entries or [],
                'error_message': error_message
            })
            
            db.table('agent_invocations').update(update_data).eq('invocation_id', invocation_id).execute()
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to complete invocation {invocation_id}: {e}")

    def _log_trace_entries(self, db, agent_name: str, entries: list):
        """Insert trace entries into the flat trace_log table."""
        if not entries:
            return
            
        try:
            trace_entries = []
            for entry in entries:
                trace_entries.append(prepare_insert_data({
                    'session_id': self.session_id,
                    'agent_name': agent_name,
                    'entry': entry,
                    'created_at': datetime.utcnow().isoformat()
                }))
            
            db.table('trace_log').insert(trace_entries).execute()
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to log trace entries for {agent_name}: {e}")

    def _safe_json(self, obj) -> Any:
        """Safely serialize an object to JSON-compatible format."""
        if obj is None:
            return None
        
        try:
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                return obj.dict()
            else:
                return obj
        except Exception:
            return str(obj)

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
        # Read from MUC state key; fallback to legacy alias for compat
        kle_uncertainty = result.get("current_uncertainty", result.get("radiologist_kle_uncertainty"))
        trace = result.get("trace", [])

        with get_db() as db:
            inv_id = self._start_invocation(
                db, "radiologist",
                input_summary={'image_paths': state.get("image_paths", [])}
            )

            try:
                if rad_output:
                    from app.config import settings
                    num_samples = getattr(settings, 'KLE_NUM_SAMPLES', 5)
                    
                    # Ensure image_path from state is also normalized
                    image_paths_state = state.get("image_paths", [])
                    if isinstance(image_paths_state, list):
                        image_paths_state = ", ".join(image_paths_state)

                    rad_log_data = prepare_insert_data({
                        'session_id': self.session_id,
                        'invocation_id': inv_id,
                        'image_path': image_paths_state,
                        'findings_text': rad_output.findings,
                        'impression_text': rad_output.impression,
                        'kle_uncertainty': kle_uncertainty,
                        'num_samples': num_samples
                    })
                    db.table('radiologist_logs').insert(rad_log_data).execute()

                self._complete_invocation(db, inv_id, "success",
                                           output_summary=self._safe_json(rad_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(db, "radiologist", trace)

            except Exception as e:
                print(f"[DB LOG ERROR] Failed to log radiologist: {e}")
                try:
                    self._complete_invocation(db, inv_id, "error", error_message=str(e), started_at=t0)
                except Exception:
                    pass

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

        with get_db() as db:
            inv_id = self._start_invocation(db, "critic")

            try:
                if critic_output:
                    kle_input = state.get("current_uncertainty", state.get("radiologist_kle_uncertainty"))
                    critic_log_data = prepare_insert_data({
                        'session_id': self.session_id,
                        'invocation_id': inv_id,
                        'is_overconfident': critic_output.is_overconfident,
                        'safety_score': critic_output.safety_score,
                        'concern_flags': critic_output.concern_flags,
                        'recommended_hedging': critic_output.recommended_hedging,
                        'kle_uncertainty_input': kle_input,
                        'similar_mistakes_count': getattr(critic_output, 'similar_mistakes_count', 0),
                        'historical_risk_level': getattr(critic_output, 'historical_risk_level', 'none')
                    })
                    db.table('critic_logs').insert(critic_log_data).execute()

                self._complete_invocation(db, inv_id, "success",
                                           output_summary=self._safe_json(critic_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(db, "critic", trace)

            except Exception as e:
                print(f"[DB LOG ERROR] Failed to log critic: {e}")
                try:
                    self._complete_invocation(db, inv_id, "error", error_message=str(e), started_at=t0)
                except Exception:
                    pass

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

        with get_db() as db:
            inv_id = self._start_invocation(
                db, "historian",
                input_summary={'patient_id': state.get("patient_id", "")}
            )

            try:
                if hist_output:
                    hist_log_data = prepare_insert_data({
                        'session_id': self.session_id,
                        'invocation_id': inv_id,
                        'patient_id': state.get("patient_id"),
                        'confidence_adjustment': hist_output.confidence_adjustment,
                        'clinical_summary': hist_output.clinical_summary,
                        'num_supporting': len(hist_output.supporting_facts),
                        'num_contradicting': len(hist_output.contradicting_facts)
                    })
                    result_obj = db.table('historian_logs').insert(hist_log_data).execute()
                    hist_log_id = result_obj.data[0]['log_id']

                    # Supporting and contradicting facts
                    all_facts = []
                    for fact in hist_output.supporting_facts + hist_output.contradicting_facts:
                        fact_data = prepare_insert_data({
                            'historian_log_id': hist_log_id,
                            'session_id': self.session_id,
                            'fact_type': fact.fact_type,
                            'description': fact.description,
                            'fhir_resource_id': fact.fhir_resource_id,
                            'fhir_resource_type': fact.fhir_resource_type
                        })
                        all_facts.append(fact_data)
                    
                    if all_facts:
                        db.table('historian_facts').insert(all_facts).execute()

                self._complete_invocation(db, inv_id, "success",
                                           output_summary=self._safe_json(hist_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(db, "historian", trace)

            except Exception as e:
                print(f"[DB LOG ERROR] Failed to log historian: {e}")
                try:
                    self._complete_invocation(db, inv_id, "error", error_message=str(e), started_at=t0)
                except Exception:
                    pass

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

        with get_db() as db:
            inv_id = self._start_invocation(db, "literature")

            try:
                if lit_output:
                    # Handle string output (fast parallel mode)
                    if isinstance(lit_output, str):
                        lit_log_data = prepare_insert_data({
                            'session_id': self.session_id,
                            'invocation_id': inv_id,
                            'overall_evidence_strength': 'unknown',
                            'num_citations': 0,
                            'raw_summary': lit_output
                        })
                        db.table('literature_logs').insert(lit_log_data).execute()
                    else:
                        # Structured LiteratureOutput
                        lit_log_data = prepare_insert_data({
                            'session_id': self.session_id,
                            'invocation_id': inv_id,
                            'overall_evidence_strength': getattr(lit_output, 'overall_evidence_strength', 'low'),
                            'num_citations': len(getattr(lit_output, 'citations', []))
                        })
                        result_obj = db.table('literature_logs').insert(lit_log_data).execute()
                        lit_log_id = result_obj.data[0]['log_id']

                        # Citations
                        citations = []
                        for citation in getattr(lit_output, 'citations', []):
                            citation_data = prepare_insert_data({
                                'literature_log_id': lit_log_id,
                                'session_id': self.session_id,
                                'pmid': citation.pmid,
                                'title': citation.title,
                                'authors': citation.authors,
                                'journal': citation.journal,
                                'year': citation.year,
                                'relevance_summary': citation.relevance_summary,
                                'evidence_strength': citation.evidence_strength,
                                'source': citation.source
                            })
                            citations.append(citation_data)
                        
                        if citations:
                            db.table('literature_citations').insert(citations).execute()

                self._complete_invocation(db, inv_id, "success",
                                           output_summary=self._safe_json(lit_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(db, "literature", trace)

            except Exception as e:
                print(f"[DB LOG ERROR] Failed to log literature: {e}")
                try:
                    self._complete_invocation(db, inv_id, "error", error_message=str(e), started_at=t0)
                except Exception:
                    pass

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

        with get_db() as db:
            inv_id = self._start_invocation(db, "debate")

            try:
                if debate_output:
                    # Main debate log
                    debate_log_data = prepare_insert_data({
                        'session_id': self.session_id,
                        'invocation_id': inv_id,
                        'num_rounds': len(debate_output.rounds),
                        'final_consensus': debate_output.final_consensus,
                        'consensus_diagnosis': debate_output.consensus_diagnosis,
                        'consensus_confidence': debate_output.consensus_confidence,
                        'escalate_to_chief': debate_output.escalate_to_chief,
                        'escalation_reason': debate_output.escalation_reason,
                        'debate_summary': debate_output.debate_summary,
                        'total_confidence_adj': debate_output.total_confidence_adjustment
                    })
                    result_obj = db.table('debate_logs').insert(debate_log_data).execute()
                    debate_log_id = result_obj.data[0]['log_id']

                    # Each debate round
                    for rnd in debate_output.rounds:
                        round_data = prepare_insert_data({
                            'debate_log_id': debate_log_id,
                            'session_id': self.session_id,
                            'round_number': rnd.round_number,
                            'round_consensus': rnd.round_consensus,
                            'confidence_delta': rnd.confidence_delta
                        })
                        round_result = db.table('debate_rounds').insert(round_data).execute()
                        round_id = round_result.data[0]['round_id']

                        # Collect all arguments for this round
                        arguments = []
                        if rnd.critic_challenge:
                            arguments.append(self._prepare_debate_argument(round_id, debate_log_id, rnd.critic_challenge))
                        if rnd.historian_response:
                            arguments.append(self._prepare_debate_argument(round_id, debate_log_id, rnd.historian_response))
                        if rnd.literature_response:
                            arguments.append(self._prepare_debate_argument(round_id, debate_log_id, rnd.literature_response))
                        
                        if arguments:
                            db.table('debate_arguments').insert(arguments).execute()

                self._complete_invocation(db, inv_id, "success",
                                           output_summary=self._safe_json(debate_output),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(db, "debate", trace)

            except Exception as e:
                print(f"[DB LOG ERROR] Failed to log debate: {e}")
                try:
                    self._complete_invocation(db, inv_id, "error", error_message=str(e), started_at=t0)
                except Exception:
                    pass

    def _prepare_debate_argument(self, round_id: int, debate_log_id: int, arg) -> dict:
        """Prepare a debate argument for insertion."""
        return prepare_insert_data({
            'round_id': round_id,
            'debate_log_id': debate_log_id,
            'session_id': self.session_id,
            'agent': arg.agent,
            'position': arg.position,
            'argument': arg.argument,
            'confidence_impact': arg.confidence_impact,
            'evidence_refs': arg.evidence_refs
        })

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

        with get_db() as db:
            inv_id = self._start_invocation(db, "chief")

            try:
                if final_dx:
                    chief_log_data = prepare_insert_data({
                        'session_id': self.session_id,
                        'invocation_id': inv_id,
                        'diagnosis': final_dx.diagnosis,
                        'calibrated_confidence': final_dx.calibrated_confidence,
                        'was_deferred': final_dx.deferred,
                        'deferral_reason': final_dx.deferral_reason,
                        'explanation': final_dx.explanation,
                        'recommended_next_steps': final_dx.recommended_next_steps
                    })
                    db.table('chief_logs').insert(chief_log_data).execute()

                self._complete_invocation(db, inv_id, "success",
                                           output_summary=self._safe_json(final_dx),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(db, "chief", trace)

            except Exception as e:
                print(f"[DB LOG ERROR] Failed to log chief: {e}")
                try:
                    self._complete_invocation(db, inv_id, "error", error_message=str(e), started_at=t0)
                except Exception:
                    pass

    # =========================================================================
    # FINALIZE LOGGING
    # =========================================================================

    def log_finalize(self, state: dict, result: dict):
        """Log the finalize node output (same structure as chief but via consensus)."""
        t0 = time.time()
        final_dx = result.get("final_diagnosis")
        trace = result.get("trace", [])

        with get_db() as db:
            inv_id = self._start_invocation(db, "finalize")

            try:
                if final_dx:
                    finalize_log_data = prepare_insert_data({
                        'session_id': self.session_id,
                        'invocation_id': inv_id,
                        'diagnosis': final_dx.diagnosis,
                        'calibrated_confidence': final_dx.calibrated_confidence,
                        'was_deferred': final_dx.deferred,
                        'deferral_reason': final_dx.deferral_reason,
                        'explanation': final_dx.explanation,
                        'recommended_next_steps': final_dx.recommended_next_steps
                    })
                    db.table('chief_logs').insert(finalize_log_data).execute()

                self._complete_invocation(db, inv_id, "success",
                                           output_summary=self._safe_json(final_dx),
                                           trace_entries=trace, started_at=t0)
                self._log_trace_entries(db, "finalize", trace)

            except Exception as e:
                print(f"[DB LOG ERROR] Failed to log finalize: {e}")
                try:
                    self._complete_invocation(db, inv_id, "error", error_message=str(e), started_at=t0)
                except Exception:
                    pass

    # =========================================================================
    # EVIDENCE GATHERING (parallel historian + literature)
    # =========================================================================

    def log_evidence_gathering(self, state: dict, result: dict):
        """Log the evidence gathering node (combined historian + literature)."""
        trace = result.get("trace", [])

        try:
            with get_db() as db:
                inv_id = self._start_invocation(db, "evidence_gathering")
                self._complete_invocation(db, inv_id, "success", trace_entries=trace, started_at=time.time())
                self._log_trace_entries(db, "evidence_gathering", trace)
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to log evidence gathering base: {e}")

        # Also log individual agents if outputs are present
        try:
            if result.get("historian_output"):
                self.log_historian(state, result)
            if result.get("literature_output"):
                self.log_literature(state, result)
        except Exception as e:
             print(f"[DB LOG ERROR] Failed to log evidence sub-agents: {e}")

    # =========================================================================
    # NEW: DOCTOR FEEDBACK LOGGING
    # =========================================================================

    def log_doctor_feedback(
        self,
        feedback_type: str,
        doctor_notes: str,
        correct_diagnosis: str = None,
        rejection_reasons: list = None,
        context_snapshot: dict = None,
        doctor_id: str = None
    ) -> int:
        """
        Log doctor feedback when diagnosis is rejected/corrected.
        
        Args:
            feedback_type: 'rejection', 'correction', or 'approval'
            doctor_notes: Doctor's explanation
            correct_diagnosis: What doctor believes is correct (optional)
            rejection_reasons: List of reasons (e.g., ['wrong_pathology', 'missed_finding'])
            context_snapshot: Full workflow state at rejection point
            doctor_id: Optional identifier for the doctor
        
        Returns:
            feedback_id for tracking reprocessing
        """
        with get_db() as db:
            # Get current final diagnosis from session
            session_result = db.table('workflow_sessions').select('final_diagnosis, final_confidence').eq('session_id', self.session_id).execute()
            
            session_data = session_result.data[0] if session_result.data else {}
            
            feedback_data = prepare_insert_data({
                'session_id': self.session_id,
                'original_diagnosis': session_data.get('final_diagnosis'),
                'original_confidence': session_data.get('final_confidence'),
                'feedback_type': feedback_type,
                'doctor_notes': doctor_notes,
                'correct_diagnosis': correct_diagnosis,
                'rejection_reason': rejection_reasons or [],
                'context_snapshot': context_snapshot or {},
                'doctor_id': doctor_id,
                'reprocessed': False,
                'created_at': datetime.utcnow().isoformat()
            })
            
            result = db.table('doctor_feedback').insert(feedback_data).execute()
            feedback_id = result.data[0]['feedback_id']
            
            # Update session to mark it has feedback
            db.table('workflow_sessions').update({
                'has_feedback': True,
                'feedback_status': feedback_type,
                'feedback_count': db.rpc('increment_feedback_count', {'session': self.session_id})
            }).eq('session_id', self.session_id).execute()
            
            return feedback_id

    def link_feedback_reprocessing(self, feedback_id: int, new_session_id: str, final_result: str, final_confidence: float):
        """
        Link feedback to the new session created for reprocessing.
        
        Called after reprocessing workflow completes.
        """
        with get_db() as db:
            update_data = prepare_insert_data({
                'reprocessed': True,
                'reprocess_session_id': new_session_id,
                'reprocess_result': final_result,
                'reprocess_confidence': final_confidence
            })
            db.table('doctor_feedback').update(update_data).eq('feedback_id', feedback_id).execute()

    # =========================================================================
    # SESSION LIFECYCLE
    # =========================================================================

    def complete_session(self, final_diagnosis=None, error_message: str = None):
        """
        Mark the workflow session as completed.
        
        Call this at the end of the pipeline with the final diagnosis.
        """
        try:
            with get_db() as db:
                if final_diagnosis:
                    update_data = prepare_insert_data({
                        'status': 'completed',
                        'completed_at': datetime.utcnow().isoformat(),
                        'final_diagnosis': final_diagnosis.diagnosis,
                        'final_confidence': final_diagnosis.calibrated_confidence,
                        'was_deferred': final_diagnosis.deferred,
                        'deferral_reason': final_diagnosis.deferral_reason,
                        'total_agents_invoked': self._agent_count
                    })
                    db.table('workflow_sessions').update(update_data).eq('session_id', self.session_id).execute()
                elif error_message:
                    update_data = prepare_insert_data({
                        'status': 'failed',
                        'completed_at': datetime.utcnow().isoformat(),
                        'error_message': error_message,
                        'total_agents_invoked': self._agent_count
                    })
                    db.table('workflow_sessions').update(update_data).eq('session_id', self.session_id).execute()
                else:
                    update_data = prepare_insert_data({
                        'status': 'completed',
                        'completed_at': datetime.utcnow().isoformat(),
                        'total_agents_invoked': self._agent_count
                    })
                    db.table('workflow_sessions').update(update_data).eq('session_id', self.session_id).execute()
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to complete session: {e}")

    def fail_session(self, error_message: str):
        """Mark the workflow session as failed."""
        try:
            self.complete_session(error_message=error_message)
        except Exception as e:
            print(f"[DB LOG ERROR] Failed to fail session: {e}")

    # =========================================================================
    # QUERY HELPERS — for retrieving logs
    # =========================================================================

    @staticmethod
    def get_session_summary(session_id: str) -> dict:
        """Get a full summary of a workflow session."""
        with get_db() as db:
            # Get session
            session_result = db.table('workflow_sessions').select('*').eq('session_id', session_id).execute()
            if not session_result.data:
                return None
            
            # Get invocations
            invocations = db.table('agent_invocations').select('*').eq('session_id', session_id).order('started_at').execute()
            
            # Get traces
            traces = db.table('trace_log').select('*').eq('session_id', session_id).order('created_at').execute()
            
            # Get debate info
            debate_result = db.table('debate_logs').select('*').eq('session_id', session_id).execute()
            debate = debate_result.data[0] if debate_result.data else None
            
            debate_rounds = []
            if debate:
                rounds_result = db.table('debate_rounds').select('*').eq('debate_log_id', debate['log_id']).order('round_number').execute()
                for rnd in rounds_result.data:
                    args_result = db.table('debate_arguments').select('*').eq('round_id', rnd['round_id']).execute()
                    debate_rounds.append({
                        "round": rnd,
                        "arguments": args_result.data
                    })
            
            # Get feedback  if any
            feedback_result = db.table('doctor_feedback').select('*').eq('session_id', session_id).execute()
            
            return {
                "session": session_result.data[0],
                "invocations": invocations.data,
                "traces": traces.data,
                "debate": {
                    "summary": debate,
                    "rounds": debate_rounds
                },
                "feedback": feedback_result.data
            }

    @staticmethod
    def list_sessions(limit: int = 50, status: str = None, patient_id: str = None) -> list:
        """List workflow sessions with optional filters."""
        with get_db() as db:
            query = db.table('workflow_sessions').select('*')
            
            if status:
                query = query.eq('status', status)
            if patient_id:
                query = query.eq('patient_id', patient_id)
            
            query = query.order('started_at', desc=True).limit(limit)
            result = query.execute()
            
            return result.data

    @staticmethod
    def get_feedback_for_reprocessing(feedback_id: int) -> dict:
        """
        Get doctor feedback with full context for reprocessing.
        
        Returns:
            Dictionary with feedback details and workflow context snapshot
        """
        with get_db() as db:
            result = db.table('doctor_feedback').select('*').eq('feedback_id', feedback_id).execute()
            
            if not result.data:
                return None
            
            feedback = result.data[0]
            
            # Get full context from the original session
            context = AgentLogger.get_session_summary(feedback['session_id'])
            
            return {
                'feedback': feedback,
                'original_context': context
            }
