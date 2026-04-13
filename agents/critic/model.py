"""
Critic Model

Overconfidence detector that evaluates consistency between linguistic certainty
in the radiology report and the externally computed KLE-based uncertainty score.

Integration points (marked with # >>> LLM-CRITIC):
  - Second-stage MedGemma semantic critic gated behind settings.ENABLE_LLM_CRITIC
  - Merges LLM output into rule-based results without overriding them
"""

import logging
import re

from app.config import settings
from .llm_critic import medgemma_critic  # >>> LLM-CRITIC

# Historical mistake memory — via repository abstraction (Supabase HNSW or DuckDB fallback)
try:
    from db.past_mistakes_repository import get_past_mistakes_repository
    from uncertainty.case_embedding import generate_case_summary, generate_case_embedding
    PAST_MISTAKES_AVAILABLE = True
except ImportError as e:
    PAST_MISTAKES_AVAILABLE = False

logger = logging.getLogger(__name__)


class CriticModel:
    """
    Linguistic certainty evaluator that detects radiologist overconfidence.
    
    Compares the assertiveness/certainty expressed in the IMPRESSION text
    against the epistemic uncertainty score (KLE) computed from semantic sampling.
    
    Does NOT use internal model signals (logits, entropy, attention).
    """
    
    def __init__(self):
        self.mock = settings.MOCK_MODELS
        # No model loading needed for rule-based linguistic analysis
    
    def _analyze_linguistic_certainty(self, impression: str) -> tuple[float, list[str]]:
        """
        Analyze the linguistic certainty level in the IMPRESSION text.
        
        Returns:
            Tuple of (certainty_score, certainty_markers)
            - certainty_score: 0.0 (very hedged) to 1.0 (very assertive)
            - certainty_markers: List of phrases that influenced the score
        """
        impression_lower = impression.lower()
        
        # High certainty phrases (increase score)
        high_certainty_patterns = [
            r'\bdefinite\b', r'\bdefinitely\b', r'\bcertain\b', r'\bcertainly\b',
            r'\bdiagnostic of\b', r'\bpathognomonic\b', r'\bconfirm\b',
            r'\bconsistent with\b(?! differential)', r'\bdemonstrates\b',
            r'\bshows\b(?! possible)', r'\bevidence of\b(?! possible)',
        ]
        
        # Low certainty phrases (decrease score)
        low_certainty_patterns = [
            r'\bpossib\w*\b', r'\blikely\b', r'\bunlikely\b', r'\bmay\b',
            r'\bcould\b', r'\bmight\b', r'\bsuggest\w*\b', r'\braise.{0,20}concern\b',
            r'\bcannot exclude\b', r'\bconsider\b', r'\bdifferential\b',
            r'\bvs\.?\b', r'\bversus\b', r'\bunclear\b', r'\bindeterminate\b',
            r'\brecommend.{0,30}correlation\b', r'\brecommend.{0,30}follow.?up\b'
        ]
        
        high_count = sum(len(re.findall(pattern, impression_lower)) for pattern in high_certainty_patterns)
        low_count = sum(len(re.findall(pattern, impression_lower)) for pattern in low_certainty_patterns)
        
        # Base score
        base_score = 0.5
        
        # Adjust based on markers
        certainty_score = base_score + (high_count * 0.15) - (low_count * 0.15)
        certainty_score = max(0.0, min(1.0, certainty_score))
        
        # Collect markers for explanation
        markers = []
        if high_count > 0:
            markers.append(f"Strong assertions ({high_count} occurrences)")
        if low_count > 0:
            markers.append(f"Hedging language ({low_count} occurrences)")
        
        return certainty_score, markers
    
    def evaluate(
        self,
        findings: str,
        impression: str,
        current_uncertainty: float,
        chexbert_output=None,
        historian_output=None,
        literature_output=None,
        doctor_feedback=None,
        uncertainty_history: list = None,
        ) -> tuple[bool, list[str], str | None, float]:

        """
        Evaluate whether the linguistic certainty is appropriate given the epistemic uncertainty
        AND enriched clinical context.
        
        Args:
            findings: FINDINGS text section
            impression: IMPRESSION text section
            current_uncertainty: MUC system uncertainty score (0.0-1.0, higher = more uncertain)
            historian_output: HistorianOutput with FHIR facts (optional)
            literature_output: LiteratureOutput with citations (optional) or string summary
            
        Returns:
            Tuple of (is_overconfident, concern_flags, recommended_hedging, safety_score)
        """

        # Stage 1: Rule-based linguistic analysis  (unchanged)
        

        # Analyze linguistic certainty in the impression
        linguistic_certainty, markers = self._analyze_linguistic_certainty(impression)
        
        # Detect overconfidence pattern:
        # High linguistic certainty BUT high epistemic uncertainty
        is_overconfident = False
        concern_flags = []
        recommended_hedging = None
        
        # Define thresholds
        HIGH_CERTAINTY_THRESHOLD = 0.65
        HIGH_UNCERTAINTY_THRESHOLD = 0.45
        
        if linguistic_certainty > HIGH_CERTAINTY_THRESHOLD and current_uncertainty > HIGH_UNCERTAINTY_THRESHOLD:
            is_overconfident = True
            concern_flags.append(
                f"High linguistic certainty ({linguistic_certainty:.2f}) despite high epistemic uncertainty ({current_uncertainty:.2f})"
            )
            concern_flags.extend(markers)
            recommended_hedging = (
                "Consider using more hedging language (e.g., 'suggestive of', 'most consistent with', "
                "'differential includes') to reflect the semantic instability observed across samples."
            )
        elif linguistic_certainty > 0.8 and current_uncertainty > 0.3:
            # Moderate concern
            concern_flags.append(
                f"Relatively assertive language ({linguistic_certainty:.2f}) with moderate uncertainty ({current_uncertainty:.2f})"
            )
            concern_flags.extend(markers)
        
        # Check for internal contradictions between findings and impression
        if "no abnormality" in findings.lower() and "consolidation" in impression.lower():
            concern_flags.append("Potential contradiction: findings vs impression")
        
        # Calculate safety score (inverse of risk)
        # Safety is high when:
        # - Low certainty + high uncertainty (appropriately cautious)
        # - High certainty + low uncertainty (appropriately confident)
        certainty_uncertainty_gap = abs(linguistic_certainty - (1.0 - current_uncertainty))
        safety_score = 1.0 - certainty_uncertainty_gap
        safety_score = max(0.0, min(1.0, safety_score))
        
        if is_overconfident:
            safety_score = min(safety_score, 0.5)  # Cap safety if overconfident

        
        # ── Stage 2: Deterministic Historian Challenge ──────────────────────────
        #
        # Always runs when historian_output is present, regardless of ENABLE_LLM_CRITIC.
        # Checks whether contradicting clinical facts are unaddressed in the impression.

        if historian_output:
            contradicting = (
                historian_output.contradicting_facts
                if hasattr(historian_output, "contradicting_facts") else []
            )
            supporting = (
                historian_output.supporting_facts
                if hasattr(historian_output, "supporting_facts") else []
            )

            if contradicting:
                # Build a brief description from top contradictions (max 3)
                top_desc = "; ".join(
                    f.description[:80] for f in contradicting[:3]
                )
                concern_flags.append(
                    f"[HISTORIAN CHALLENGE] {len(contradicting)} contradicting clinical "
                    f"fact(s) not addressed in impression. Top: {top_desc}"
                )
                # Proportional penalty: 5% per contradiction, capped at 20%
                hist_penalty = min(0.05 * len(contradicting), 0.20)
                safety_score = max(0.0, safety_score - hist_penalty)
                logger.info(
                    "[CRITIC] Historian challenge: %d contradictions, penalty=%.2f",
                    len(contradicting), hist_penalty,
                )

            if len(supporting) > 3 and linguistic_certainty < 0.4:
                # Strong clinical support exists but impression is overly cautious
                # (not a problem — deliberately NOT penalised)
                concern_flags.append(
                    f"[HISTORIAN NOTE] {len(supporting)} supporting clinical facts available; "
                    "impression is appropriately cautious."
                )

        # ── Stage 3: Deterministic Literature Challenge ──────────────────────────
        #
        # Always runs when literature_output is present.
        # Flags when published evidence suggests differentials the impression omits.

        if literature_output:
            impression_mentions_differentials = self._mentions_differentials(impression)

            if isinstance(literature_output, str):
                # String summary path
                lit_lower = literature_output.lower()
                if (
                    ("differential" in lit_lower or "alternative" in lit_lower)
                    and not impression_mentions_differentials
                ):
                    concern_flags.append(
                        "[LITERATURE CHALLENGE] Literature evidence suggests alternative "
                        "diagnoses; impression does not mention differentials."
                    )
                    safety_score = max(0.0, safety_score - 0.05)
                    logger.info("[CRITIC] Literature challenge (string): differentials omitted")
            else:
                # Structured LiteratureOutput path
                citations = (
                    literature_output.citations
                    if hasattr(literature_output, "citations") else []
                )
                if citations and not impression_mentions_differentials:
                    top_title = citations[0].title[:80] if citations else ""
                    concern_flags.append(
                        f"[LITERATURE CHALLENGE] {len(citations)} relevant study(ies) found; "
                        f"impression omits differentials. Top study: \"{top_title}\""
                    )
                    safety_score = max(0.0, safety_score - 0.08)
                    logger.info(
                        "[CRITIC] Literature challenge: %d citations, differentials omitted",
                        len(citations),
                    )


        # ── Stage 4: Historical Mistake Memory Retrieval ─────────────────────────
        # >>> PAST-MISTAKES
        
        similar_mistakes_count = 0
        historical_risk_level = "none"
        historical_context: list[dict] = []
        
        if settings.ENABLE_PAST_MISTAKES_MEMORY and PAST_MISTAKES_AVAILABLE:
            try:
                # Extract disease type from chexbert labels or impression
                chexbert_labels_dict = {}
                if chexbert_output and hasattr(chexbert_output, "labels"):
                    chexbert_labels_dict = chexbert_output.labels
                
                disease_type = self._extract_disease_type(impression, chexbert_labels_dict)
                logger.debug("[CRITIC] disease_type=%s", disease_type)

                # Generate current case summary and embedding
                current_summary = generate_case_summary(
                    disease_type=disease_type,
                    original_diagnosis=impression[:200],  # Use impression as proxy
                    corrected_diagnosis="[Current Case - Not Yet Validated]",
                    error_type="unknown",
                    uncertainty_score=current_uncertainty,
                    chexbert_labels=chexbert_labels_dict,
                    clinical_summary=historian_output.clinical_summary if historian_output else None,
                    debate_summary=None
                )
                current_embedding = generate_case_embedding(current_summary)
                
                # Retrieve historically similar mistakes via hybrid repository
                uncertainty_min = 0.0
                uncertainty_max = 1.0
                _repo = get_past_mistakes_repository()
                logger.info(
                    f"[CRITIC] Past-mistakes backend: {_repo.backend_name}"
                )

                try:
                    similar_mistakes = _repo.retrieve_similar_mistakes(
                        disease_type=disease_type,
                        embedding=current_embedding,
                        kle_uncertainty_range=(uncertainty_min, uncertainty_max),
                        severity_min=1,
                        top_k=settings.PAST_MISTAKES_TOP_K,
                        similarity_threshold=settings.PAST_MISTAKES_SIMILARITY_THRESHOLD,
                    )
                except Exception as _repo_err:
                    logger.warning(
                        f"[CRITIC] Past-mistakes retrieval via {_repo.backend_name} failed: "
                        f"{_repo_err}. Skipping historical context for this evaluation."
                    )
                    similar_mistakes = []
                
                # Apply neural re-ranking if enabled
                if similar_mistakes and getattr(settings, 'ENABLE_PAST_MISTAKES_RERANKING', False):
                    try:
                        from db.rerank_mistakes import rerank_mistakes
                        similar_mistakes = rerank_mistakes(
                            current_impression=impression,
                            current_kle=current_uncertainty,
                            current_chexbert=chexbert_labels_dict,
                            retrieved_mistakes=similar_mistakes,
                            use_medgemma=False,  # Fast mode (embeddings already used)
                            recency_weight_factor=0.3,
                            clinical_relevance_factor=0.5,
                            feedback_factor=0.2
                        )
                        logger.debug(f"[CRITIC] Applied neural re-ranking to {len(similar_mistakes)} cases")
                    except Exception as e:
                        logger.warning(f"[CRITIC] Re-ranking failed, using original order: {e}")
                
                similar_mistakes_count = len(similar_mistakes)
                
                # --- Build structured top-3 context list -------------------------
                historical_context: list[dict] = []
                for m in similar_mistakes[:3]:
                    historical_context.append({
                        "disease_type":    m.get("disease_type", ""),
                        "error_type":      m.get("error_type", ""),
                        "severity_level":  m.get("severity_level"),
                        "uncertainty_score": m.get("uncertainty_score", m.get("kle_uncertainty")),
                        "clinical_summary": (
                            (m.get("clinical_summary") or "")[:300]
                        ),
                        "similarity": round(float(m.get("similarity", 0.0)), 4),
                    })
                
                if similar_mistakes:
                    # Analyze severity of similar mistakes
                    high_severity_count = sum(1 for m in similar_mistakes if m['severity_level'] >= 4)
                    medium_severity_count = sum(1 for m in similar_mistakes if m['severity_level'] == 3)
                    
                    # Determine historical risk level
                    if high_severity_count >= 2:
                        historical_risk_level = "high"
                        is_overconfident = True
                        concern_flags.append(
                            f"[HISTORY] {high_severity_count} high-severity similar errors detected "
                            f"in {disease_type} with uncertainty={current_uncertainty:.2f}"
                        )
                        
                        # Show specific past error types
                        error_types_hist = [m['error_type'] for m in similar_mistakes[:3]]
                        concern_flags.append(
                            f"[HISTORY] Past error patterns: {', '.join(set(error_types_hist))}"
                        )
                        
                        # Increase risk weighting significantly
                        historical_penalty = 0.15 * high_severity_count
                        safety_score = max(0.0, safety_score - historical_penalty)
                        
                    elif high_severity_count >= 1:
                        historical_risk_level = "medium"
                        concern_flags.append(
                            f"[HISTORY] {high_severity_count} high-severity + {medium_severity_count} medium-severity "
                            f"similar cases in {disease_type}"
                        )
                        # Moderate penalty
                        safety_score = max(0.0, safety_score - 0.10)
                        
                    elif len(similar_mistakes) >= 3:
                        historical_risk_level = "low"
                        concern_flags.append(
                            f"[HISTORY] {len(similar_mistakes)} similar past cases detected in {disease_type}"
                        )
                        # Small penalty
                        safety_score = max(0.0, safety_score - 0.05)
                    
                    # --- Generate human-readable narrative paragraph -----------
                    n = len(similar_mistakes)
                    count_word = {1: "One", 2: "Two", 3: "Three"}.get(n, str(n))
                    primary_errors = list({
                        m["error_type"].replace("_", " ") for m in similar_mistakes[:3]
                    })
                    error_phrase = (
                        primary_errors[0] if len(primary_errors) == 1
                        else ", ".join(primary_errors[:-1]) + f" and {primary_errors[-1]}"
                    )
                    avg_sim = sum(m.get("similarity", 0.0) for m in similar_mistakes[:3]) / min(n, 3)
                    avg_uncertainty = [
                        m.get("uncertainty_score", m.get("kle_uncertainty")) for m in similar_mistakes[:3]
                        if m.get("uncertainty_score", m.get("kle_uncertainty")) is not None
                    ]
                    uncertainty_phrase = (
                        f" under similar uncertainty conditions (avg ≈ {sum(avg_uncertainty)/len(avg_uncertainty):.2f})"
                        if avg_uncertainty else ""
                    )
                    top_summary = historical_context[0]["clinical_summary"] if historical_context else ""
                    summary_snippet = (
                        f' The most relevant case notes: "{top_summary[:120]}…"'
                        if top_summary else ""
                    )
                    narrative = (
                        f"{count_word} similar {disease_type} case(s) were previously matched "
                        f"(avg cosine similarity {avg_sim:.2f}), primarily involving {error_phrase}{uncertainty_phrase}."
                        f"{summary_snippet}"
                    )
                    concern_flags.append(f"[HISTORY CONTEXT] {narrative}")
                    
                    # Log for debugging
                    logger.info(
                        f"CRITIC-HISTORY: Found {similar_mistakes_count} similar mistakes "
                        f"for {disease_type}, risk={historical_risk_level}"
                    )
                    
            except Exception as exc:
                historical_context = []
                logger.warning(f"[CRITIC] Past mistakes retrieval failed: {exc}")
                # Don't fail the entire evaluation if history lookup fails


        # Stage 2: LLM-based semantic critic  >>> LLM-CRITIC
        
        # Only invoked when:
        #   1. ENABLE_LLM_CRITIC is True
        #   2. Uncertainty is high OR rule-based already flagged overconfidence
        # This prevents unnecessary latency on low-risk cases.

        if settings.ENABLE_LLM_CRITIC and (current_uncertainty > 0.3 or is_overconfident):
            try:
                llm_output = medgemma_critic.critique(
                    findings=findings,
                    impression=impression,
                    kle_uncertainty=current_uncertainty,
                    historian_output=historian_output,
                    literature_output=literature_output,
                    doctor_feedback=doctor_feedback,
                    uncertainty_history=uncertainty_history,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CRITIC-LLM: Semantic critic raised exception: %s — using rule-based output only", exc
                )
                llm_output = None

            if llm_output is not None:
                # -- Merge: is_overconfident ----------------------------------
                if llm_output.overconfidence_reason is not None:
                    is_overconfident = True
                    concern_flags.append(
                        f"[LLM] Overconfidence: {llm_output.overconfidence_reason}"
                    )

                # -- Merge: justification_gap ---------------------------------
                if llm_output.justification_gap:
                    concern_flags.append(
                        f"[LLM] Justification gap: {llm_output.justification_gap}"
                    )

                # -- Merge: missing_differentials -----------------------------
                if llm_output.missing_differentials:
                    formatted = ", ".join(llm_output.missing_differentials)
                    concern_flags.append(
                        f"[LLM] Missing differentials: {formatted}"
                    )

                # -- Merge: recommended_hedging (LLM takes priority) ----------
                if llm_output.suggested_hedging:
                    recommended_hedging = llm_output.suggested_hedging

                # -- Merge: safety_score adjustment ---------------------------
                # Never let LLM directly SET safety_score; only penalise.
                if llm_output.semantic_risk_score > 0.5:
                    safety_score *= (1.0 - 0.3 * llm_output.semantic_risk_score)
                    safety_score = max(0.0, min(1.0, safety_score))

                # -- Trace logging --------------------------------------------
                logger.info(
                    "CRITIC-LLM: Semantic risk=%.2f, Missing differentials=%d",
                    llm_output.semantic_risk_score,
                    len(llm_output.missing_differentials),
                )
            else:
                # LLM failed — log and continue with rule-based output only
                logger.warning(
                    "CRITIC-LLM: Semantic critic unavailable — using rule-based output only"
                )

        # Return with historical signals (7-tuple)
        return (
            is_overconfident,
            concern_flags,
            recommended_hedging,
            safety_score,
            similar_mistakes_count,
            historical_risk_level,
            historical_context,
        )
    
    def _mentions_differentials(self, impression: str) -> bool:
        """Check if impression mentions differential diagnoses."""
        impression_lower = impression.lower()
        differential_patterns = [
            r'\bdifferential\b', r'\bconsider\b', r'\bvs\.?\b', 
            r'\bversus\b', r'\balternatively\b', r'\balternative\b'
        ]
        return any(re.search(pattern, impression_lower) for pattern in differential_patterns)
    
    def _extract_disease_type(self, impression: str, chexbert_labels: dict) -> str:
        """
        Extract primary disease category from impression and CheXbert labels.
        
        Priority:
        1. CheXbert labels marked as 'present'
        2. Impression keyword matching
        3. 'unknown' fallback
        
        Returns:
            Disease type string (lowercase, e.g., 'pneumonia', 'effusion', 'cardiomegaly')
        """
        # Priority 1: CheXbert present labels
        if chexbert_labels:
            present_labels = [k.lower() for k, v in chexbert_labels.items() if v == 'present']
            if present_labels:
                # Map common CheXbert labels to disease categories
                label_map = {
                    'consolidation': 'pneumonia',
                    'infiltration': 'pneumonia',
                    'pneumonia': 'pneumonia',
                    'edema': 'edema',
                    'effusion': 'pleural_effusion',
                    'pleural effusion': 'pleural_effusion',
                    'atelectasis': 'atelectasis',
                    'cardiomegaly': 'cardiomegaly',
                    'enlarged cardiomediastinum': 'cardiomegaly',
                    'pneumothorax': 'pneumothorax',
                    'mass': 'mass',
                    'nodule': 'nodule',
                    'fracture': 'fracture'
                }
                for label in present_labels:
                    if label in label_map:
                        return label_map[label]
                # If no mapping found, use first present label as-is
                return present_labels[0]
        
        # Priority 2: Impression keyword matching
        impression_lower = impression.lower()
        disease_keywords = [
            ('pneumonia', 'pneumonia'),
            ('consolidation', 'pneumonia'),
            ('infiltrate', 'pneumonia'),
            ('effusion', 'effusion'),
            ('edema', 'edema'),
            ('pulmonary edema', 'edema'),
            ('atelectasis', 'atelectasis'),
            ('cardiomegaly', 'cardiomegaly'),
            ('enlarged heart', 'cardiomegaly'),
            ('pneumothorax', 'pneumothorax'),
            ('mass', 'mass'),
            ('nodule', 'nodule'),
            ('fracture', 'fracture'),
            ('rib fracture', 'fracture')
        ]
        
        for keyword, disease_type in disease_keywords:
            if keyword in impression_lower:
                return disease_type
        
        # Fallback: unknown
        return 'unknown'



# Singleton instance
critic_model = CriticModel()
