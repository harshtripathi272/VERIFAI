"""
Neural Re-Ranking for Past Mistakes Retrieval

Enhances HNSW vector similarity search with:
1. Clinical relevance scoring (MedGemma-based semantic analysis)
2. Temporal recency weighting
3. User feedback signals (optional)

Provides more contextually relevant results for critic historical memory.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPORAL RECENCY WEIGHTING
# =============================================================================

def calculate_recency_weight(created_at: datetime, decay_days: int = 180) -> float:
    """
    Calculate time-decay weight for mistake relevance.
    
    Recent mistakes are more relevant than old ones (practices evolve).
    Uses exponential decay with configurable half-life.
    
    Args:
        created_at: Timestamp when mistake was created
        decay_days: Half-life in days (default: 6 months)
    
    Returns:
        Weight between 0.0 and 1.0
    """
    now = datetime.now()
    
    # Handle timezone-naive datetime
    if created_at.tzinfo is None and now.tzinfo is not None:
        import pytz
        created_at = pytz.utc.localize(created_at)
    elif created_at.tzinfo is not None and now.tzinfo is None:
        now = now.replace(tzinfo=created_at.tzinfo)
    
    days_ago = (now - created_at).days
    
    if days_ago < 0:
        # Future timestamp (shouldn't happen)
        return 1.0
    
    # Exponential decay: weight = 0.5^(days_ago / decay_days)
    weight = 0.5 ** (days_ago / decay_days)
    
    return max(0.1, weight)  # Floor at 0.1 to keep old cases minimally relevant


# =============================================================================
# CLINICAL RELEVANCE SCORING
# =============================================================================

def calculate_clinical_relevance(
    current_impression: str,
    current_kle: float,
    current_chexbert: Dict[str, str],
    mistake_original_diagnosis: str,
    mistake_corrected_diagnosis: str,
    mistake_kle: Optional[float],
    mistake_error_type: str,
    mistake_severity: int
) -> float:
    """
    Calculate clinical relevance score for a past mistake.
    
    Factors:
    - Diagnostic similarity (beyond just embedding cosine)
    - Uncertainty similarity (did both have similar KLE?)
    - Severity weight (higher severity = more relevant)
    - Error type applicability
    
    Args:
        current_impression: Current case impression
        current_kle: Current KLE uncertainty
        current_chexbert: Current CheXbert labels
        mistake_original_diagnosis: Past mistake's incorrect diagnosis
        mistake_corrected_diagnosis: Past mistake's correct diagnosis
        mistake_kle: Past mistake's KLE score
        mistake_error_type: Past mistake's error classification
        mistake_severity: Past mistake's severity (1-5)
    
    Returns:
        Relevance score 0.0-1.0
    """
    relevance = 0.0
    
    # 1. Severity weight (high-severity mistakes are more important to avoid)
    severity_weight = mistake_severity / 5.0  # Normalize to 0-1
    relevance += 0.3 * severity_weight
    
    # 2. Uncertainty similarity (if both had similar uncertainty, more relevant)
    if mistake_kle is not None:
        kle_diff = abs(current_kle - mistake_kle)
        kle_similarity = 1.0 - min(kle_diff, 1.0)
        relevance += 0.2 * kle_similarity
    
    # 3. Diagnostic pattern similarity (keyword overlap)
    current_lower = current_impression.lower()
    mistake_lower = mistake_original_diagnosis.lower()
    
    # Extract key medical terms (simple approach)
    medical_terms = ['pneumonia', 'effusion', 'edema', 'atelectasis', 'consolidation',
                     'cardiomegaly', 'pneumothorax', 'mass', 'nodule', 'fracture']
    
    current_terms = {term for term in medical_terms if term in current_lower}
    mistake_terms = {term for term in medical_terms if term in mistake_lower}
    
    if current_terms and mistake_terms:
        term_overlap = len(current_terms & mistake_terms) / len(current_terms | mistake_terms)
        relevance += 0.2 * term_overlap
    
    # 4. CheXbert label similarity
    if current_chexbert and mistake_corrected_diagnosis:
        # Simple check: if current has uncertain labels that match mistake's final diagnosis
        uncertain_labels = {k.lower() for k, v in current_chexbert.items() if v == 'uncertain'}
        mistake_corrected_lower = mistake_corrected_diagnosis.lower()
        
        for label in uncertain_labels:
            if label in mistake_corrected_lower:
                relevance += 0.15
                break
    
    # 5. Error type bonus (some error types are more instructive)
    if mistake_error_type == 'misdiagnosis':
        relevance += 0.15  # Most directly applicable
    elif mistake_error_type == 'overconfidence':
        relevance += 0.10  # High priority to avoid
    
    # Cap at 1.0
    return min(relevance, 1.0)


# =============================================================================
# MEDGEMMA-BASED SEMANTIC RELEVANCE (OPTIONAL)
# =============================================================================

def calculate_medgemma_relevance(
    current_case_summary: str,
    mistake_case_summary: str,
    use_llm: bool = False
) -> float:
    """
    Use MedGemma to assess semantic relevance between current case and past mistake.
    
    This is more sophisticated than simple embedding cosine similarity,
    as it can understand medical reasoning and context.
    
    Args:
        current_case_summary: Summary of current case
        mistake_case_summary: Summary of past mistake
        use_llm: If True, use MedGemma LLM (slow). If False, use embeddings (fast)
    
    Returns:
        Relevance score 0.0-1.0
    """
    if not use_llm:
        # Fast path: Use embeddings (already computed via HNSW)
        # This method is called AFTER vector search, so we skip redundant embedding
        return 0.0  # Placeholder - embeddings already factored in
    
    # Slow path: Use MedGemma LLM for deep semantic analysis
    try:
        from agents.critic.llm_critic import medgemma_critic
        
        if not medgemma_critic or not settings.ENABLE_LLM_CRITIC:
            return 0.0
        
        prompt = f"""You are a medical AI assistant evaluating the relevance of a past diagnostic error to a current case.

**Current Case:**
{current_case_summary}

**Past Mistake:**
{mistake_case_summary}

**Question:** On a scale of 0.0 to 1.0, how clinically relevant is this past mistake to the current case? Consider:
- Diagnostic similarity
- Clinical presentation overlap
- Potential for similar reasoning errors
- Applicability of lessons learned

**Response Format:** Return only a number between 0.0 and 1.0.

**Relevance Score:**"""
        
        response = medgemma_critic._generate(prompt, max_tokens=10)
        
        # Parse response
        try:
            score = float(response.strip())
            return max(0.0, min(score, 1.0))
        except ValueError:
            logger.warning(f"[RERANK] Failed to parse MedGemma score: {response}")
            return 0.5
            
    except Exception as e:
        logger.warning(f"[RERANK] MedGemma relevance scoring failed: {e}")
        return 0.0


# =============================================================================
# USER FEEDBACK SIGNALS (OPTIONAL)
# =============================================================================

def calculate_feedback_weight(
    mistake_id: str,
    feedback_store: Optional[Dict[str, Dict]] = None
) -> float:
    """
    Calculate weight based on user feedback signals.
    
    Feedback signals:
    - Explicit ratings (1-5 stars)
    - Implicit signals (was this mistake useful in preventing errors?)
    - Dismissal count (how often was this ignored?)
    
    Args:
        mistake_id: Unique mistake ID
        feedback_store: Optional dict of feedback {mistake_id: {rating, dismissals, helpfulness}}
    
    Returns:
        Weight 0.0-1.0
    """
    if not feedback_store or mistake_id not in feedback_store:
        return 0.5  # Neutral weight for no feedback
    
    feedback = feedback_store[mistake_id]
    
    # Explicit rating (1-5 stars)
    rating = feedback.get('rating', 3)  # Default to 3/5
    rating_weight = rating / 5.0
    
    # Helpfulness count (how many times this helped prevent errors)
    helpfulness = feedback.get('helpfulness_count', 0)
    helpfulness_weight = min(helpfulness / 10.0, 1.0)  # Cap at 10 helps = 1.0
    
    # Dismissal penalty (how often was this dismissed as irrelevant)
    dismissals = feedback.get('dismissal_count', 0)
    dismissal_penalty = min(dismissals / 20.0, 0.5)  # Max 50% penalty
    
    # Combine
    weight = (0.5 * rating_weight + 0.5 * helpfulness_weight) * (1.0 - dismissal_penalty)
    
    return max(0.1, min(weight, 1.0))


# =============================================================================
# RE-RANKING ALGORITHM
# =============================================================================

def rerank_mistakes(
    current_impression: str,
    current_kle: float,
    current_chexbert: Dict[str, str],
    retrieved_mistakes: List[Dict[str, Any]],
    use_medgemma: bool = False,
    feedback_store: Optional[Dict[str, Dict]] = None,
    recency_weight_factor: float = 0.3,
    clinical_relevance_factor: float = 0.5,
    feedback_factor: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Re-rank HNSW-retrieved mistakes using combined scoring.
    
    Final score = 
        (1 - recency_weight_factor - clinical_relevance_factor - feedback_factor) * vector_similarity
        + recency_weight_factor * temporal_recency
        + clinical_relevance_factor * clinical_relevance
        + feedback_factor * user_feedback
    
    Args:
        current_impression: Current case impression
        current_kle: Current KLE uncertainty
        current_chexbert: Current CheXbert labels
        retrieved_mistakes: List of mistakes from HNSW search (with 'similarity' field)
        use_medgemma: If True, use MedGemma for semantic relevance (slow)
        feedback_store: Optional feedback storage
        recency_weight_factor: Weight for temporal recency (default: 0.3)
        clinical_relevance_factor: Weight for clinical relevance (default: 0.5)
        feedback_factor: Weight for user feedback (default: 0.2)
    
    Returns:
        Re-ranked list of mistakes (sorted by combined score, descending)
    """
    if not retrieved_mistakes:
        return []
    
    # Normalize weights
    total_weight = recency_weight_factor + clinical_relevance_factor + feedback_factor
    if total_weight > 1.0:
        # Rescale
        recency_weight_factor /= total_weight
        clinical_relevance_factor /= total_weight
        feedback_factor /= total_weight
    
    vector_weight = 1.0 - recency_weight_factor - clinical_relevance_factor - feedback_factor
    
    reranked = []
    
    for mistake in retrieved_mistakes:
        # Original vector similarity (from HNSW)
        vector_sim = mistake.get('similarity', 0.0)
        
        # Temporal recency
        created_at = mistake.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        recency = calculate_recency_weight(created_at) if created_at else 0.5
        
        # Clinical relevance
        clinical_rel = calculate_clinical_relevance(
            current_impression=current_impression,
            current_kle=current_kle,
            current_chexbert=current_chexbert,
            mistake_original_diagnosis=mistake.get('original_diagnosis', ''),
            mistake_corrected_diagnosis=mistake.get('corrected_diagnosis', ''),
            mistake_kle=mistake.get('kle_uncertainty'),
            mistake_error_type=mistake.get('error_type', ''),
            mistake_severity=mistake.get('severity_level', 1)
        )
        
        # User feedback (if available)
        feedback_weight = 0.5  # Default neutral
        if feedback_store:
            feedback_weight = calculate_feedback_weight(
                mistake_id=mistake.get('mistake_id', ''),
                feedback_store=feedback_store
            )
        
        # Combined score
        combined_score = (
            vector_weight * vector_sim +
            recency_weight_factor * recency +
            clinical_relevance_factor * clinical_rel +
            feedback_factor * feedback_weight
        )
        
        # Add scores to mistake dict for transparency
        mistake['rerank_score'] = combined_score
        mistake['recency_score'] = recency
        mistake['clinical_relevance_score'] = clinical_rel
        mistake['feedback_score'] = feedback_weight
        
        reranked.append(mistake)
    
    # Sort by combined score (descending)
    reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    logger.info(f"[RERANK] Re-ranked {len(reranked)} mistakes. Top score: {reranked[0]['rerank_score']:.3f}")
    
    return reranked
