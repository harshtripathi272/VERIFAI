"""
Monotonic Uncertainty Cascade (MUC) — Core Module

Implements the Information Gain (IG) framework for VERIFAI's multi-agent
uncertainty tracking. Each agent computes:

    IG = agent_confidence × direction × scaling_factor

Where:
    direction = (alignment_score - 0.5) × 2       # maps [0,1] → [-1,+1]

    alignment > 0.5  →  CONFIRMS diagnosis  →  IG > 0  →  uncertainty DECREASES
    alignment = 0.5  →  NEUTRAL/GARBAGE     →  IG = 0  →  uncertainty UNCHANGED
    alignment < 0.5  →  CONTRADICTS         →  IG < 0  →  uncertainty INCREASES

System uncertainty is updated:

    U_sys(k) = clamp( U_sys(k-1) - IG(k) , 0.05, 0.95 )

References:
- "Attention Head Entropy of LLMs" (Ostmeier et al., ICML 2026)
- "MARS: Meaning-Aware Response Scoring" (Bakman et al., ACL 2024)
- "Agentic Uncertainty Quantification" (Zhang et al., arXiv 2601.15703)
- "A Mathematical Theory of Evidence" (Shafer, 1976)
"""

import math
from collections import Counter
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# SCALING FACTORS — LEGACY (trace display only, NOT used in formula)
# ═══════════════════════════════════════════════════════════════════════════════

SCALING_FACTORS = {
    "chexbert":    0.20,  # [MARS] Deterministic, structured extraction — high semantic validity
    "historian":   0.15,  # [Agentic Uncertainty] Patient-specific EHR grounding
    "literature":  0.10,  # [Agentic Uncertainty] General medical evidence
    "critic":      0.10,  # Safety gating limit
    "debate":      0.25,  # [Dempster-Shafer] Resolves K-conflict for 3 agents; highest mass capacity
    "validator":   0.15,  # Terminal bounding check against absolute rules
}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE IG FORMULA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IGResult:
    """Result of an Information Gain computation for one agent."""
    agent_name: str
    agent_uncertainty: float   # Agent's own uncertainty [0, 1]
    agent_confidence: float    # 1 - agent_uncertainty
    alignment_score: float     # Agreement with radiologist [0, 1]
    scaling_factor: float      # Agent-specific weight
    information_gain: float    # Can be NEGATIVE if agent contradicts
    system_uncertainty_before: float
    system_uncertainty_after: float

    def __repr__(self):
        return (
            f"IG({self.agent_name}): {self.information_gain:+.4f} "
            f"[conf={self.agent_confidence:.3f} × align={self.alignment_score:.3f} "
            f"× scale={self.scaling_factor:.2f}] "
            f"U: {self.system_uncertainty_before:.4f} → {self.system_uncertainty_after:.4f}"
        )


def compute_ig(
    agent_name: str,
    agent_uncertainty: float,
    alignment_score: float,
    system_uncertainty: float,
    scaling_factor: Optional[float] = None,
) -> IGResult:
    """
    Compute Information Gain for an agent and update system uncertainty.

    Alignment is BIDIRECTIONAL around 0.5:
        > 0.5 → agent CONFIRMS diagnosis  → IG > 0 → uncertainty drops
        = 0.5 → agent is NEUTRAL          → IG = 0 → uncertainty unchanged
        < 0.5 → agent CONTRADICTS          → IG < 0 → uncertainty RISES

    Args:
        agent_name: Name of the agent (e.g. "chexbert", "historian")
        agent_uncertainty: Agent's own uncertainty [0, 1]
        alignment_score: How much agent agrees with diagnosis [0, 1]
            > 0.5 = confirms, < 0.5 = contradicts
        system_uncertainty: Current system uncertainty before this agent
        scaling_factor: Override scaling factor (uses default if None)

    Returns:
        IGResult with the computed IG and updated system uncertainty
    """
    # Clamp inputs
    agent_uncertainty = max(0.0, min(1.0, agent_uncertainty))
    alignment_score = max(0.0, min(1.0, alignment_score))
    system_uncertainty = max(0.0, min(1.0, system_uncertainty))

    # Legacy scaling factor — trace display only, NOT used in formula
    sf = scaling_factor if scaling_factor is not None else SCALING_FACTORS.get(agent_name, 0.0)

    agent_confidence = 1.0 - agent_uncertainty
    direction = (alignment_score - 0.5) * 2.0  # maps [0,1] → [-1,+1]

    # === BAYESIAN LOG-ODDS UPDATE ===
    # Ref: Zhang et al., "Agentic UQ", arXiv:2601.15703, Eq. 5-6
    # P(V_t|h_t) = c_t * P(V_{t-1}|h_{t-1}) — multiplicative
    # Implemented via logit transform for numerical stability
    eps = 1e-7
    U = max(eps, min(1 - eps, system_uncertainty))
    log_odds = math.log(U / (1 - U))

    # Evidence strength = agent confidence * |direction|
    # Agent confidence IS the natural weight — no scaling factor needed
    evidence_strength = agent_confidence * abs(direction)

    # Confirming -> decrease log-odds -> decrease uncertainty
    # Contradicting -> increase log-odds -> increase uncertainty
    # Neutral (direction=0) -> evidence_strength=0 -> no change
    sign = -1.0 if direction >= 0 else 1.0
    log_odds_new = log_odds + sign * evidence_strength

    # Back to probability via sigmoid
    new_uncertainty = 1.0 / (1.0 + math.exp(-log_odds_new))
    new_uncertainty = max(0.05, min(0.95, new_uncertainty))

    # Effective IG for trace compatibility (positive = uncertainty decreased)
    ig = system_uncertainty - new_uncertainty

    return IGResult(
        agent_name=agent_name,
        agent_uncertainty=agent_uncertainty,
        agent_confidence=agent_confidence,
        alignment_score=alignment_score,
        scaling_factor=sf,
        information_gain=ig,
        system_uncertainty_before=system_uncertainty,
        system_uncertainty_after=new_uncertainty,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0: RADIOLOGIST — Token Entropy
# ═══════════════════════════════════════════════════════════════════════════════

def compute_token_entropy(
    logits_list: List[np.ndarray],
    vocab_size: int = 32000,
) -> float:
    """
    Compute normalized token-level entropy from generation logits.

    H_token = -(1/T) × Σ_t Σ_v p(v|t) × log(p(v|t))
    U_rad   = H_token / log(V)

    Args:
        logits_list: List of (vocab_size,) logit arrays, one per generated token
        vocab_size: Vocabulary size for normalization (default: 32000 for Gemma)

    Returns:
        Normalized uncertainty in [0, 1]
    """
    if not logits_list:
        return 0.5  # Default if no logits

    total_entropy = 0.0
    for logits in logits_list:
        # Softmax → probabilities
        logits = np.array(logits, dtype=np.float64)
        logits -= logits.max()  # Numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        # Entropy: -Σ p log p
        # Filter zeros to avoid log(0)
        mask = probs > 1e-10
        entropy = -np.sum(probs[mask] * np.log(probs[mask]))
        total_entropy += entropy

    # Average entropy per token
    avg_entropy = total_entropy / len(logits_list)

    # Normalize by log(V) → [0, 1]
    max_entropy = math.log(vocab_size)
    normalized = avg_entropy / max_entropy if max_entropy > 0 else 0.0

    return max(0.0, min(1.0, normalized))


def compute_token_entropy_from_text(text: str) -> float:
    """
    Heuristic token entropy when logits are unavailable (mock mode / API mode).

    Counts hedging vs confidence markers in the text.

    Returns:
        Estimated uncertainty in [0, 1]
    """
    text_lower = text.lower()

    hedging_markers = [
        "may", "might", "could", "possibly", "suggestive of",
        "cannot exclude", "cannot rule out", "questionable",
        "uncertain", "equivocal", "differential includes",
        "consider", "correlate clinically", "limited evaluation",
        "possible", "probable", "suspicious"
    ]

    confidence_markers = [
        "consistent with", "diagnostic of", "clearly",
        "definitively", "confirms", "characteristic of",
        "pathognomonic", "classical", "typical",
        "no evidence of", "normal", "unremarkable"
    ]

    hedge_count = sum(1 for m in hedging_markers if m in text_lower)
    conf_count = sum(1 for m in confidence_markers if m in text_lower)

    total = hedge_count + conf_count
    if total == 0:
        return 0.5  # Neutral

    # More hedging → higher uncertainty
    uncertainty = hedge_count / total
    return max(0.05, min(0.95, uncertainty))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: CHEXBERT — Label Distribution Uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

def compute_chexbert_uncertainty(labels: Dict[str, str]) -> float:
    """
    Compute CheXbert uncertainty from label distribution.

    Since f1chexbert returns categorical labels (present/absent/uncertain/not_mentioned)
    rather than softmax probabilities, we estimate uncertainty from the proportion
    of "uncertain" and "not_mentioned" labels vs total.

    Args:
        labels: Dict of {condition: status} where status ∈ {present, absent, uncertain, not_mentioned}

    Returns:
        Uncertainty in [0, 1]
    """
    if not labels:
        return 0.8  # No labels = high uncertainty

    # Shannon entropy of label distribution, normalized by H_max
    # Ref: Shannon, "A Mathematical Theory of Communication", 1948
    # If all labels agree (e.g. all "present") -> H=0 -> low uncertainty
    # If evenly split across categories -> H=H_max -> high uncertainty
    category_counts = Counter(labels.values())
    total = sum(category_counts.values())
    categories = ["present", "absent", "uncertain", "not_mentioned"]

    probs = [category_counts.get(cat, 0) / total for cat in categories]
    H = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
    H_max = math.log(len(categories))  # log(4)
    uncertainty = H / H_max if H_max > 0 else 0.0

    return max(0.05, min(0.95, uncertainty))


def compute_chexbert_alignment(
    labels: Dict[str, str],
    impression_text: str,
) -> float:
    """
    Compute alignment between CheXbert labels and radiologist impression.

    Checks how many CheXbert "present" labels overlap with conditions
    mentioned in the impression text.

    Args:
        labels: CheXbert output labels
        impression_text: Radiologist impression text

    Returns:
        Alignment score in [0, 1]
    """
    if not labels or not impression_text:
        return 0.5  # Neutral

    impression_lower = impression_text.lower()

    present_labels = [
        cond for cond, status in labels.items()
        if status == "present"
    ]

    if not present_labels:
        # No positive findings — check if impression also says "no finding" / "normal"
        normal_keywords = ["normal", "no finding", "unremarkable", "no acute", "no significant"]
        if any(kw in impression_lower for kw in normal_keywords):
            return 0.9  # Both agree: nothing found
        return 0.4  # CheXbert found nothing but impression says something

    # Count how many present conditions are mentioned in the impression
    mentioned = 0
    for cond in present_labels:
        cond_lower = cond.lower()
        # Check both full name and common abbreviations
        if cond_lower in impression_lower:
            mentioned += 1
        elif any(word in impression_lower for word in cond_lower.split()):
            mentioned += 0.5  # Partial match

    alignment = mentioned / len(present_labels)
    return max(0.05, min(0.95, alignment))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: HISTORIAN — Evidence Ratio
# ═══════════════════════════════════════════════════════════════════════════════

def compute_historian_uncertainty(
    supporting_count: int,
    contradicting_count: int,
) -> float:
    """
    Compute historian uncertainty from evidence balance.

    More contradicting facts relative to supporting → higher uncertainty.
    More total evidence → more confidence in measurement (evidence_factor).

    Args:
        supporting_count: Number of supporting FHIR facts
        contradicting_count: Number of contradicting FHIR facts

    Returns:
        Uncertainty in [0, 1]
    """
    total = supporting_count + contradicting_count

    if total == 0:
        return 0.7  # No evidence = moderate-high uncertainty

    # Contradiction ratio
    contradiction_ratio = contradicting_count / total

    # Evidence factor: more evidence = more confidence in measurement
    evidence_factor = 1.0 / (1.0 + 0.1 * total)

    uncertainty = contradiction_ratio * (1.0 - evidence_factor) + evidence_factor
    return max(0.05, min(0.95, uncertainty))


def compute_historian_alignment(
    supporting_count: int,
    contradicting_count: int,
    confidence_adjustment: float = 0.0,
) -> float:
    """
    Compute historian alignment with primary diagnosis.

    Alignment = proportion of supporting facts + boost from confidence adjustment.

    Args:
        supporting_count: Number of supporting facts
        contradicting_count: Number of contradicting facts
        confidence_adjustment: Historian's net confidence adjustment

    Returns:
        Alignment in [0, 1]
    """
    total = supporting_count + contradicting_count

    if total == 0:
        return 0.5  # Neutral — no evidence either way

    base_alignment = supporting_count / total
    # Boost from confidence adjustment (can be negative)
    boosted = base_alignment + (confidence_adjustment * 0.5)
    return max(0.05, min(0.95, boosted))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: LITERATURE — Citation Strength
# ═══════════════════════════════════════════════════════════════════════════════

def compute_literature_uncertainty(
    citation_count: int,
    evidence_strength: str,  # "high", "medium", "low", "none"
    high_strength_ratio: float = 0.0,
) -> float:
    """
    Compute literature uncertainty from citation count and evidence strength.

    Args:
        citation_count: Number of citations found
        evidence_strength: Overall evidence strength rating
        high_strength_ratio: Proportion of high-strength citations

    Returns:
        Uncertainty in [0, 1]
    """
    strength_scores = {"high": 0.15, "medium": 0.35, "low": 0.55, "none": 0.80}
    base = strength_scores.get(evidence_strength, 0.60)

    # Citation factor: more citations = lower uncertainty
    citation_factor = 1.0 / (1.0 + 0.15 * citation_count) if citation_count > 0 else 1.0

    uncertainty = base * citation_factor * (1.0 - high_strength_ratio * 0.3)
    return max(0.05, min(0.95, uncertainty))


def compute_literature_alignment(
    evidence_strength: str,
    has_contradicting_differentials: bool = False,
    synthesis_text: str = "",
) -> float:
    """
    Compute literature alignment with primary diagnosis.

    Based on evidence strength, penalized if papers mention contradicting
    differential diagnoses.

    Args:
        evidence_strength: "high", "medium", "low", "none"
        has_contradicting_differentials: Whether papers contradict
        synthesis_text: Text of the synthesis for keyword analysis

    Returns:
        Alignment in [0, 1]
    """
    strength_alignment = {"high": 0.90, "medium": 0.65, "low": 0.40, "none": 0.20}
    base = strength_alignment.get(evidence_strength, 0.40)

    # Penalty for contradicting differentials
    if has_contradicting_differentials:
        base *= 0.7

    # Keyword boost/penalty from synthesis text
    if synthesis_text:
        text_lower = synthesis_text.lower()
        support_kw = ["strongly support", "robust evidence", "consistent with the literature",
                       "well-established", "confirmed by"]
        contra_kw = ["contradicts", "does not support", "inconsistent with",
                      "argues against", "conflicting evidence"]

        if any(kw in text_lower for kw in support_kw):
            base = min(0.95, base + 0.10)
        elif any(kw in text_lower for kw in contra_kw):
            base = max(0.10, base - 0.15)

    return max(0.05, min(0.95, base))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: CRITIC — Safety-Based
# ═══════════════════════════════════════════════════════════════════════════════

def compute_critic_uncertainty(safety_score: float) -> float:
    """Critic uncertainty = 1 - safety_score."""
    return max(0.05, min(0.95, 1.0 - safety_score))


def compute_critic_alignment(
    safety_score: float,
    is_overconfident: bool,
    concern_flag_count: int,
) -> float:
    """
    Compute critic alignment.

    Starts with safety_score, penalized for overconfidence and concern flags.

    Args:
        safety_score: Critic's safety score [0, 1]
        is_overconfident: Whether critic flagged overconfidence
        concern_flag_count: Number of concern flags raised

    Returns:
        Alignment in [0, 1]
    """
    alignment = safety_score

    # Overconfidence = Critic CONTRADICTS the system's confidence level
    # Semantically: "despite appearing safe, this is overconfident"
    # So we FLIP alignment (high safety -> low alignment = contradiction)
    if is_overconfident:
        alignment = 1.0 - alignment

    # Log-diminishing flag penalty (information-theoretic):
    # First flag is most informative, each subsequent adds less
    # penalty = log(1+n) / log(1+N_max), where N_max=10
    # Ref: Shannon information -- diminishing marginal information
    if concern_flag_count > 0:
        flag_penalty = math.log1p(concern_flag_count) / math.log1p(10)
        alignment = max(0.05, alignment * (1.0 - flag_penalty * 0.5))

    return max(0.05, min(0.95, alignment))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: DEBATE — Dempster-Shafer Fusion
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DSMassFunction:
    """Dempster-Shafer mass function for an agent's belief."""
    confirm: float = 0.0    # Mass on "diagnosis is correct"
    deny: float = 0.0       # Mass on "diagnosis is incorrect"
    uncertain: float = 1.0  # Mass on "don't know" (frame of discernment)

    def __post_init__(self):
        total = self.confirm + self.deny + self.uncertain
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.confirm /= total
            self.deny /= total
            self.uncertain /= total


def build_mass_function(
    agent_name: str,
    agent_confidence: float,
    alignment_score: float,
) -> DSMassFunction:
    """
    Build a Dempster-Shafer mass function from an agent's confidence and alignment.

    High confidence + high alignment → mass on "confirm"
    High confidence + low alignment → mass on "deny"
    Low confidence → mass on "uncertain"

    Args:
        agent_name: Name of the agent
        agent_confidence: 1 - agent_uncertainty
        alignment_score: Agreement with diagnosis [0, 1]

    Returns:
        DSMassFunction
    """
    # How much belief the agent has to distribute
    belief_mass = agent_confidence * 0.8  # Reserve 20% for uncertainty
    remaining = 1.0 - belief_mass

    # Split belief between confirm and deny based on alignment
    confirm = belief_mass * alignment_score
    deny = belief_mass * (1.0 - alignment_score)

    return DSMassFunction(
        confirm=confirm,
        deny=deny,
        uncertain=remaining,
    )


def dempster_combine(m1: DSMassFunction, m2: DSMassFunction) -> DSMassFunction:
    """
    Combine two mass functions using Dempster's rule of combination.

    m_combined(A) = (1/(1-K)) × Σ m1(B) × m2(C) for all B∩C = A

    where K is the conflict coefficient.

    Args:
        m1: First mass function
        m2: Second mass function

    Returns:
        Combined mass function
    """
    # Compute all focal element intersections
    # {confirm} ∩ {confirm} = {confirm}
    # {confirm} ∩ {deny}    = ∅ (conflict)
    # {confirm} ∩ {uncertain} = {confirm}
    # {deny} ∩ {deny}      = {deny}
    # {deny} ∩ {confirm}   = ∅ (conflict)
    # {deny} ∩ {uncertain} = {deny}
    # {uncertain} ∩ * = *

    # Combined masses before normalization
    confirm = (
        m1.confirm * m2.confirm +
        m1.confirm * m2.uncertain +
        m1.uncertain * m2.confirm
    )

    deny = (
        m1.deny * m2.deny +
        m1.deny * m2.uncertain +
        m1.uncertain * m2.deny
    )

    uncertain = m1.uncertain * m2.uncertain

    # Conflict
    K = (
        m1.confirm * m2.deny +
        m1.deny * m2.confirm
    )

    # Normalization factor
    if K >= 0.99:
        # Total conflict — agents completely disagree
        return DSMassFunction(confirm=0.0, deny=0.0, uncertain=1.0)

    norm = 1.0 / (1.0 - K)

    return DSMassFunction(
        confirm=confirm * norm,
        deny=deny * norm,
        uncertain=uncertain * norm,
    )


def compute_debate_ds_fusion(
    critic_confidence: float,
    critic_alignment: float,
    historian_confidence: float,
    historian_alignment: float,
    literature_confidence: float,
    literature_alignment: float,
) -> Tuple[float, float, float]:
    """
    Fuse three agents' beliefs using Dempster-Shafer combination.

    Returns:
        Tuple of (fused_alignment, fused_uncertainty, conflict_coefficient)
    """
    m_critic = build_mass_function("critic", critic_confidence, critic_alignment)
    m_historian = build_mass_function("historian", historian_confidence, historian_alignment)
    m_literature = build_mass_function("literature", literature_confidence, literature_alignment)

    # Combine pairwise: (critic ⊕ historian) ⊕ literature
    m_ch = dempster_combine(m_critic, m_historian)
    m_fused = dempster_combine(m_ch, m_literature)

    # Compute conflict coefficient
    K = (
        m_critic.confirm * m_historian.deny +
        m_critic.deny * m_historian.confirm +
        m_ch.confirm * m_literature.deny +
        m_ch.deny * m_literature.confirm
    )
    K = min(1.0, K)

    # Fused alignment: how much the group believes the diagnosis is correct
    fused_alignment = m_fused.confirm / (m_fused.confirm + m_fused.deny + 1e-10)

    # Fused uncertainty: how much the group is unsure
    fused_uncertainty = m_fused.uncertain + m_fused.deny * 0.5

    return fused_alignment, min(0.95, fused_uncertainty), K


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6: VALIDATOR — Entity F1 + Rules
# ═══════════════════════════════════════════════════════════════════════════════

def compute_validator_uncertainty(
    entity_f1: float,
    has_critical_flags: bool,
    flag_count: int,
    retrieval_agrees: bool = True,
) -> float:
    """
    Compute validator uncertainty from entity matching and rules.

    Args:
        entity_f1: RadGraph entity F1 score [0, 1]
        has_critical_flags: Whether critical safety flags were triggered
        flag_count: Total rules engine flags
        retrieval_agrees: Whether retrieval consensus matches

    Returns:
        Uncertainty in [0, 1]
    """
    # Entity uncertainty: low F1 = high uncertainty
    entity_unc = 1.0 - entity_f1 if entity_f1 is not None else 0.5

    # Retrieval factor
    retrieval_factor = 0.8 if retrieval_agrees else 1.2

    # Flag penalty
    flag_penalty = 0.15 if has_critical_flags else min(0.1, flag_count * 0.03)

    uncertainty = entity_unc * retrieval_factor * 0.6 + flag_penalty
    return max(0.05, min(0.95, uncertainty))


def compute_validator_alignment(recommendation: str, entity_f1: float = 0.5) -> float:
    """
    Validator alignment from discrete decision + continuous Entity F1.

    Base alignment uses ordinal quantile midpoints (3 categories
    uniformly partitioning [0,1]):
        FINALIZE -> upper quartile midpoint = 0.75
        FINALIZE_LOW_CONFIDENCE -> middle = 0.50
        FLAG_FOR_HUMAN -> lower quartile midpoint = 0.25

    Entity F1 provides continuous refinement within each category.

    Args:
        recommendation: FINALIZE, FINALIZE_LOW_CONFIDENCE, or FLAG_FOR_HUMAN
        entity_f1: RadGraph entity F1 score [0, 1] for refinement

    Returns:
        Alignment in [0, 1]
    """
    # Ordinal quantile midpoints (defensible: 3 categories in [0,1])
    base_map = {
        "FINALIZE": 0.75,
        "FINALIZE_LOW_CONFIDENCE": 0.50,
        "FLAG_FOR_HUMAN": 0.25,
    }
    base = base_map.get(recommendation, 0.50)

    # Continuous refinement from Entity F1 (centered at 0.5, +/-0.20)
    refinement = (entity_f1 - 0.5) * 0.4
    return max(0.05, min(0.95, base + refinement))


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADE RUNNER — Utility for running the full cascade
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CascadeResult:
    """Full result of running the MUC cascade."""
    initial_uncertainty: float
    final_uncertainty: float
    ig_results: List[IGResult] = field(default_factory=list)
    total_ig: float = 0.0

    def add(self, ig: IGResult):
        self.ig_results.append(ig)
        self.total_ig += ig.information_gain
        self.final_uncertainty = ig.system_uncertainty_after

    def summary(self) -> str:
        lines = [
            f"MUC Cascade: {self.initial_uncertainty:.4f} → {self.final_uncertainty:.4f} "
            f"(total IG = {self.total_ig:.4f})",
            "─" * 70,
        ]
        for ig in self.ig_results:
            lines.append(
                f"  {ig.agent_name:12s} │ conf={ig.agent_confidence:.3f} "
                f"│ align={ig.alignment_score:.3f} │ scale={ig.scaling_factor:.2f} "
                f"│ IG={ig.information_gain:+.4f} │ U={ig.system_uncertainty_after:.4f}"
            )
        return "\n".join(lines)
