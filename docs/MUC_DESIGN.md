# Monotonic Uncertainty Cascade (MUC)
### VERIFAI Uncertainty Design Document

---

## Table of Contents

1. [Why We Replaced KLE](#1-why-we-replaced-kle)
2. [Research Foundation](#2-research-foundation)
3. [The Core Formula — Bidirectional IG](#3-the-core-formula--bidirectional-ig)
4. [Stage-by-Stage Flow with Code](#4-stage-by-stage-flow-with-code)
5. [Dempster-Shafer Fusion (Debate)](#5-dempster-shafer-fusion-debate)
6. [Live Cascade Example](#6-live-cascade-example)
7. [Implementation Map](#7-implementation-map)

---

## 1. Why We Replaced KLE

The old system used **Kernel Language Entropy (KLE)**, a multi-sample approach that:
- Generated 3–5 separate model completions per X-ray, took 5–15× longer
- Used cosine similarity between samples to build a semantic density matrix and compute von Neumann entropy
- Was a **single, global** uncertainty value applied at the Radiologist stage only, never updated again
- Had no feedback mechanism if downstream agents contradicted the diagnosis

### What's wrong with that

The system would produce a final diagnosis with 35% uncertainty from the Radiologist, and then even if the Critic flagged it as dangerous overconfidence and the Debate ended in deadlock, the system would still output 35% uncertainty. Nothing could ever *increase* confidence in a bad diagnosis, but nothing could ever *decrease* uncertainty either.

KLE is still kept alive in `uncertainty/kle.py` only for **case similarity search** (`case_embedding.py`) because it produces meaningful semantic distances between historical cases. It is never called during live inference.

---

## 2. Research Foundation

Each component of MUC is grounded in a specific paper.

| Paper | Published | Key Idea Used |
|---|---|---|
| **Attention Head Entropy of Large Language Models** (Ostmeier et al.) | ICML 2026 | Single-pass uncertainty from token-level distributions using entropy of the generation logit vectors. Eliminates need for multiple samples. |
| **MARS: Meaning-Aware Response Scoring** (Bakman et al., ACL 2024, 49 citations) | ACL 2024 | Tokens should be weighted by semantic importance. Low-probability tokens in medically important positions carry more uncertainty signal than random filler. |
| **Agentic Uncertainty Quantification** (Zhang et al., arXiv:2601.15703) | Jan 2026 | Bidirectional uncertainty propagation as control signals across agents. Uncertainty can increase when agents produce contradicting evidence, not just decrease monotonically. |
| **A Mathematical Theory of Evidence** (Shafer, 1976) | 1976 | Dempster-Shafer Evidence Theory — the mathematical framework for combining heterogeneous belief functions from multiple sources into a single coherent belief state. |
| **Conformal Prediction for Medical AI** (MICCAI 2024) | MICCAI 2024 | Coverage-guaranteed confidence calibration. Uncertainty values should be interpretable as calibrated probabilities, not raw model scores. |

---

## 3. The Core Formula — Bidirectional IG

### Old Formula (BROKEN)

```
IG = agent_confidence × alignment_score × scaling_factor
```

This is always `≥ 0`. Uncertainty could only **decrease**. Even if Literature found contradicting papers, or the Critic screamed "OVERCONFIDENT!", the uncertainty still dropped a tiny amount instead of going up.

### New Formula (CORRECT)

```
direction  =  (alignment_score − 0.5) × 2         → maps [0, 1] to [−1, +1]

IG  =  agent_confidence × direction × scaling_factor

U_sys(k)  =  clamp( U_sys(k−1) − IG(k),  0.05, 0.95 )
```

Where:

```
alignment > 0.5  →  direction > 0  →  IG > 0  →  uncertainty DECREASES  ✓
alignment = 0.5  →  direction = 0  →  IG = 0  →  uncertainty UNCHANGED   ✓
alignment < 0.5  →  direction < 0  →  IG < 0  →  uncertainty INCREASES   ✓
```

### The Code (`uncertainty/muc.py`)

```python
def compute_ig(
    agent_name: str,
    agent_uncertainty: float,   # Agent's own confidence in its result
    alignment_score: float,     # Does it agree with the diagnosis? [0,1]
    system_uncertainty: float,  # Current system U before this agent
    scaling_factor: float = None,
) -> IGResult:

    sf = scaling_factor or SCALING_FACTORS[agent_name]
    agent_confidence = 1.0 - agent_uncertainty

    # BIDIRECTIONAL: alignment centered at 0.5
    direction = (alignment_score - 0.5) * 2.0    # → [-1, +1]
    ig = agent_confidence * direction * sf

    new_uncertainty = max(0.05, min(0.95, system_uncertainty - ig))

    return IGResult(
        agent_name=agent_name,
        information_gain=ig,                     # CAN be negative
        system_uncertainty_before=system_uncertainty,
        system_uncertainty_after=new_uncertainty,
        ...
    )
```

### Scaling Factors (per-agent budgets)

```python
SCALING_FACTORS = {
    "chexbert":    0.20,  # [MARS] Deterministic, structured extraction — high semantic validity
    "historian":   0.15,  # [Agentic Uncertainty] Patient-specific EHR grounding
    "literature":  0.10,  # [Agentic Uncertainty] General medical evidence
    "critic":      0.10,  # Safety gating limit
    "debate":      0.25,  # [Dempster-Shafer] Resolves K-conflict for 3 agents; highest mass capacity
    "validator":   0.15,  # Terminal bounding check against absolute rules
}
```

The scaling factors are explicitly grounded in the [Research Foundation](#2-research-foundation):
- **CheXbert (0.20)**: Derived from *MARS*. As a deterministic labeler of medical conditions, its output represents the highest semantic importance tokens. 
- **Historian (0.15) & Literature (0.10)**: Derived from *Agentic Uncertainty Quantification* bidirectional signaling. The Historian carries more weight because patient-specific longitudinal data (EHR/FHIR) is a stronger prior than general literature distributions.
- **Debate (0.25)**: Derived from *Dempster-Shafer*. Because this stage fuses the belief mass of three distinct agents and normalizes by the conflict metric $K$, it mathematically requires the largest capacity to shift system uncertainty. Reaching consensus here represents the strongest evidence signal in the cascade.
- **Validator (0.15)**: The final bounding mechanism.

The sum of all positive-direction IGs (if all agents perfectly agree) is `0.95`. This allows the system to drop from a max uncertainty of `0.95` down to the hard floor of `0.05` (95% confidence) in the best case, matching clinical calibration targets.

---

## 4. Stage-by-Stage Flow with Code

### DAG Order

```
START
  │
  ▼
[Radiologist] ──────────── Sets U_sys(0) via token entropy
  │
  ▼
[CheXbert] ──────────────── IG: labels match impression?
  │
  ▼
[Evidence Gathering] ──────  IG: FHIR history + Literature (parallel)
  │          │
 [Historian]  [Literature]
  │
  ▼
[Critic] ────────────────── IG: safety score, flags, overconfidence
  │
  ▼
[Debate] ────────────────── IG: Dempster-Shafer fusion of 3 agents
  │
  ▼
[Validator] ─────────────── IG: entity F1, rules engine, recommendation
  │
  ▼
[Finalize] ──────────────── Reads final U_sys → calibrated_confidence = 1 - U_sys
  │
  ▼
 END
```

---

### Stage 0 — Radiologist (`agents/radiologist/agent.py`)

**Old code:** Called KLE 3–5 times, measured cosine similarity between embeddings.

**New code:** Single forward pass. Reads hedging language in the generated text.

```python
from uncertainty.muc import compute_token_entropy_from_text

# === SINGLE-PASS GENERATION (MUC replaces multi-sample KLE) ===
raw_output = generate_findings(image_path, view=view)

# Analyze hedging vs confidence language
full_text = f"{output.findings} {output.impression}"
token_uncertainty = compute_token_entropy_from_text(full_text)

result = {
    "radiologist_output": output,
    "current_uncertainty": token_uncertainty,    # ← Sets U_sys(0) for the pipeline
    "radiologist_kle_uncertainty": token_uncertainty,  # Legacy DB column name
    ...
}
```

**How `compute_token_entropy_from_text` works:**

Counts hedging markers (`"may", "might", "cannot exclude", "possible", "suggestive of"`, etc.) vs confidence markers (`"consistent with", "diagnostic of", "confirms"`, etc.) in the generated report text. Returns `hedge_count / total` normalized to `[0.05, 0.95]`.

When logits are available (GPU mode with full model), this will switch to true per-token entropy: `H = -Σ p(v|t) × log p(v|t)` averaged across generated tokens, normalized by `log(vocab_size)`.

---

### Stage 1 — CheXbert (`agents/chexbert/agent.py`)

**Old code:** Ran labeling but never touched `current_uncertainty` at all.

**New code:** Computes uncertainty from label distribution + alignment from label-impression overlap.

```python
from uncertainty.muc import compute_ig, compute_chexbert_uncertainty, compute_chexbert_alignment

# Get all 14 labels (present / absent / uncertain / not_mentioned)
all_labels = label_report(report_text)

# Uncertainty = fraction of ambiguous labels
chexbert_uncertainty = compute_chexbert_uncertainty(all_labels)

# Alignment = do CheXbert "present" labels appear in the radiologist's impression?
chexbert_alignment = compute_chexbert_alignment(all_labels, rad_output.impression)

ig_result = compute_ig(
    agent_name="chexbert",
    agent_uncertainty=chexbert_uncertainty,
    alignment_score=chexbert_alignment,
    system_uncertainty=system_uncertainty,   # read from state
)

return {
    "chexbert_output": output,
    "current_uncertainty": ig_result.system_uncertainty_after,  # ← written back to state
    ...
}
```

**Example behavior:**
- If CheXbert finds Pneumonia=present and Cardiomegaly=present, and both appear in the impression → alignment ≈ 0.90 → `U` drops.
- If CheXbert finds Effusion=present but the impression says "No acute findings" → alignment ≈ 0.10 → `U` rises.

---

### Stage 2 & 3 — Historian + Literature (`graph/workflow.py`)

These run in parallel via `ThreadPoolExecutor`. Their IG is computed in the `logged_evidence_gathering_node` wrapper, sequentially after both threads complete.

```python
# === MUC: Compute IG for Historian ===
hist = result.get('historian_output')
if hist:
    hist_unc   = compute_historian_uncertainty(len(hist.supporting_facts),
                                               len(hist.contradicting_facts))
    hist_align = compute_historian_alignment(len(hist.supporting_facts),
                                              len(hist.contradicting_facts),
                                              hist.confidence_adjustment)
    hist_ig    = compute_ig("historian", hist_unc, hist_align, u_current)
    u_current  = hist_ig.system_uncertainty_after

# === MUC: Compute IG for Literature ===
lit = result.get('literature_output')
if lit:
    lit_unc    = compute_literature_uncertainty(citation_count, ev_strength, ...)
    lit_align  = compute_literature_alignment(ev_strength, has_contradictions, ...)
    lit_ig     = compute_ig("literature", lit_unc, lit_align, u_current)
    u_current  = lit_ig.system_uncertainty_after

result["current_uncertainty"] = u_current   # written to state after both
```

**Historian alignment logic:**
- `supporting_count / total` → base alignment
- Boosted or penalized by `confidence_adjustment` (Historian's own net signal)
- Total contradicting facts > supporting facts → alignment < 0.5 → uncertainty rises

**Literature alignment logic:**
- `"high"` evidence strength → base alignment 0.90
- `"none"` evidence strength → base alignment 0.20
- Contradicting differential diagnoses found → multiplied by 0.7
- Synthesis contains "strongly support" → +0.10 boost
- Synthesis contains "contradicts" / "does not support" → −0.15 penalty

---

### Stage 4 — Critic (`agents/critic/agent.py`)

**Old code:** Direct assignment `current_uncertainty = 1 - safety_score` — a blunt overwrite.

**New code:** Safety score feeds _into_ the IG formula properly.

```python
from uncertainty.muc import compute_ig, compute_critic_uncertainty, compute_critic_alignment

# run the actual critic model (LLM evaluation) ...
critic_output = CriticOutput(safety_score=safety_score, ...)

# MUC IG
critic_unc   = compute_critic_uncertainty(safety_score)     # = 1 - safety_score
critic_align = compute_critic_alignment(
    safety_score=safety_score,
    is_overconfident=is_overconfident,       # halves alignment if True
    concern_flag_count=len(concern_flags),   # -0.08 per flag, capped at -0.30
)
ig_result = compute_ig(
    agent_name="critic",
    agent_uncertainty=critic_unc,
    alignment_score=critic_align,
    system_uncertainty=current_uncertainty,  # read from state
)

return {
    "critic_output": output,
    "current_uncertainty": ig_result.system_uncertainty_after,  # written to state
    ...
}
```

**Example behavior:**
- Safety=0.90, no overconfidence, 0 flags: alignment ≈ 0.90 → big drop
- Safety=0.40, IS overconfident, 4 flags: alignment ≈ `0.4 × 0.5 − 0.30 = 0.05` → alignment < 0.5 → **uncertainty spikes up**

---

### Stage 5 — Debate (`agents/debate/agent.py`)

**Old code:** Hardcoded `+0.20` uncertainty on consensus, `+0.10` on no-consensus. No physics, just magic numbers.

**New code:** Dempster-Shafer (DS) fusion of three agents (Critic + Historian + Literature), then IG formula.

```python
# Extract per-agent confidence and alignment from debate round impacts
for rnd in debate_output.rounds:
    total_critic_impact  += rnd.critic_challenge.confidence_impact
    total_hist_impact    += rnd.historian_response.confidence_impact
    total_lit_impact     += rnd.literature_response.confidence_impact

# Convert round impact → confidence (absolute strength of position)
critic_conf = max(0.1, min(0.9, 0.5 + abs(total_critic_impact)))

# Convert round impact → alignment (sign = supporting vs opposing diagnosis)
critic_align = max(0.05, min(0.95, 0.5 + total_critic_impact * 2))

# DS fusion across all three agents
fused_alignment, fused_uncertainty, conflict_K = compute_debate_ds_fusion(
    critic_confidence=critic_conf, critic_alignment=critic_align,
    historian_confidence=hist_conf, historian_alignment=hist_align,
    literature_confidence=lit_conf, literature_alignment=lit_align,
)

# Apply IG formula
ig_result = compute_ig(
    agent_name="debate",
    agent_uncertainty=fused_uncertainty,
    alignment_score=fused_alignment,
    system_uncertainty=state.get("current_uncertainty"),
)
```

The DS fusion is covered in detail in [Section 5](#5-dempster-shafer-fusion-debate).

---

### Stage 6 — Validator (`graph/workflow.py` — validator wrapper)

```python
from uncertainty.muc import compute_ig, compute_validator_uncertainty, compute_validator_alignment

entity_f1         = validator_out.get("entity_f1", 0.5)
recommendation    = validator_out.get("recommendation", "FINALIZE")
has_critical_flags = validator_out.get("has_critical_flags", False)

val_unc   = compute_validator_uncertainty(entity_f1, has_critical_flags, flag_count)
val_align = compute_validator_alignment(recommendation)
#   "FINALIZE"              → 0.95 alignment → big drop
#   "FINALIZE_LOW_CONFIDENCE" → 0.55 alignment → neutral
#   "FLAG_FOR_HUMAN"         → 0.15 alignment → uncertainty RISES

val_ig = compute_ig("validator", val_unc, val_align, u_current)
result["current_uncertainty"] = val_ig.system_uncertainty_after
```

---

### Stage 7 — Finalize (`graph/workflow.py`)

No IG computed here. Just reads the final `current_uncertainty` from state and converts it to a `calibrated_confidence`:

```python
uncertainty = state.get("current_uncertainty", 0.5)
calibrated_confidence = round(1.0 - uncertainty, 4)
```

---

## 5. Dempster-Shafer Fusion (Debate)

Dempster-Shafer theory allows combining evidence from multiple independent agents who each express belief as a **mass function** over a frame of discernment `Ω = {diagnosis_correct, diagnosis_wrong, unknown}`.

### Building a Mass Function

Each agent's confidence and alignment is converted to a DS mass function:

```python
def build_mass_function(agent_confidence, alignment_score) -> DSMassFunction:
    belief_mass = agent_confidence * 0.8   # reserve 20% as "unknown"
    confirm  = belief_mass * alignment_score          # mass on "diagnosis correct"
    deny     = belief_mass * (1.0 - alignment_score)  # mass on "diagnosis wrong"
    uncertain = 1.0 - belief_mass                     # mass on "don't know"
    return DSMassFunction(confirm, deny, uncertain)
```

### Combining Two Mass Functions (Dempster's Rule)

```python
def dempster_combine(m1, m2) -> DSMassFunction:
    # Intersect all focal elements
    confirm  = m1.confirm*m2.confirm + m1.confirm*m2.uncertain + m1.uncertain*m2.confirm
    deny     = m1.deny*m2.deny + m1.deny*m2.uncertain + m1.uncertain*m2.deny
    uncertain = m1.uncertain * m2.uncertain

    # Conflict coefficient K = mass assigned to empty set
    K = m1.confirm*m2.deny + m1.deny*m2.confirm

    if K >= 0.99:
        return DSMassFunction(confirm=0, deny=0, uncertain=1.0)  # total conflict

    # Normalize by (1 - K)
    norm = 1.0 / (1.0 - K)
    return DSMassFunction(confirm*norm, deny*norm, uncertain*norm)
```

### What K (Conflict) Means

- `K = 0.0` — agents agree perfectly, combination is strong
- `K = 0.5` — moderate conflict, normalization dilutes the joint belief 2×
- `K ≥ 0.99` — total conflict (Critic says "correct", others say "wrong"), result is complete uncertainty — **no IG is produced**, system uncertainty stays put

This is exactly the right behavior: if agents violently disagree, the system correctly remains uncertain rather than arbitrarily picking a winner.

The fused alignment feeds into the standard `compute_ig("debate", ...)` call.

---

## 6. Live Cascade Example

### Bad Diagnosis (Agents Contradict)

```
Stage        │ Uncertainty IN │ Alignment │ IG      │ Uncertainty OUT │ Direction
─────────────┼────────────────┼───────────┼─────────┼─────────────────┼──────────
Radiologist  │      —         │    —      │    —    │   0.700         │ Sets U₀
CheXbert     │   0.700        │  0.80     │ +0.077  │   0.623         │   ↓
Historian    │   0.623        │  0.70     │ +0.036  │   0.587         │   ↓
Literature   │   0.587        │  0.25     │ −0.035  │   0.622         │   ↑ contradicts
Critic       │   0.622        │  0.20     │ −0.054  │   0.676         │   ↑ flags danger
Debate       │   0.676        │  0.50     │  0.000  │   0.676         │   → no consensus
Validator    │   0.676        │  0.15     │ −0.045  │   0.721         │   ↑ FLAG_FOR_HUMAN
─────────────┴────────────────┴───────────┴─────────┴─────────────────┴──────────
FINAL: U = 0.721   →   confidence = 28%   →   FLAG_FOR_HUMAN
```

### Good Diagnosis (Agents Confirm)

```
Stage        │ Uncertainty IN │ Alignment │ IG      │ Uncertainty OUT │ Direction
─────────────┼────────────────┼───────────┼─────────┼─────────────────┼──────────
Radiologist  │      —         │    —      │    —    │   0.700         │ Sets U₀
CheXbert     │   0.700        │  0.90     │ +0.108  │   0.592         │   ↓
Historian    │   0.592        │  0.85     │ +0.071  │   0.521         │   ↓
Literature   │   0.521        │  0.90     │ +0.072  │   0.449         │   ↓
Critic       │   0.449        │  0.90     │ +0.076  │   0.373         │   ↓
Debate       │   0.373        │  0.90     │ +0.130  │   0.243         │   ↓
Validator    │   0.243        │  0.95     │ +0.065  │   0.178         │   ↓
─────────────┴────────────────┴───────────┴─────────┴─────────────────┴──────────
FINAL: U = 0.178   →   confidence = 82%   →   FINALIZE
```

---

## 7. Implementation Map

| File | Role |
|---|---|
| `uncertainty/muc.py` | **Core engine.** Bidirectional IG formula, all per-agent uncertainty/alignment helpers, DS mass functions and combination rule |
| `uncertainty/__init__.py` | Public API — exports all MUC functions. Also re-exports legacy KLE for `case_embedding.py` only |
| `uncertainty/kle.py` | **LEGACY. Not called during inference.** Only used by `case_embedding.py` for historical case search |
| `agents/radiologist/agent.py` | Single-pass generation + `compute_token_entropy_from_text` → sets `current_uncertainty` |
| `agents/chexbert/agent.py` | Label distribution uncertainty + impression alignment → `compute_ig("chexbert", ...)` |
| `agents/critic/agent.py` | Safety score + overconfidence + flag count → `compute_ig("critic", ...)` |
| `agents/debate/agent.py` | Debate round impacts → DS mass functions → `compute_debate_ds_fusion` → `compute_ig("debate", ...)` |
| `graph/workflow.py` | Evidence gathering wrapper (Historian IG + Literature IG), Validator wrapper (Validator IG), all logged node wrappers with Uncertainty IN/OUT printing |

### How `current_uncertainty` Flows Through LangGraph

LangGraph merges any keys returned in a node's result dict back into the shared `VerifaiState`. So as long as every node writes `"current_uncertainty"` into its return dict, it is automatically propagated as the next node's input. Each node reads it via `state.get("current_uncertainty", 0.5)`.

```
radiologist result → {"current_uncertainty": 0.70}
          ↓
chexbert state reads 0.70, returns {"current_uncertainty": 0.62}
          ↓
evidence_gathering state reads 0.62, wrapper writes {"current_uncertainty": 0.51}
          ↓
critic state reads 0.51, returns {"current_uncertainty": 0.44}
          ↓
debate state reads 0.44, returns {"current_uncertainty": 0.32}
          ↓
validator state reads 0.32, wrapper writes {"current_uncertainty": 0.18}
          ↓
finalize state reads 0.18 → calibrated_confidence = 0.82
```
