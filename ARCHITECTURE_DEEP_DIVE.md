# VERIFAI Architecture Deep Dive

*A comprehensive guide to understanding what we're building, why, and how*

---

## Table of Contents

1. [The Problem We're Solving](#part-1-what-problem-are-we-solving)
2. [Dual-Head Architecture](#part-2-the-core-innovation---dual-head-architecture)
3. [Uncertainty-Gated Routing](#part-3-the-uncertainty-gated-routing-system)
4. [The Agent Council](#part-4-the-agent-council)
5. [MCP Integration](#part-5-mcp-model-context-protocol---how-agents-access-tools)
6. [The Proof Layer](#part-6-the-proof-layer)
7. [Training Pipeline](#part-7-training-pipeline)
8. [Implementation Roadmap](#part-8-implementation-roadmap)
9. [Competition Strategy](#part-9-why-this-wins)

---

## Part 1: What Problem Are We Solving?

### The Current State of Medical AI

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TODAY'S MEDICAL AI PROBLEMS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. BLACK BOX PROBLEM                                                    │
│     ┌──────────┐      ┌──────────┐      ┌──────────┐                    │
│     │ X-ray    │ ───► │   CNN    │ ───► │ "Pneumonia" │                 │
│     └──────────┘      │ (hidden) │      │ (WHY???)    │                 │
│                       └──────────┘      └──────────┘                    │
│     Doctor: "I can't trust this. I don't know WHY it said pneumonia."   │
│                                                                          │
│  2. OVERCONFIDENCE PROBLEM                                               │
│     ┌──────────┐      ┌──────────┐      ┌──────────────────┐            │
│     │ Blurry   │ ───► │  GPT-4V  │ ───► │ "Definitely TB"  │            │
│     │ X-ray    │      │          │      │ Confidence: 95%  │            │
│     └──────────┘      └──────────┘      └──────────────────┘            │
│     Reality: It was just a smudge. Model was WRONG but CONFIDENT.       │
│                                                                          │
│  3. ISOLATION PROBLEM                                                    │
│     Model sees ONLY the image. Doesn't know:                            │
│     • Patient has diabetes (from FHIR records)                          │
│     • Patient was on immunosuppressants (from medications)              │
│     • Recent paper shows new diagnostic criteria (from PubMed)          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### What VERIFAI Does Differently

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VERIFAI'S APPROACH                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. EXPLAINABLE: Every diagnosis comes with PROOF                       │
│     • Visual: "I focused on THIS region" (Grad-CAM)                     │
│     • Clinical: "Patient's diabetes is relevant because..."             │
│     • Literary: "PubMed study X supports this diagnosis"                │
│                                                                          │
│  2. UNCERTAINTY-AWARE: Model knows when it DOESN'T know                 │
│     • Low uncertainty → Quick diagnosis                                 │
│     • High uncertainty → Calls for backup (more agents)                 │
│     • Very high uncertainty → "I need human review"                     │
│                                                                          │
│  3. CONTEXT-INTEGRATED: Pulls in external knowledge                     │
│     • FHIR: Patient's medical history                                   │
│     • PubMed: Latest research                                           │
│     • Prior imaging: Previous X-rays for comparison                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: The Core Innovation - Dual-Head Architecture

### Why Two Heads?

Think of it like having two radiologists look at an X-ray:
- **Radiologist Head**: "I think this is pneumonia"
- **Critic Head**: "Wait, are you sure? Your attention is scattered, your confidence seems too high for this blurry image"

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE DUAL-HEAD CONCEPT                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         X-ray Image                                      │
│                              │                                           │
│                              ▼                                           │
│                    ┌──────────────────┐                                  │
│                    │   MedSigLIP      │  ◄── Shared vision encoder       │
│                    │   (Frozen)       │      Converts image to           │
│                    │                  │      numerical representation    │
│                    └────────┬─────────┘                                  │
│                             │                                            │
│              ┌──────────────┴──────────────┐                             │
│              │                             │                             │
│              ▼                             ▼                             │
│    ┌─────────────────┐          ┌─────────────────┐                      │
│    │ RADIOLOGIST HEAD│          │   CRITIC HEAD   │                      │
│    │                 │          │                 │                      │
│    │ "What do I see?"│          │ "How confident  │                      │
│    │ "What's the Dx?"│          │  should we be?" │                      │
│    │                 │          │                 │                      │
│    │ Output:         │          │ Output:         │                      │
│    │ • Diagnosis     │          │ • Uncertainty   │                      │
│    │ • Findings      │          │   score         │                      │
│    │ • Confidence    │          │                 │                      │
│    └─────────────────┘          └─────────────────┘                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### How the Critic Head Learns to Detect Overconfidence

This is our **Novel Task** - training on PCam (pathology patches) to detect overconfidence:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 CRITIC HEAD TRAINING STRATEGY                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: Train on PCam (Pathology Patches)                              │
│  ─────────────────────────────────────────                              │
│  PCam has 327,680 tiny patches (96x96 pixels)                           │
│  Task: "Does this patch contain cancer cells?"                          │
│                                                                          │
│  WHY PCam?                                                               │
│  • Very subtle differences between cancer/non-cancer                    │
│  • Models often get overconfident on hard cases                         │
│  • We KNOW ground truth, so we can measure overconfidence               │
│                                                                          │
│  Training signal:                                                        │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │                                                              │        │
│  │  If model predicts 95% confidence but is WRONG:             │        │
│  │     → This is OVERCONFIDENCE → Label = 1                    │        │
│  │                                                              │        │
│  │  If model predicts 95% confidence and is RIGHT:             │        │
│  │     → This is CALIBRATED → Label = 0                        │        │
│  │                                                              │        │
│  │  If model predicts 55% confidence (uncertain):              │        │
│  │     → This is APPROPRIATE UNCERTAINTY → Label = 0           │        │
│  │                                                              │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                          │
│  STEP 2: Transfer to Radiology                                          │
│  ────────────────────────────────                                       │
│  The Critic Head learns PATTERNS of overconfidence:                     │
│  • Scattered attention (looking everywhere = unsure)                    │
│  • Low logit margin (top 2 predictions are close)                       │
│  • High entropy (probability spread across many classes)                │
│                                                                          │
│  These patterns TRANSFER to chest X-rays!                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Critic Head Architecture Details

```python
class CriticHead(nn.Module):
    """
    Detects overconfidence in the Radiologist's predictions.
    Trained on PCam to recognize patterns of miscalibration.
    """
    def __init__(self, embedding_dim=768, hidden_dim=256):
        super().__init__()
        
        # Takes MedSigLIP embeddings as input
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Auxiliary inputs from Radiologist
        self.aux_processor = nn.Sequential(
            nn.Linear(4, 32),  # logit_margin, entropy, attn_dispersion, pred_stability
            nn.ReLU(),
        )
        
        # Final overconfidence prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings, radiologist_signals):
        """
        Args:
            embeddings: MedSigLIP image embeddings [B, 768]
            radiologist_signals: dict with keys:
                - logit_margin: difference between top 2 logits
                - entropy: prediction entropy
                - attention_dispersion: Gini of attention weights
                - prediction_stability: std across dropout runs
        
        Returns:
            overconfidence_prob: probability radiologist is overconfident [B, 1]
        """
        features = self.feature_extractor(embeddings)
        
        aux_tensor = torch.stack([
            radiologist_signals['logit_margin'],
            radiologist_signals['entropy'],
            radiologist_signals['attention_dispersion'],
            radiologist_signals['prediction_stability']
        ], dim=1)
        aux_features = self.aux_processor(aux_tensor)
        
        combined = torch.cat([features, aux_features], dim=1)
        return self.classifier(combined)
```

---

## Part 3: The Uncertainty-Gated Routing System

### Why Gate by Uncertainty?

Not all cases need the full system. A clear pneumonia case doesn't need PubMed search or a 27B model.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              UNCERTAINTY-GATED ROUTING EXPLAINED                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CASE 1: Clear pneumonia (U = 0.15)                                     │
│  ───────────────────────────────────                                    │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────────────┐             │
│  │ X-ray   │───►│ Radiologist │───►│ "Pneumonia, 92%"     │             │
│  └─────────┘    │ + Critic    │    │ U = 0.15 < 0.30     │             │
│                 └─────────────┘    │ DONE! Return result │             │
│                                    └──────────────────────┘             │
│  Time: ~2 seconds                                                        │
│  Cost: $0.01 (edge device)                                              │
│                                                                          │
│                                                                          │
│  CASE 2: Unclear infiltrate (U = 0.35)                                  │
│  ──────────────────────────────────────                                 │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────────────┐             │
│  │ X-ray   │───►│ Radiologist │───►│ "Maybe pneumonia?"   │             │
│  └─────────┘    │ + Critic    │    │ U = 0.35 ≥ 0.30     │             │
│                 └─────────────┘    └──────────┬───────────┘             │
│                                               │                          │
│                                               ▼                          │
│                                    ┌──────────────────────┐             │
│                                    │ HISTORIAN AGENT      │             │
│                                    │ "Patient is diabetic │             │
│                                    │  + immunocompromised"│             │
│                                    │ → Higher risk!       │             │
│                                    │ New U = 0.28         │             │
│                                    └──────────────────────┘             │
│  Time: ~5 seconds                                                        │
│  Cost: $0.03                                                             │
│                                                                          │
│                                                                          │
│  CASE 3: Unusual pattern (U = 0.55)                                     │
│  ──────────────────────────────────                                     │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────────────┐             │
│  │ X-ray   │───►│ Radiologist │───►│ "TB? Fungal? Cancer?"│             │
│  └─────────┘    │ + Critic    │    │ U = 0.55 (HIGH!)     │             │
│                 └─────────────┘    └──────────┬───────────┘             │
│                                               │                          │
│                      ┌────────────────────────┼────────────────────┐    │
│                      ▼                        ▼                    ▼    │
│           ┌──────────────────┐    ┌──────────────────┐  ┌────────────┐ │
│           │ HISTORIAN        │    │ LITERATURE       │  │ CHIEF      │ │
│           │ "HIV+ patient"   │    │ "New TB variant  │  │ "Given all │ │
│           └────────┬─────────┘    │  in this region" │  │  evidence, │ │
│                    │              └────────┬─────────┘  │  likely TB" │ │
│                    └───────────────────────┴────────────┴────────────┘  │
│  Time: ~15 seconds                                                       │
│  Cost: $0.15 (uses 27B model)                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Routing Thresholds Summary

| Uncertainty Level | Routing Decision | % of Cases | Compute Location |
|-------------------|------------------|------------|------------------|
| **U < 0.30** | Radiologist only → Direct diagnosis | ~80% | Edge (4B) |
| **0.30 ≤ U < 0.40** | + Historian Agent → Add patient context | ~10% | Edge+ |
| **0.40 ≤ U < 0.50** | + Literature Agent → Add evidence | ~5% | Cloud (4B) |
| **U ≥ 0.50** | + Chief Orchestrator → Full council | ~5% | Cloud (27B) |

### The Math Behind Uncertainty

```python
def calculate_uncertainty(radiologist_output, critic_output):
    """
    Combine signals from both heads into single uncertainty score.
    
    The key insight: Radiologist might be confident but WRONG.
    Critic Head detects this by looking at HOW the radiologist
    made its prediction (attention patterns, logit distributions).
    """
    
    # 1. Radiologist's entropy (from token probabilities)
    # High entropy = model is unsure which tokens to generate
    token_probs = radiologist_output.token_probabilities
    radiologist_entropy = -sum(p * log(p) for p in token_probs if p > 0)
    max_entropy = log(len(token_probs))  # Maximum possible entropy
    radiologist_entropy_normalized = radiologist_entropy / max_entropy
    
    # 2. Critic's disagreement score
    # Trained to output high value when radiologist is overconfident
    critic_score = critic_output.overconfidence_probability
    
    # 3. Additional signals from radiologist
    logit_margin = radiologist_output.top_logit - radiologist_output.second_logit
    logit_margin_normalized = 1 - min(logit_margin / 10, 1)  # Invert: low margin = high uncertainty
    
    attention_weights = radiologist_output.attention_weights
    attention_dispersion = gini_coefficient(attention_weights)  # High Gini = focused attention
    attention_uncertainty = 1 - attention_dispersion  # Scattered attention = uncertain
    
    # 4. Combine into final uncertainty (weighted average)
    uncertainty = (
        0.35 * radiologist_entropy_normalized +  # What the model says
        0.35 * critic_score +                     # What the critic thinks
        0.20 * logit_margin_normalized +          # How decisive the logits are
        0.10 * attention_uncertainty              # How focused the attention is
    )
    
    return uncertainty  # Value between 0 and 1


def gini_coefficient(weights):
    """
    Gini coefficient measures inequality in attention distribution.
    High Gini (close to 1) = attention focused on few regions (confident)
    Low Gini (close to 0) = attention spread everywhere (uncertain)
    """
    sorted_weights = sorted(weights)
    n = len(sorted_weights)
    cumulative = sum((i + 1) * w for i, w in enumerate(sorted_weights))
    return (2 * cumulative) / (n * sum(sorted_weights)) - (n + 1) / n
```

---

## Part 4: The Agent Council

### Agent Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      THE SPECIALIST COUNCIL                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ RADIOLOGIST AGENT                                                │    │
│  │ ════════════════                                                 │    │
│  │ Model: MedGemma 4B + LoRA adapter                               │    │
│  │ Role: Primary visual interpretation                              │    │
│  │                                                                  │    │
│  │ Input: DICOM image                                               │    │
│  │ Output:                                                          │    │
│  │   • Structured findings ("opacity in right lower lobe")         │    │
│  │   • Differential diagnosis (["pneumonia", "atelectasis"])       │    │
│  │   • Attention maps (where did it look?)                         │    │
│  │   • Initial confidence (0.0 - 1.0)                              │    │
│  │                                                                  │    │
│  │ Trained on: CheXpert (224k), MIMIC-CXR (377k)                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ HISTORIAN AGENT                                                  │    │
│  │ ════════════════                                                 │    │
│  │ Model: MedGemma 4B + LoRA adapter (different from Radiologist)  │    │
│  │ Role: Patient context retrieval                                  │    │
│  │                                                                  │    │
│  │ Tools (via MCP):                                                 │    │
│  │   • get_patient_conditions(patient_id) → ["diabetes", "HIV"]    │    │
│  │   • get_recent_labs(patient_id) → {"WBC": 15000, ...}           │    │
│  │   • get_medications(patient_id) → ["prednisone", ...]           │    │
│  │   • get_prior_imaging(patient_id) → [previous X-rays]           │    │
│  │                                                                  │    │
│  │ Output:                                                          │    │
│  │   • Relevant clinical context summary                           │    │
│  │   • Risk factor identification                                  │    │
│  │   • "This changes the probability because..."                   │    │
│  │                                                                  │    │
│  │ Trained on: MIMIC-IV-on-FHIR (real EHR data in FHIR format)    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ LITERATURE AGENT                                                 │    │
│  │ ═════════════════                                                │    │
│  │ Model: MedGemma 4B + RAG (Retrieval Augmented Generation)       │    │
│  │ Role: Evidence retrieval from medical literature                 │    │
│  │                                                                  │    │
│  │ Tools (via MCP):                                                 │    │
│  │   • search_pubmed(query) → [relevant papers]                    │    │
│  │   • get_paper_abstract(pmid) → full abstract text               │    │
│  │   • search_clinical_trials(condition) → ongoing trials          │    │
│  │                                                                  │    │
│  │ Output:                                                          │    │
│  │   • Supporting evidence for diagnosis                           │    │
│  │   • Contradicting evidence (if any)                             │    │
│  │   • Citations with relevance scores                             │    │
│  │                                                                  │    │
│  │ Why needed: Medicine evolves! New diagnostic criteria, new      │    │
│  │ diseases (COVID was unknown to pre-2020 models)                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ CRITIC AGENT                                                     │    │
│  │ ═════════════                                                    │    │
│  │ Model: MedGemma 4B + LoRA (adversarial training)                │    │
│  │ Role: Challenge and verify hypotheses                            │    │
│  │                                                                  │    │
│  │ Behavior:                                                        │    │
│  │   • "What evidence AGAINST pneumonia exists?"                   │    │
│  │   • "Could this be something else?"                             │    │
│  │   • "Is this conclusion supported by the FHIR data?"            │    │
│  │                                                                  │    │
│  │ Output:                                                          │    │
│  │   • Counterarguments                                            │    │
│  │   • Missing evidence flags                                      │    │
│  │   • Adjusted uncertainty (often INCREASES it)                   │    │
│  │                                                                  │    │
│  │ Why needed: Prevents groupthink, ensures robustness             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ CHIEF ORCHESTRATOR                                               │    │
│  │ ═══════════════════                                              │    │
│  │ Model: MedGemma 27B (larger, more capable)                      │    │
│  │ Role: Final arbitration when agents disagree                     │    │
│  │                                                                  │    │
│  │ Only activated when:                                             │    │
│  │   • Uncertainty > 0.50                                          │    │
│  │   • Agents have conflicting conclusions                         │    │
│  │   • Safety-critical decision needed                             │    │
│  │                                                                  │    │
│  │ Output:                                                          │    │
│  │   • Final diagnosis (or explicit "defer to human")              │    │
│  │   • Reasoning trace                                             │    │
│  │   • Calibrated confidence                                       │    │
│  │                                                                  │    │
│  │ Why 27B: More reasoning capacity for complex cases              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Communication Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENT MESSAGE PASSING                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Each agent produces a structured output that others can consume:       │
│                                                                          │
│  RadilogistOutput = {                                                   │
│      "findings": [                                                      │
│          {"location": "RLL", "observation": "opacity", "severity": 0.7} │
│      ],                                                                 │
│      "differential": [                                                  │
│          {"diagnosis": "pneumonia", "probability": 0.65},               │
│          {"diagnosis": "atelectasis", "probability": 0.25}              │
│      ],                                                                 │
│      "attention_map": <tensor>,                                         │
│      "confidence": 0.72,                                                │
│      "reasoning": "Opacity pattern consistent with consolidation..."    │
│  }                                                                      │
│                                                                          │
│  HistorianOutput = {                                                    │
│      "relevant_conditions": ["diabetes", "recent_hospitalization"],     │
│      "risk_factors": ["immunocompromised"],                             │
│      "relevant_labs": {"WBC": 14500, "CRP": 85},                        │
│      "prior_imaging_comparison": "New opacity vs 2 weeks ago",          │
│      "clinical_summary": "High-risk patient with new infiltrate...",    │
│      "probability_adjustment": +0.15  # Increases pneumonia likelihood  │
│  }                                                                      │
│                                                                          │
│  LiteratureOutput = {                                                   │
│      "supporting_evidence": [                                           │
│          {"pmid": "38472615", "relevance": 0.92, "excerpt": "..."},     │
│      ],                                                                 │
│      "contradicting_evidence": [],                                      │
│      "diagnostic_criteria_met": ["fever", "productive_cough", "opacity"]│
│      "evidence_strength": "moderate"                                    │
│  }                                                                      │
│                                                                          │
│  CriticOutput = {                                                       │
│      "challenges": [                                                    │
│          "Could be viral, not bacterial - affects treatment",           │
│          "Atelectasis cannot be ruled out without lateral view"         │
│      ],                                                                 │
│      "missing_evidence": ["sputum_culture", "procalcitonin"],           │
│      "uncertainty_adjustment": +0.08,                                   │
│      "recommendation": "Consider CT if no improvement in 48h"           │
│  }                                                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: MCP (Model Context Protocol) - How Agents Access Tools

MCP is a standard protocol for giving LLMs access to external tools. Think of it like USB for AI.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MCP ARCHITECTURE                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐         ┌─────────────────────────────────────┐    │
│  │                 │         │           MCP SERVERS                │    │
│  │   MedGemma      │◄───────►│                                     │    │
│  │   Agent         │   MCP   │  ┌─────────────────────────────┐   │    │
│  │                 │  Proto  │  │ FHIR MCP Server              │   │    │
│  └─────────────────┘   col   │  │                              │   │    │
│                              │  │ Tools:                        │   │    │
│  The agent can "call"        │  │ • get_patient(id)            │   │    │
│  these tools naturally       │  │ • get_conditions(id)         │   │    │
│  in its generation:          │  │ • get_observations(id)       │   │    │
│                              │  │ • get_medications(id)        │   │    │
│  "I need to check the        │  │                              │   │    │
│   patient's history...       │  │ Connects to: Hospital FHIR   │   │    │
│   <tool>get_conditions       │  └─────────────────────────────┘   │    │
│   (patient_id)</tool>"       │                                     │    │
│                              │  ┌─────────────────────────────┐   │    │
│                              │  │ PubMed MCP Server            │   │    │
│                              │  │                              │   │    │
│                              │  │ Tools:                        │   │    │
│                              │  │ • search(query, max=10)      │   │    │
│                              │  │ • get_abstract(pmid)         │   │    │
│                              │  │ • get_full_text(pmid)        │   │    │
│                              │  │                              │   │    │
│                              │  │ Connects to: NCBI E-utils    │   │    │
│                              │  └─────────────────────────────┘   │    │
│                              │                                     │    │
│                              │  ┌─────────────────────────────┐   │    │
│                              │  │ DICOM MCP Server             │   │    │
│                              │  │                              │   │    │
│                              │  │ Tools:                        │   │    │
│                              │  │ • load_study(study_uid)      │   │    │
│                              │  │ • get_series(study_uid)      │   │    │
│                              │  │ • get_pixel_array(sop_uid)   │   │    │
│                              │  │                              │   │    │
│                              │  │ Connects to: PACS/DICOMweb   │   │    │
│                              │  └─────────────────────────────┘   │    │
│                              └─────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example MCP Server Implementation

```python
# mcp_servers/fhir_mcp/server.py

from mcp import Server, Tool
from fhirclient import client
from fhirclient.models.patient import Patient
from fhirclient.models.condition import Condition

server = Server("fhir-mcp")

@server.tool()
async def get_patient_conditions(patient_id: str) -> list[dict]:
    """
    Retrieve all active conditions for a patient from FHIR.
    
    Args:
        patient_id: The FHIR patient resource ID
    
    Returns:
        List of conditions with code, display name, and onset date
    """
    fhir_client = get_fhir_client()
    
    # Search for conditions
    search = Condition.where(struct={
        'patient': patient_id,
        'clinical-status': 'active'
    })
    conditions = search.perform_resources(fhir_client.server)
    
    return [
        {
            "code": c.code.coding[0].code,
            "display": c.code.coding[0].display,
            "onset": str(c.onsetDateTime) if c.onsetDateTime else None
        }
        for c in conditions
    ]


@server.tool()
async def get_recent_labs(patient_id: str, days: int = 30) -> dict:
    """
    Retrieve recent laboratory results for a patient.
    
    Args:
        patient_id: The FHIR patient resource ID
        days: How far back to look (default 30 days)
    
    Returns:
        Dictionary of lab name -> most recent value
    """
    # Implementation...
    pass


@server.tool()
async def get_medications(patient_id: str) -> list[dict]:
    """
    Retrieve active medications for a patient.
    """
    # Implementation...
    pass
```

---

## Part 6: The Proof Layer

Every diagnosis must be VERIFIABLE. This is crucial for medical AI.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EVIDENCE PACKET STRUCTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ EVIDENCE PACKET FOR: Patient MRN-12345                          │    │
│  │ Generated: 2026-02-01 14:32:15 UTC                              │    │
│  │ Diagnosis: Community-Acquired Pneumonia (CAP)                   │    │
│  │ Confidence: 87% | Uncertainty: 0.25                             │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │ SECTION 1: VISUAL EVIDENCE                                       │    │
│  │ ─────────────────────────                                        │    │
│  │ ┌─────────────────┐  ┌─────────────────┐                        │    │
│  │ │ Original X-ray  │  │ Grad-CAM        │                        │    │
│  │ │                 │  │ Heatmap         │                        │    │
│  │ │    [IMAGE]      │  │    [IMAGE]      │                        │    │
│  │ │                 │  │  (Red = focus)  │                        │    │
│  │ └─────────────────┘  └─────────────────┘                        │    │
│  │ Caption: "Model focused on right lower lobe opacity (red),      │    │
│  │ consistent with consolidation pattern."                         │    │
│  │                                                                  │    │
│  │ SECTION 2: CLINICAL CONTEXT (from FHIR)                         │    │
│  │ ───────────────────────────────────────                         │    │
│  │ • Active Conditions: Type 2 Diabetes (E11.9)                    │    │
│  │ • Recent Labs: WBC 14,500 (elevated), CRP 85 mg/L (elevated)   │    │
│  │ • Current Medications: Metformin 1000mg BID                     │    │
│  │ • Relevance: "Diabetes increases pneumonia risk and may         │    │
│  │   indicate need for broader antibiotic coverage."               │    │
│  │                                                                  │    │
│  │ SECTION 3: LITERATURE SUPPORT                                    │    │
│  │ ─────────────────────────────                                   │    │
│  │ [1] Smith et al. (2024) "Radiographic patterns in CAP"          │    │
│  │     PMID: 38472615 | Relevance: 0.92                            │    │
│  │     "Lobar consolidation with air bronchograms is highly        │    │
│  │     specific for bacterial pneumonia (specificity 94%)"         │    │
│  │                                                                  │    │
│  │ [2] Johnson et al. (2025) "Diabetes and pneumonia outcomes"     │    │
│  │     PMID: 39182734 | Relevance: 0.78                            │    │
│  │     "Diabetic patients show 2.3x mortality risk in CAP"         │    │
│  │                                                                  │    │
│  │ SECTION 4: REASONING TRACE                                       │    │
│  │ ──────────────────────────                                      │    │
│  │ Step 1: Radiologist identified RLL opacity (U=0.35)             │    │
│  │ Step 2: Historian found diabetes + elevated WBC (U→0.30)        │    │
│  │ Step 3: Literature confirmed pattern specificity (U→0.25)       │    │
│  │ Step 4: Critic found no contradicting evidence                  │    │
│  │ Final: CAP diagnosis with 87% confidence                        │    │
│  │                                                                  │    │
│  │ SECTION 5: UNCERTAINTY DISCLOSURE                                │    │
│  │ ─────────────────────────────                                   │    │
│  │ Remaining uncertainty (13%) attributed to:                      │    │
│  │ • Could be viral pneumonia (would change treatment)             │    │
│  │ • Suboptimal image quality in costophrenic angle                │    │
│  │ Recommendation: Correlate with sputum culture                   │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Grad-CAM Visual Explanation

```python
# proof_layer/visual_evidence.py

import torch
import numpy as np
from PIL import Image

class GradCAMExplainer:
    """
    Generate visual explanations for model predictions using Grad-CAM.
    Shows WHERE the model looked to make its decision.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The vision model (MedSigLIP encoder)
            target_layer: Which layer to extract gradients from
                         (usually the last convolutional layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific class prediction.
        
        Args:
            image: Input image tensor [1, C, H, W]
            target_class: Which class to explain (e.g., pneumonia=1)
        
        Returns:
            heatmap: Normalized heatmap same size as input image
        """
        # Forward pass
        self.model.eval()
        output = self.model(image)
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)  # ReLU to keep only positive influences
        heatmap /= torch.max(heatmap)  # Normalize to [0, 1]
        
        # Resize to match input image
        heatmap = heatmap.cpu().numpy()
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (image.shape[3], image.shape[2]))
        
        return heatmap
    
    def overlay_heatmap(self, image: Image, heatmap: np.ndarray) -> Image:
        """
        Overlay heatmap on original image for visualization.
        """
        # Convert heatmap to colormap (blue=cold, red=hot)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        image_array = np.array(image)
        if len(image_array.shape) == 2:  # Grayscale X-ray
            image_array = np.stack([image_array] * 3, axis=-1)
        
        blended = cv2.addWeighted(image_array, 0.6, heatmap_colored, 0.4, 0)
        
        return Image.fromarray(blended)
```

---

## Part 7: Training Pipeline

### What We Need to Train

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING COMPONENTS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  COMPONENT 1: Radiologist LoRA Adapter                                  │
│  ═════════════════════════════════════                                  │
│  Base model: MedGemma 4B (frozen)                                       │
│  Trainable: LoRA matrices (rank=16, ~0.1% of params)                   │
│  Dataset: CheXpert + MIMIC-CXR                                          │
│  Task: Image → Structured findings + diagnosis                          │
│                                                                          │
│  COMPONENT 2: Critic Head                                               │
│  ════════════════════════                                               │
│  Architecture: MLP on top of MedSigLIP features                         │
│  Dataset: PCam + CheXpert uncertain labels                              │
│  Task: Predict overconfidence probability                               │
│                                                                          │
│  COMPONENT 3: Historian LoRA Adapter                                    │
│  ═══════════════════════════════════                                    │
│  Base model: MedGemma 4B (frozen)                                       │
│  Trainable: Different LoRA matrices than Radiologist                   │
│  Dataset: MIMIC-IV-on-FHIR                                              │
│  Task: Given FHIR data, summarize relevant clinical context             │
│                                                                          │
│  COMPONENT 4: Literature LoRA Adapter                                   │
│  ════════════════════════════════════                                   │
│  Base model: MedGemma 4B (frozen)                                       │
│  Dataset: PubMed QA, MedQA                                              │
│  Task: Given query + retrieved papers, synthesize evidence              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Radiologist LoRA Training

```python
# training/train_radiologist_lora.py

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset
import torch

def train_radiologist_adapter():
    """
    Fine-tune MedGemma 4B with LoRA for chest X-ray interpretation.
    """
    
    # 1. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
    
    # 2. Configure LoRA
    lora_config = LoraConfig(
        r=16,                     # Rank of update matrices
        lora_alpha=32,            # Scaling factor
        target_modules=[          # Which layers to adapt
            "q_proj", "v_proj",   # Attention
            "gate_proj", "up_proj", "down_proj"  # MLP
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    print(f"Trainable parameters: {model.print_trainable_parameters()}")
    # Output: trainable params: 4,194,304 || all params: 4,000,000,000 || trainable%: 0.1%
    
    # 3. Load datasets
    chexpert = load_dataset("stanfordmlgroup/chexpert")
    mimic_cxr = load_dataset("physionet/mimic-cxr")
    
    # 4. Training format
    def format_example(example):
        """
        Convert dataset example to training format.
        """
        prompt = """<image>
        Analyze this chest X-ray. Provide:
        1. FINDINGS: Describe all visible abnormalities
        2. DIFFERENTIAL: List possible diagnoses with probabilities
        3. RECOMMENDATION: Suggest follow-up if needed
        """
        
        response = f"""FINDINGS:
        {example['findings']}
        
        DIFFERENTIAL:
        {format_differential(example['labels'])}
        
        RECOMMENDATION:
        {example['recommendation']}
        """
        
        return {
            "image": example['image'],
            "prompt": prompt,
            "response": response
        }
    
    # 5. Training loop (simplified)
    from transformers import Trainer, TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="./radiologist_lora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        bf16=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    trainer.train()
    
    # 6. Save adapter only (small file!)
    model.save_pretrained("./radiologist_lora")
    # This saves only ~16MB instead of 8GB for full model
```

### Critic Head Training

```python
# training/train_critic_head.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.medsigslip_encoder import MedSigLIPEncoder
from models.critic_head import CriticHead

def train_critic_head():
    """
    Train the Critic Head to detect overconfidence.
    
    Key insight: Train on PCam (pathology) where we can measure 
    overconfidence directly, then transfer to radiology.
    """
    
    # 1. Load PCam dataset
    pcam = load_dataset("pcam")
    
    # 2. Load frozen encoder
    encoder = MedSigLIPEncoder()
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # 3. Initialize Critic Head
    critic = CriticHead(embedding_dim=768, hidden_dim=256)
    
    # 4. First, train a simple classifier to get predictions
    # (We need predictions to know what's overconfident)
    simple_classifier = nn.Linear(768, 2)  # Cancer / No cancer
    
    # Train simple classifier first...
    # [Training code here]
    
    # 5. Now identify overconfident predictions
    def find_overconfident_samples(dataset, classifier, encoder):
        """
        Find samples where classifier is confident but wrong.
        """
        overconfident_samples = []
        
        for sample in dataset:
            image = sample['image']
            label = sample['label']
            
            # Get prediction
            with torch.no_grad():
                embedding = encoder(image)
                logits = classifier(embedding)
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs)
                confidence = probs.max().item()
            
            # Check for overconfidence
            is_wrong = (pred != label)
            is_confident = (confidence > 0.8)
            
            if is_wrong and is_confident:
                # This is overconfidence!
                overconfident_samples.append({
                    'embedding': embedding,
                    'overconfident': 1
                })
            else:
                overconfident_samples.append({
                    'embedding': embedding,
                    'overconfident': 0
                })
        
        return overconfident_samples
    
    # 6. Train Critic Head on this data
    overconfidence_data = find_overconfident_samples(pcam['train'], simple_classifier, encoder)
    
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    for epoch in range(10):
        for batch in DataLoader(overconfidence_data, batch_size=32):
            embeddings = batch['embedding']
            labels = batch['overconfident'].float()
            
            # Forward pass
            predictions = critic(embeddings)
            loss = criterion(predictions.squeeze(), labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # 7. Save trained Critic Head
    torch.save(critic.state_dict(), "./critic_head.pth")
```

---

## Part 8: Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: Build the foundation                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  □ Set up project structure                                             │
│  □ Implement MedSigLIP encoder wrapper                                  │
│  □ Implement MedGemma 4B loading with LoRA support                      │
│  □ Create DICOM loader (for X-ray input)                                │
│  □ Create basic inference pipeline                                      │
│                                                                          │
│  Deliverable: Can load X-ray → Get MedGemma response                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Dual-Head Architecture (Week 2-3)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: Build the novel component                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  □ Implement Critic Head architecture                                   │
│  □ Download and preprocess PCam dataset                                 │
│  □ Train Critic Head on PCam                                            │
│  □ Implement uncertainty calculation                                    │
│  □ Implement uncertainty-gated router                                   │
│                                                                          │
│  Deliverable: Radiologist + Critic working together with routing        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Agent Council (Week 3-4)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: Build the agent system                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  □ Implement MCP servers (FHIR, PubMed, DICOM)                          │
│  □ Implement Historian agent                                            │
│  □ Implement Literature agent                                           │
│  □ Implement Critic agent (adversarial)                                 │
│  □ Implement Chief orchestrator (27B)                                   │
│  □ Wire up LangGraph for agent coordination                             │
│                                                                          │
│  Deliverable: Full agent council working end-to-end                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Proof Layer & Demo (Week 4-5)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: Make it presentable                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  □ Implement Grad-CAM visualization                                     │
│  □ Implement PDF evidence packet generator                              │
│  □ Build Gradio/Streamlit demo UI                                       │
│  □ Run evaluation on CheXpert test set                                  │
│  □ Generate calibration plots                                           │
│  □ Record 3-minute video                                                │
│  □ Write competition writeup                                            │
│                                                                          │
│  Deliverable: Complete submission package                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 9: Why This Wins

### Competition Alignment

| Criterion | How VERIFAI Addresses It | Weight |
|-----------|-------------------------|--------|
| **Effective use of HAI-DEF** | Uses MedSigLIP, MedGemma 4B, MedGemma 27B - the full stack | 20% |
| **Problem domain** | Diagnostic uncertainty is a real clinical problem | 15% |
| **Impact potential** | 80% cost reduction, edge deployment enables rural healthcare | 15% |
| **Product feasibility** | Clear architecture, uses existing tools (LoRA, MCP, LangGraph) | 20% |
| **Execution & communication** | Multi-track submission with video, writeup, and code | 30% |

### Prize Track Alignment

| Track | How VERIFAI Qualifies | Prize |
|-------|----------------------|-------|
| **Main Track** | Full multi-agent diagnostic system with evidence generation | $30k-$10k |
| **Agentic Workflow Prize** | Uncertainty-gated routing is a sophisticated agentic pattern | $5k |
| **Edge AI Prize** | 80% of cases run on 4B model (can be quantized for mobile) | $5k |
| **Novel Task Prize** | Critic Head trained on PCam for overconfidence detection | $5k |

### Unique Selling Points

1. **Novel Task Prize**: Critic Head trained on PCam for overconfidence detection - **nobody else will do this**

2. **Agentic Workflow Prize**: Uncertainty-gated routing is a sophisticated agentic pattern that **reimagines the diagnostic workflow**

3. **Edge AI Prize**: 80% of cases run on 4B model which can be quantized for mobile deployment - **enables rural healthcare**

4. **Main Track**: Full system with evidence packets addresses a **real clinical need** (liability, trust, explainability)

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VERIFAI IN ONE PICTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                           ┌─────────────┐                               │
│                           │   X-RAY     │                               │
│                           └──────┬──────┘                               │
│                                  │                                       │
│                                  ▼                                       │
│                           ┌─────────────┐                               │
│                           │  MedSigLIP  │                               │
│                           └──────┬──────┘                               │
│                                  │                                       │
│              ┌───────────────────┼───────────────────┐                  │
│              ▼                                       ▼                  │
│     ┌─────────────────┐                   ┌─────────────────┐          │
│     │  RADIOLOGIST    │                   │   CRITIC HEAD   │          │
│     │     HEAD        │                   │                 │          │
│     │  "Pneumonia"    │                   │  "U = 0.35"     │          │
│     └────────┬────────┘                   └────────┬────────┘          │
│              │                                     │                    │
│              └──────────────────┬──────────────────┘                    │
│                                 │                                       │
│                                 ▼                                       │
│                    ┌────────────────────────┐                           │
│                    │   UNCERTAINTY ROUTER   │                           │
│                    │   U ≥ 0.30 → Agents    │                           │
│                    └───────────┬────────────┘                           │
│                                │                                        │
│         ┌──────────────────────┼──────────────────────┐                │
│         ▼                      ▼                      ▼                │
│  ┌────────────┐        ┌────────────┐        ┌────────────┐            │
│  │ HISTORIAN  │        │ LITERATURE │        │   CRITIC   │            │
│  │ (FHIR)     │        │ (PubMed)   │        │ (Adversary)│            │
│  └─────┬──────┘        └─────┬──────┘        └─────┬──────┘            │
│        │                     │                     │                    │
│        └──────────────────────┬─────────────────────┘                   │
│                               ▼                                         │
│                    ┌────────────────────────┐                           │
│                    │   CHIEF (27B, if U>0.5)│                           │
│                    └───────────┬────────────┘                           │
│                                │                                        │
│                                ▼                                        │
│                    ┌────────────────────────┐                           │
│                    │     PROOF LAYER        │                           │
│                    │  • Grad-CAM            │                           │
│                    │  • FHIR snippets       │                           │
│                    │  • PubMed citations    │                           │
│                    └───────────┬────────────┘                           │
│                                │                                        │
│                                ▼                                        │
│                    ┌────────────────────────┐                           │
│                    │   VERIFIED DIAGNOSIS   │                           │
│                    │   + Evidence PDF       │                           │
│                    └────────────────────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**VERIFAI = MedSigLIP + Dual Heads + Uncertainty Routing + Agent Council + Proof Layer**

The key insight is: **Don't just diagnose—verify the diagnosis with evidence, and know when you don't know.**

---

*Document Version: 1.0 | Last Updated: February 2, 2026*
