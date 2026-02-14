# VERIFAI Architecture Update - CheXbert Integration

## Updated Workflow (Sequential Debate Architecture)

The VERIFAI system now follows a **sequential debate workflow** with structured pathology labeling:

```
START
  ↓
┌─────────────────────────┐
│   Radiologist Agent     │  ← Generates findings + impression
│   (MedGemma 4B)         │
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│    CheXbert Node        │  ← Labels 14 pathology conditions
│ (Structured Pathology)  │     (Saves only present/uncertain)
└───────────┬─────────────┘
            ↓
┌─────────────────────────────────────────┐
│   EVIDENCE GATHERING (Parallel)        │
│                                         │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │  Historian   │  │   Literature    │ │
│  │  (FHIR)      │  │   (PubMed)      │ │
│  └──────────────┘  └─────────────────┘ │
└───────────────┬─────────────────────────┘
                ↓
┌─────────────────────────┐
│     Critic Agent        │  ← Validates consistency
└───────────┬─────────────┘
            ↓
┌─────────────────────────┐
│    DEBATE ROUNDS        │  ← Consensus building
└───────────┬─────────────┘
            ↓
      ┌─────┴─────┐
      ↓           ↓
┌──────────┐  ┌──────────┐
│ Finalize │  │  Chief   │  ← Conflict resolution
└──────────┘  └──────────┘
      ↓           ↓
      └─────┬─────┘
            ↓
    VERIFIED OUTPUT
```

## Key Changes from Previous Architecture

### 1. **CheXbert Integration**
- **Position**: Immediately after Radiologist, before Evidence Gathering
- **Input**: Radiologist's findings + impression (merged text)
- **Output**: Dictionary of pathology labels (only "present" and "uncertain")
- **Purpose**: Provides structured pathology information to downstream agents

### 2. **Sequential Flow (Not Router-Based)**
- **Old**: Uncertainty-gated router decided which agents to invoke
- **New**: All agents run in sequence, with parallel evidence gathering
- **Benefit**: More comprehensive analysis, simpler debugging

### 3. **Enhanced Data Flow**

**Historian Agent** now receives:
- `radiologist_output.findings` (detailed observations)
- `radiologist_output.impression` (diagnostic conclusion)
- `chexbert_output.labels` (structured pathologies)

**Literature Agent** now receives:
- `radiologist_output.findings`
- `radiologist_output.impression`
- `chexbert_output.labels` (separated into confirmed vs uncertain)

### 4. **Improved Prompts**

**Literature Agent Query Structure:**
```
Visual findings: [Radiologist findings text]
Diagnostic impression: [Radiologist impression text]
Confirmed findings: [Present conditions from CheXbert]
Uncertain findings: [Uncertain conditions from CheXbert]

Clinical history summary: [From Historian]

Retrieve supporting or contradicting biomedical literature.
```

## Agent Specifications (Updated)

| Agent | Model | Input | Output | Always Runs |
|-------|-------|-------|--------|-------------|
| **Radiologist** | MedGemma 4B | X-ray image | Findings + Impression | ✅ |
| **CheXbert** | F1-CheXbert (BERT) | Findings + Impression | 14 pathology labels | ✅ |
| **Historian** | MedGemma 4B | Findings + Impression + Labels | FHIR evidence | ✅ |
| **Literature** | MedGemma 4B + RAG | Findings + Impression + Labels | PubMed citations | ✅ |
| **Critic** | MedGemma 4B | All agent outputs | Consistency check | ✅ |
| **Debate** | Multi-agent | Critic output | Consensus/Conflict | ✅ |
| **Chief** | MedGemma 27B | Debate output | Final resolution | On Conflict |

## CheXbert Output Format

```python
CheXbertOutput(
    labels={
        "Pneumonia": "present",
        "Consolidation": "present",
        "Pleural Effusion": "uncertain"
    }
)
```

**What's NOT saved:**
- Absent conditions
- Not mentioned conditions
- Extra metadata fields

**What IS saved:**
- Only conditions marked as "present" or "uncertain"
- Clean dictionary format for easy downstream access

## Validation Logic (Updated)

All agents now validate **both** findings and impression:

```python
if not rad_output:
    return error("No radiologist output")
    
if not rad_output.impression or not rad_output.findings:
    return error("Missing findings or impression")
```

This ensures complete data is available for all downstream processing.

## Installation Requirements

```bash
# Add to requirements.txt
f1chexbert
```

## Benefits of This Architecture

1. **Structured Pathology**: CheXbert provides standardized labels for precise queries
2. **Comprehensive Evidence**: Both Historian and Literature get rich, multi-source input
3. **Better Debugging**: Sequential flow is easier to trace and debug
4. **Consistent Validation**: All agents check for complete radiologist output
5. **Cleaner Prompts**: Separate "confirmed" vs "uncertain" findings in literature search

---

**Note**: The main README.md has been updated with:
- ✅ CheXbert added to Agent Specifications table
- ✅ CheXbert directory added to Project Structure
- ⏳ Architecture diagram update pending (see this document for current flow)
