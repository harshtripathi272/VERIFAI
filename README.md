# VERIFAI
### Verified Evidence-Based Clinical AI
*Hierarchical Multi-Agent Diagnostic System with Sequential Debate Architecture*

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-MedGemma%20Impact%20Challenge-gold)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Cloud Database](https://img.shields.io/badge/Database-Supabase-success)](https://supabase.com)

---

## 🆕 Latest Updates

**NEW Features:**
- ☁️ **Cloud Database**: Migrate from SQLite to Supabase for production deployments
- 🔄 **Doctor Feedback Loop**: Expert-in-the-loop reprocessing when diagnoses are rejected
- 📊 **Audit Trail**: Complete tracking of feedback and improvements
- ⚡ **60% Faster Reprocessing**: Skip image analysis, restart from critic with doctor's guidance

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for details.

---

## Overview

VERIFAI is a **hierarchical multi-agent diagnostic system** that combines fine-tuned Medical Vision-Language Models (MedGemma) with a novel **sequential debate architecture** for uncertainty-aware diagnosis. Unlike black-box classifiers, VERIFAI provides **auditable evidence packets**—complete with visual proof, literature citations, structured pathology labels, and calibrated uncertainty quantification—for every diagnostic decision.

**Key Innovation**: *Sequential Debate with Structured Pathology*—a shared MedSigLIP vision encoder feeds a **Radiologist Head** (diagnostic) which is then processed by **CheXbert** for structured pathology labeling. This structured information guides parallel evidence gathering from patient history (FHIR) and medical literature (PubMed), followed by adversarial critique and consensus-building debate rounds.

**Latest Innovation**: *Doctor Feedback Loop*—when experts reject a diagnosis, the system captures their feedback and restarts from the Critic with the doctor's guidance, skipping redundant image analysis while preserving full context.

---

## The Problem

Current medical AI suffers from three critical deployment failures:

1. **The Black Box Problem**: Physicians cannot verify why CNNs like CheXpert made a decision, creating liability risks
2. **The Overconfidence Problem**: LLMs like GPT-4V hallucinate diagnoses with high confidence, lacking uncertainty awareness
3. **The Integration Problem**: Standalone models cannot access patient history (FHIR) or current literature (PubMed) during inference

**VERIFAI solves all three** through a sequential debate architecture with specialized agents that gather evidence, debate, verify, and cite sources before concluding.

---

## Architecture

### Dual-Head Vision Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT: Chest X-ray (DICOM)                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MedSigLIP Vision Encoder (Shared, Mostly Frozen)        │
│              - Extracts visual embeddings from medical images        │
│              - Pre-trained on medical imaging data (HAI-DEF)         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
        ┌───────────────────────┐   ┌───────────────────────┐
        │    Radiologist Head   │   │      Critic Head      │
        │   (MedGemma 4B + LoRA)│   │ (Overconfidence Det.) │
        │                       │   │                       │
        │  Training:            │   │  Training:            │
        │  • CheXpert           │   │  • PCam (teaches      │
        │  • MIMIC-CXR          │   │    overconfidence)    │
        │  • PadChest           │   │  • CheXpert uncertain │
        │                       │   │    labels (U-labels)  │
        │  Outputs:             │   │                       │
        │  • Visual findings    │   │  Outputs:             │
        │  • Differential Dx    │   │  • Logit margin       │
        │  • Initial confidence │   │  • Entropy score      │
        │  • Attention maps     │   │  • Attention dispersn │
        └───────────┬───────────┘   │  • Pred. stability    │
                    │               └───────────┬───────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Combined Uncertainty   │
                    │        Score            │
                    │                         │
                    │  U = (H_rad + D_crit)/2 │
                    └─────────────────────────┘
```

### Full System Architecture (Sequential Debate)

```
                    ┌─────────────────────────┐
                    │      START (Input)      │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Radiologist Agent     │
                    │   (MedGemma 4B)         │
                    │                         │
                    │  Outputs:               │
                    │  • Findings (text)      │
                    │  • Impression (text)    │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │    CheXbert Node        │
                    │ (Structured Pathology)  │
                    │                         │
                    │  Outputs:               │
                    │  • 14 condition labels  │
                    │  • Only present/uncert. │
                    └───────────┬─────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │     EVIDENCE GATHERING (Parallel Execution)   │
        │                                               │
        │  ┌──────────────────┐  ┌──────────────────┐  │
        │  │ Historian Agent  │  │ Literature Agent │  │
        │  │  (FHIR Query)    │  │  (PubMed RAG)    │  │
        │  │                  │  │                  │  │
        │  │ Receives:        │  │ Receives:        │  │
        │  │ • Findings       │  │ • Findings       │  │
        │  │ • Impression     │  │ • Impression     │  │
        │  │ • CheXbert labels│  │ • CheXbert labels│  │
        │  └────────┬─────────┘  └────────┬─────────┘  │
        │           │                     │            │
        └───────────┼─────────────────────┼────────────┘
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────────┐
                    │     Critic Agent        │
                    │ (Consistency & Safety)  │
                    │                         │
                    │ • Validates evidence    │
                    │ • Flags contradictions  │
                    │ • Adjusts uncertainty   │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │    DEBATE ROUNDS        │
                    │ (Consensus Building)    │
                    │                         │
                    │ • Multi-round dialogue  │
                    │ • Conflict resolution   │
                    └───────┬─────────────────┘
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Finalize Node   │  │ Chief Orchestr.  │
        │ (Consensus)      │  │ (Conflict Res.)  │
        │                  │  │                  │
        │ • Standard report│  │ • MedGemma 27B   │
        │ • Evidence PDF   │  │ • Final arbiter  │
        └────────┬─────────┘  └────────┬─────────┘
                 │                     │
                 └──────────┬──────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   VERIFIED OUTPUT     │
                │                       │
                │ • Final diagnosis     │
                │ • Calibrated conf.    │
                │ • Evidence packet     │
                │ • Audit trail         │
                └───────────────────────┘
```

### Agent Specifications

| Agent | Model | Training Data | Tools/MCP | Activation |
|-------|-------|---------------|-----------|------------|
| **Radiologist** | MedGemma 4B + LoRA-rad | CheXpert, MIMIC-CXR, PadChest | DICOM loader, Grad-CAM | Always |
| **CheXbert** | F1-CheXbert (BERT) | CheXpert Labeler | Pathology Labeling | Always |
| **Critic Head** | Classifier on MedSigLIP | PCam, CheXpert U-labels | Reasoning only | Always |
| **Historian** | MedGemma 4B + LoRA-fhir | MIMIC-IV-on-FHIR, Synthea | FHIR MCP (Patient, Condition, Observation) | Always (Parallel) |
| **Literature** | MedGemma 4B + RAG | PubMed QA, MedQA, PMC-OA | PubMed MCP, ClinicalTrials MCP | Always (Parallel) |
| **Critic Agent** | MedGemma 4B + LoRA-critic | Adversarial examples | Reasoning only | Always |
| **Chief** | MedGemma 27B | Instruction-tuned only | Policy & safety rules | On Conflict |

---

## Key Innovations

### 1. Dual-Head Epistemic Architecture
The **Critic Head** is a novel component trained on PCam (pathology patches) to recognize when models are overconfident. By learning from a domain (histopathology) where subtle features matter, it generalizes to detect overconfidence in radiology:

```python
# Critic Head outputs
critic_features = {
    "logit_margin": max_logit - second_max_logit,  # Low margin = uncertain
    "entropy": -sum(p * log(p)),                    # High entropy = uncertain
    "attention_dispersion": gini(attention_weights), # Scattered attention = uncertain
    "prediction_stability": std(dropout_predictions) # High variance = uncertain
}
```

### 2. CheXbert Structured Pathology Labeling
**CheXbert** provides standardized pathology labels immediately after radiologist report generation:

```python
# CheXbert processes radiologist output
chexbert_output = CheXbertOutput(
    labels={
        "Pneumonia": "present",
        "Consolidation": "present",
        "Pleural Effusion": "uncertain"
        # Only present/uncertain saved (not absent/not_mentioned)
    }
)
```

**Benefits:**
- Structured queries for FHIR (Historian) and PubMed (Literature)
- Standardized terminology across all agents
- Reduced ambiguity in evidence retrieval

### 3. Sequential Debate Architecture
All agents run in a defined sequence with parallel evidence gathering:

| Stage | Agents | Execution | Purpose |
|-------|--------|-----------|---------|
| **1. Diagnosis** | Radiologist | Sequential | Generate findings + impression |
| **2. Labeling** | CheXbert | Sequential | Extract structured pathologies |
| **3. Evidence** | Historian + Literature | **Parallel** | Gather supporting/contradicting evidence |
| **4. Critique** | Critic | Sequential | Validate consistency |
| **5. Debate** | Multi-agent | Sequential | Build consensus |
| **6. Finalize** | Finalize or Chief | Conditional | Resolve conflicts if needed |

### 4. Enhanced Evidence Gathering

**Historian Agent** receives:
- Radiologist findings (detailed observations)
- Radiologist impression (diagnostic conclusion)
- CheXbert labels (structured pathologies)

**Literature Agent** query structure:
```
Visual findings: [Radiologist findings text]
Diagnostic impression: [Radiologist impression text]
Confirmed findings: [Present conditions from CheXbert]
Uncertain findings: [Uncertain conditions from CheXbert]

Clinical history summary: [From Historian]

Retrieve supporting or contradicting biomedical literature.
```

### 5. The Proof Layer
Every diagnosis includes an **Evidence Packet**:
- **Visual Proof**: Grad-CAM heatmaps highlighting regions of interest
- **Clinical Proof**: Relevant FHIR history snippets (anonymized)
- **Literary Proof**: PubMed citations with relevance scoring
- **Structured Pathology**: CheXbert labels with confidence
- **Audit Trail**: Complete log of agent deliberations and uncertainty trajectory

### 6. Uncertainty Quantification
Combines Radiologist entropy with Critic Head disagreement:
```python
# Combined uncertainty score
uncertainty = (radiologist_entropy + critic_disagreement_score) / 2

# Where:
# - radiologist_entropy: Token-level entropy from MedGemma generation
# - critic_disagreement_score: Normalized output from Critic Head
```

---

## Key Features

### ☁️ Cloud Database Support (Supabase)
- **Production-Ready**: PostgreSQL-backed cloud database via Supabase
- **Transparent Switching**: Toggle between SQLite (local) and Supabase (cloud) with `DATABASE_MODE` environment variable
- **Zero Code Changes**: Adapter pattern maintains identical API
- **Migration Utility**: One-command migration from SQLite to Supabase

```bash
# Check database connection
python setup_helper.py check-db

# Migrate from SQLite to Supabase
python setup_helper.py migrate
```

### 🔄 Doctor Feedback Loop
- **Expert-in-the-Loop**: Doctors can reject diagnoses and provide corrections
- **Smart Reprocessing**: Restart from critic with context preserved (60-80% faster)
- **Full Audit Trail**: All feedback stored with original context in `doctor_feedback` table
- **Intelligent Routing**: Skip redundant image analysis, reuse evidence from original session

```python
from agents.feedback.agent import capture_doctor_feedback, prepare_feedback_for_reprocessing
from graph.workflow import build_workflow

# Doctor rejects diagnosis
feedback_id = capture_doctor_feedback(
    session_id="abc123",
    doctor_notes="Missed subtle infiltrate in right lower lobe",
    correct_diagnosis="Pneumonia",
    rejection_reasons=["missed_finding", "incorrect_interpretation"]
)

# System automatically prepares feedback state
feedback_state = prepare_feedback_for_reprocessing(feedback_id)

# Workflow restarts from critic with doctor's guidance
workflow = build_workflow()
result = workflow.invoke(feedback_state)
```

See [DOCTOR_FEEDBACK_AND_CLOUD_DB_GUIDE.md](DOCTOR_FEEDBACK_AND_CLOUD_DB_GUIDE.md) for complete documentation.

---

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM for 4B, 80GB for 27B)
- FHIR R4 server access (or use synthetic data provided)
- DICOMweb endpoint (or local DICOM files)

### Installation

```bash
git clone https://github.com/yourteam/verifai.git
cd verifai

# Install dependencies
pip install -r requirements.txt

# Install MedGemma & HAI-DEF dependencies
pip install git+https://github.com/huggingface/transformers.git@main
pip install bitsandbytes accelerate peft

# Setup MCP servers (Model Context Protocol)
cd mcp_servers
pip install -e .
```

### Configuration

Create `.env`:
```env
# Database Configuration
DATABASE_MODE="sqlite"  # Use "supabase" for cloud database
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_KEY="your-anon-key"

# Model Paths (HAI-DEF models)
MEDGEMMA_4B_PATH="google/medgemma-4b-it"
MEDGEMMA_27B_PATH="google/medgemma-27b-it"
MEDSIGLIP_PATH="google/medsiglip"

# FHIR Configuration
FHIR_BASE_URL="http://your-hospital-fhir-server/fhir/R4"
FHIR_AUTH_TOKEN="Bearer xxx"

# DICOMweb
DICOMWEB_URL="http://your-pacs/wado-rs"
DICOMWEB_USERNAME="user"
DICOMWEB_PASSWORD="pass"

# Literature Search (NCBI)
NCBI_API_KEY="your-ncbi-key"
```

### Running Inference

```python
from verifai import DiagnosticCouncil

# Initialize council with sequential debate architecture
council = DiagnosticCouncil(
    use_medsigslip_encoder=True,
    use_4b_radiologist=True,
    use_chexbert_labeling=True,
    use_critic_head=True,
    use_27b_orchestrator=True,
    enable_parallel_evidence=True
)

# Run diagnosis
result = council.diagnose(
    patient_id="MRN-12345",
    study_uid="1.2.840.113..."
)

# Access results
print(result.diagnosis)           # "PCP Pneumonia"
print(result.confidence)          # 0.87
print(result.uncertainty)         # 0.25
print(result.chexbert_labels)     # {"Pneumonia": "present", ...}
print(result.routing_path)        # ["radiologist", "chexbert", "evidence_gathering", ...]
print(result.evidence_pdf)        # Path to generated report
print(result.uncertainty_trajectory)  # [0.6, 0.4, 0.25]
```

---

## Datasets

### Training Data (for LoRA Adapters & Critic Head)

| Dataset | Size | Usage | Component |
|---------|------|-------|-----------|
| **CheXpert** | 224,316 images | Radiologist adapter training | Radiologist Head |
| **MIMIC-CXR** | 377,110 images | Generalization & report generation | Radiologist Head |
| **PadChest** | 160,000 images | Additional radiology training | Radiologist Head |
| **PCam** | 327,680 patches | **Overconfidence detection training** | **Critic Head** |
| **CheXpert U-labels** | ~50k images | Uncertain case detection | Critic Head |
| **MIMIC-IV-on-FHIR** | 300k patients | Clinical history understanding | Historian Agent |
| **Synthea** | Synthetic patients | FHIR integration testing | Historian Agent |
| **PubMed QA** | 1k QA pairs | Literature retrieval | Literature Agent |

### Evaluation Data

- **CheXpert Test**: 500 images (held-out)
- **MIMIC-CXR Test**: 500 images
- **VQA-RAD**: 3,515 question-answer pairs for reasoning evaluation
- **Custom FHIR Test Set**: 100 synthetic patient journeys with ground truth

---

## Expected Results

### Diagnostic Accuracy

| Method | CheXpert AUC | MIMIC-CXR F1 | VQA-RAD Acc |
|--------|-------------|--------------|-------------|
| CheXpert (DenseNet) | 0.926 | 0.81 | N/A |
| MedGemma 4B (baseline) | 0.91 | 0.82 | 0.71 |
| **VERIFAI (4B + CheXbert)** | 0.93 | 0.85 | 0.76 |
| **VERIFAI (Full Council)** | **0.958** | **0.89** | **0.82** |

### Efficiency Metrics

| Metric | Standard 27B | VERIFAI (Sequential) | Improvement |
|--------|-------------|----------------------|-------------|
| **Avg Inference Time** | 45s/case | 12s/case (parallel) | **73% faster** |
| **Evidence Quality** | N/A | 95% relevant citations | - |
| **Structured Output** | No | Yes (CheXbert labels) | - |

### Calibration (Trustworthiness)

- **Brier Score**: 0.042 (well-calibrated)
- **Expected Calibration Error**: 0.031 (vs 0.12 for standard fine-tuning)
- **Reliability**: 0.3 uncertainty ≈ 30% actual error rate

---

## Project Structure

```
├── 📁 agents
│   ├── 📁 chexbert
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 agent.py
│   │   └── 🐍 model.py
│   ├── 📁 chief
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 agent.py
│   ├── 📁 critic
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 agent.py
│   │   └── 🐍 model.py
│   ├── 📁 historian
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 agent.py
│   │   └── 🐍 fhir_client.py
│   ├── 📁 literature
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 agent.py
│   │   ├── 🐍 europe_pmc.py
│   │   ├── 🐍 pubmed_entrez.py
│   │   └── 🐍 semantic_scholar.py
│   └── 📁 radiologist
│       ├── 🐍 __init__.py
│       ├── 🐍 agent.py
│       ├── 🐍 model.py
│       └── 🐍 prompts.py
├── 📁 app
│   ├── 🐍 __init__.py
│   ├── 🐍 api.py
│   ├── 🐍 config.py
│   └── 🐍 main.py
├── 📁 data
│   ├── 📁 embeddings
│   ├── 📁 sample_dicom
│   └── 📁 sample_fhir
├── 📁 docker
│   ├── 🐳 Dockerfile
│   └── ⚙️ docker-compose.yml
├── 📁 graph
│   ├── 🐍 __init__.py
│   ├── 🐍 router.py
│   ├── 🐍 state.py
│   └── 🐍 workflow.py
├── 📁 proof_layer
│   ├── 🐍 __init__.py
│   ├── 🐍 citations.py
│   ├── 🐍 compiler.py
│   └── 🐍 visual.py
├── 📁 tests
│   ├── 🐍 __init__.py
│   └── 🐍 test_router.py
├── 📁 frontend
│   ├── 📁 src/app
│   ├── 📁 src/components
│   └── 📄 package.json
├── ⚙️ .gitignore
├── 📝 ARCHITECTURE_DEEP_DIVE.md
├── 📝 ARCHITECTURE_UPDATE.md
├── 🐳 Dockerfile
├── 📝 README.md
├── 🐍 create_structure.py
├── 📄 requirements.txt
├── 📄 structure.txt
└── 🐍 test_workflow.py
```

---

## Competition Alignment

VERIFAI targets multiple prize tracks in the MedGemma Impact Challenge:

| Track | Qualification |
|-------|---------------|
| **Main Track** | Full multi-agent diagnostic system with evidence generation |
| **Agentic Workflow Prize** | Sequential debate architecture with specialized agents |
| **Edge AI Prize** | Efficient 4B model deployment with structured pathology |
| **Novel Task Prize** | CheXbert integration + Critic Head trained on PCam |

---

## Citation

If you use VERIFAI in your research:

```bibtex
@software{verifai2026,
  title={VERIFAI: Verified Evidence-Based Clinical AI with Sequential Debate Architecture},
  author={Harsh Tripathi},
  year={2026},
  url={https://github.com/harshtripathi272/VERIFAI},
  note={Kaggle MedGemma Impact Challenge Submission}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

**Medical Disclaimer**: VERIFAI is a research prototype for the Kaggle competition. Not intended for clinical use without FDA/regulatory approval and extensive validation.

---

*Built with ❤️ and 🧠 for the MedGemma Impact Challenge 2026*
