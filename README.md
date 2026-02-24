<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.5+-orange?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/CUDA-12.x-green?logo=nvidia" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Next.js-15-black?logo=nextdotjs" alt="Next.js"/>
  <img src="https://img.shields.io/badge/MedGemma-4B--IT-4285F4?logo=google" alt="MedGemma"/>
  <img src="https://img.shields.io/badge/LangGraph-0.2-purple" alt="LangGraph"/>
</p>

# VERIFAI — Verified Evidence-based Radiology Interpretive Framework with Agentic Intelligence

> **A multi-agent AI system for chest X-ray interpretation that produces clinically trustworthy, evidence-backed diagnoses with built-in safety guardrails, adversarial debate, and human-in-the-loop review.**

VERIFAI orchestrates **multiple specialized AI agents** through a LangGraph state machine to analyze chest X-rays, cross-reference medical literature, debate diagnostic uncertainty, and produce FDA-traceable diagnoses — all runnable on a **single consumer GPU (12 GB+ VRAM)**.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Agent Pipeline](#agent-pipeline)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Running the System](#running-the-system)
- [Fine-Tuning MedGemma](#fine-tuning-medgemma)
- [Dataset Preparation](#dataset-preparation)
- [Building the FHIR + FAISS Retrieval Index](#building-the-fhir--faiss-retrieval-index)
- [Frontend Dashboard](#frontend-dashboard)

- [Observability & Monitoring](#observability--monitoring)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Configuration Reference](#configuration-reference)
- [Technical Design Decisions](#technical-design-decisions)
- [License](#license)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VERIFAI Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────┐   ┌──────────┐   ┌────────────────────────────────┐ │
│  │ Chest     │──▶│ MedGemma │──▶│ CheXbert Structured Labeling   │ │
│  │ X-ray     │   │ 4B (4bit)│   │ (14 CXR Pathology Labels)      │ │
│  └───────────┘   └──────────┘   └─────────────┬──────────────────┘ │
│                                                │                    │
│                    ┌───────────────────────────┬┘                   │
│                    ▼                           ▼                    │
│  ┌─────────────────────────┐   ┌─────────────────────────────────┐ │
│  │ Historian Agent (FHIR)  │   │ Literature Agent (PubMed/PMC)   │ │
│  │ DuckDB + FAISS Vector   │   │ BioPython E-Utilities + ReAct   │ │
│  └────────────┬────────────┘   └──────────────┬──────────────────┘ │
│               │                                │                    │
│               ▼                                ▼                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │            Critic Agent (Adversarial Verification)           │   │
│  │   MUC Uncertainty + Hedging Analysis + Past Mistakes FAISS   │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         Multi-Agent Debate (Up to 3 Rounds)                  │   │
│  │    Critic ◄──► Historian ◄──► Literature → Consensus         │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │    Validator (MedSigLIP Similarity + Clinical Rules)         │   │
│  │    → FINALIZE / FINALIZE_LOW_CONFIDENCE / FLAG_FOR_HUMAN     │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │    Finalize + Reproducibility Hash (SHA-256, FDA 21 CFR 11)  │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │    Human-in-the-Loop Review (LangGraph Interrupt)            │   │
│  │    Doctor Approve / Reject + Feedback → Re-process           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Cross-Cutting: SSE Streaming │ Safety Guardrails │ Metrics  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agent Pipeline

| # | Agent | Role | Model/Tool | Output |
|---|-------|------|-----------|--------|
| 1 | **Radiologist** | Analyze CXR image, generate findings + impression | MedGemma 4B-IT + MedSigLIP | `RadiologistOutput` (findings, impression, disease probabilities, heatmaps) |
| 2 | **CheXbert** | Extract structured pathology labels from report text | CheXbert BERT | `CheXbertOutput` (14 CXR condition labels) |
| 3 | **Historian** | Retrieve patient history from EHR/FHIR records | DuckDB + FAISS vector search | `HistorianOutput` (supporting/contradicting facts) |
| 4 | **Literature** | Search PubMed/Europe PMC for evidence | BioPython + Semantic Scholar API | `LiteratureOutput` (citations, evidence strength) |
| 5 | **Critic** | Adversarial verification — detect overconfidence, check past mistakes | MUC Uncertainty + SentenceTransformers | `CriticOutput` (safety score, concern flags) |
| 6 | **Debate** | Multi-round debate between Critic, Historian, and Literature | LangGraph orchestration | `DebateOutput` (consensus, confidence adjustment) |
| 7 | **Validator** | Final quality gate — visual similarity + clinical rules | MedSigLIP cosine similarity | Recommendation: `FINALIZE` / `FLAG_FOR_HUMAN` |
| 8 | **Finalize** | Build final diagnosis with reproducibility hash | SHA-256, Pydantic | `FinalDiagnosis` (diagnosis, confidence, hash) |
| 9 | **Human Review** | Doctor approves/rejects; rejected cases re-enter at Critic | LangGraph `interrupt()` | Approve/Reject + Feedback loop |

---

## Key Features

### Core
- **Multi-Agent Orchestration** — Multiple agents coordinated via LangGraph state machine with typed state
- **Multi-View Support** — Accepts multiple X-ray views (AP, PA, Lateral) simultaneously
- **MUC (Monotonic Uncertainty Cascade)** — Bidirectional Information Gain per agent; each agent either confirms (↓ uncertainty) or contradicts (↑ uncertainty) the current diagnosis via log-odds updates. Dempster-Shafer evidence fusion during debate rounds. Grounded in 5 research papers (ICML 2026, ACL 2024, MICCAI 2024, Shafer 1976, arXiv:2601.15703). See [`docs/MUC_DESIGN.md`](docs/MUC_DESIGN.md) for full derivation.
- **Multi-Agent Debate** — Up to 3 rounds of adversarial debate before consensus
- **Human-in-the-Loop** — LangGraph interrupt-based doctor review with feedback reprocessing

### Safety & Trust
- **Medical Safety Guardrails** — Rule-based + embedding-based checks for dangerous hallucinations
- **Reproducibility Hash** — SHA-256 fingerprint (image + FHIR + config) for FDA 21 CFR Part 11 audit trail
- **Past Mistakes Memory** — FAISS-indexed historical errors inform future diagnoses via Critic
- **CheXbert Cross-Validation** — Structured labels validate free-text radiologist findings

### Infrastructure
- **SSE Real-Time Streaming** — Server-Sent Events for live agent progress to frontend
- **Observability Dashboard** — Prometheus-style metrics (latency, confidence, safety scores)
- **Evidence Report Generator** — Rich HTML reports with citations, heatmaps, and audit trail
- **Edge Deployable** — Runs on a single consumer GPU (12 GB+ VRAM) with optional 4-bit quantization

---

## Project Structure

```
VERIFAI/
├── agents/                      # AI Agent implementations
│   ├── radiologist/             # MedGemma 4B vision + MedSigLIP heatmaps
│   │   ├── model.py             # Model loading (4-bit quantized)
│   │   ├── agent.py             # Radiologist agent logic
│   │   ├── classifier.py        # Disease probability classifier
│   │   └── interpretability.py  # Grad-CAM / attention heatmaps
│   ├── chexbert/                # CheXbert structured labeling
│   │   └── agent.py             # Extract 14 CXR condition labels
│   ├── historian/               # FHIR patient history retrieval
│   │   ├── agent.py             # DuckDB + FAISS vector search
│   │   └── fhir_retriever.py    # FHIR R4 resource parser
│   ├── literature/              # PubMed / Europe PMC search
│   │   ├── agent.py             # Literature search orchestrator
│   │   ├── pubmed_tool.py       # BioPython E-Utilities wrapper
│   │   ├── europepmc_tool.py    # Europe PMC REST API
│   │   └── semantic_scholar.py  # Semantic Scholar API
│   ├── critic/                  # Adversarial verification
│   │   ├── agent.py             # Overconfidence detection + past mistakes
│   │   └── past_mistakes.py     # FAISS-indexed error memory
│   ├── debate/                  # Multi-agent debate protocol
│   │   └── agent.py             # 3-round structured debate
│   ├── validator/               # Final quality gate
│   │   └── agent.py             # MedSigLIP similarity + clinical rules
│   └── feedback/                # Doctor feedback handler
│       └── agent.py             # Process rejection → re-enter pipeline
│
├── graph/                       # LangGraph workflow
│   ├── state.py                 # VerifaiState TypedDict + Pydantic models
│   ├── workflow.py              # Full graph definition + node wrappers
│   └── router.py                # Uncertainty-based routing logic
│
├── app/                         # FastAPI backend
│   ├── main.py                  # App entry point + middleware
│   ├── api.py                   # REST endpoints (start, status, resume)
│   ├── config.py                # Settings (models, thresholds, flags)
│   └── streaming.py             # SSE event bus + streaming endpoint
│
├── frontend/                    # Next.js 15 dashboard
│   ├── src/app/
│   │   ├── diagnose/page.tsx    # Upload X-ray + start workflow
│   │   ├── results/[id]/page.tsx # Live results + SSE feed + tabs
│   │   └── observability/page.tsx # Metrics dashboard
│   └── src/lib/api.ts           # TypeScript API client
│
├── monitoring/                  # Observability layer
│   └── metrics.py               # Prometheus-style counters + histograms
│
├── safety/                      # Medical safety guardrails
│   └── guardrails.py            # Critical finding detection + hallucination checks
│
├── uncertainty/                 # MUC framework
│   ├── muc.py                   # Multi-agent Uncertainty Calibration (Information Gain)
│   ├── kle.py                   # Semantic uncertainty computation
│   └── case_embedding.py        # Case-level embedding for similarity
│
├── output/                      # Report generation
│   └── evidence_report.py       # HTML evidence report builder
│

├── db/                          # Database adapters
│   ├── logger.py                # Session-scoped structured logging
│   ├── supabase_logger.py       # Cloud database adapter
│   └── sqlite_logger.py         # Local SQLite adapter
│
├── scripts/                     # Utility scripts
│   ├── build_retrieval_index.py # Build FHIR FAISS index
│   ├── seed_pb.py               # Seed past mistakes database
│   └── install_radgraph_model.py# Install RadGraph model
│
├── qlora_medgemma.py            # QLoRA fine-tuning script for MedGemma
├── fine_tune_hugging_face.py    # HuggingFace Trainer fine-tuning
├── train_classifier.py          # Disease classifier training
├── extract_fhir_to_duckdb.py   # FHIR bundle → DuckDB ETL
│
├── tests/                       # Test suite
│   └── test_workflow.py         # End-to-end workflow test
│
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── Dockerfile                   # Container build
└── README.md                    # This file
```

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.10 |
| **CUDA** | 12.1 | 12.6 |
| **GPU VRAM** | 12 GB | 24+ GB |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 20 GB (models) | 50 GB |
| **Node.js** | 18 | 20+ |
| **OS** | Windows 10 / Ubuntu 22.04 | Any |

**Required accounts:**
- [Hugging Face](https://huggingface.co/settings/tokens) — Token for gated models (MedGemma, MedSigLIP)
- [NCBI](https://www.ncbi.nlm.nih.gov/account/settings/) — API key for PubMed access
- (Optional) [Supabase](https://supabase.com) — For cloud database logging

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/VERIFAI.git
cd VERIFAI
```

### 2. Create Python Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install PyTorch with CUDA

```bash
# CUDA 12.1+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 6. Install RadGraph Model (for NLP validation)

```bash
python scripts/install_radgraph_model.py
```

### 7. Install Frontend

```bash
cd frontend
npm install
cd ..
```

---

## Environment Configuration

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` with your actual values:

```env
# ── Required ──
HUGGINGFACE_TOKEN=hf_your_token_here    # MedGemma/MedSigLIP access
NCBI_EMAIL=your.email@example.com       # PubMed policy requirement

# ── Recommended ──
NCBI_API_KEY=your_ncbi_key              # 10 req/s vs 3 req/s
MOCK_MODELS=False                       # True = skip model download, use mocks

# ── Database ──
DATABASE_MODE=sqlite                    # sqlite (local) or supabase (cloud)

# ── Optional: Supabase Cloud DB ──
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_key

# ── Optional: Semantic Scholar ──
SEMANTIC_SCHOLAR_API_KEY=your_key
```

> **First run with `MOCK_MODELS=True`** to verify everything works before downloading ~20 GB of model weights.

---

## Running the System

### Quick Start (Mock Mode — No GPU Required)

```bash
# 1. Set MOCK_MODELS=True in .env
# 2. Start backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. In another terminal, start frontend
cd frontend
npm run dev
# → Open http://localhost:3000
```

### Production Mode (Real Models — GPU Required)

```bash
# 1. Set MOCK_MODELS=False in .env
# 2. Ensure HUGGINGFACE_TOKEN is set (for gated model download)

# 3. Start backend (first run downloads models ~20 GB)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. Monitor GPU usage
nvidia-smi -l 1

# 5. In another terminal, start frontend
cd frontend
npm run dev
```

### Running via Test Script (No Frontend)

```bash
python tests/test_workflow.py
```

This runs the full 7-agent pipeline on a test image and prints:
- Radiologist findings and impression
- CheXbert structured labels
- Critic safety assessment
- Debate consensus
- Final diagnosis with confidence
- Reproducibility hash
- Full audit trace

### API-Only Mode (cURL)

```bash
# Start workflow
curl -X POST http://localhost:8000/api/v1/workflows/start \
  -F "images=@img1.jpg" \
  -F "views=AP"

# Response: {"session_id": "abc-123", ...}

# Poll status
curl http://localhost:8000/api/v1/workflows/abc-123/status

# SSE live stream
curl -N http://localhost:8000/api/v1/workflows/abc-123/stream
```

---

## Fine-Tuning MedGemma

VERIFAI includes two fine-tuning scripts for adapting MedGemma to your specific dataset:

### Option A: QLoRA Fine-Tuning (Recommended)

Memory-efficient fine-tuning using 4-bit quantization + Low-Rank Adaptation.

```bash
python qlora_medgemma.py \
  --dataset_path ../dataset/med/official_data_iccv_final \
  --output_dir ../dataset/med/fine_tuned_model/v1 \
  --num_epochs 3 \
  --batch_size 2 \
  --learning_rate 2e-4 \
  --lora_rank 16 \
  --lora_alpha 32
```

**Requirements:** 6 GB VRAM minimum (4-bit base + LoRA adapters)

**Dataset format:** Directory containing subdirectories per patient, each with:
- `study1/` containing `.jpg` chest X-ray images
- `findings.txt` and `impression.txt` (ground truth)

### Option B: HuggingFace Trainer Fine-Tuning

Full-featured training with HuggingFace's `Trainer` API:

```bash
python fine_tune_hugging_face.py \
  --model_name google/medgemma-1.5-4b-it \
  --dataset_path ../dataset/med/official_data_iccv_final \
  --output_dir ../dataset/med/fine_tuned_model/v2
```

### Training the Disease Classifier

Train a custom chest X-ray disease classifier on CheXpert/NIH labels:

```bash
python train_classifier.py \
  --data_dir ../dataset/med/official_data_iccv_final \
  --output_dir models/classifier
```

### Using Fine-Tuned Weights

After training, point VERIFAI to your adapters:

```env
# .env
MEDGEMMA_LORA_ROOT=../dataset/med/fine_tuned_model/v1/
MEDGEMMA_LORA_ADAPTERS=../dataset/med/fine_tuned_model/v1/
```

Then uncomment line 81 in `agents/radiologist/model.py`:

```python
_llm = PeftModel.from_pretrained(_llm, settings.MEDGEMMA_LORA_ADAPTERS)
```

---

## Dataset Preparation

### Supported Datasets

| Dataset | Size | Access |
|---------|------|--------|
| **CheXpert** | 224K images | [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) |
| **MIMIC-CXR** | 377K images | [PhysioNet](https://physionet.org/content/mimic-cxr/) |
| **NIH ChestX-ray14** | 112K images | [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| **PadChest** | 160K images | [BIMCV](https://bimcv.cipf.es/bimcv-projects/padchest/) |

### Quick Test Images

Two test images are included in the repository root:

```
img1.jpg  — For quick testing
img2.jpg  — For quick testing
```

---

## Building the FHIR + FAISS Retrieval Index

The Historian agent uses a FAISS vector index over FHIR patient records for semantic retrieval.

### Step 1: Extract FHIR Bundles to DuckDB

```bash
python extract_fhir_to_duckdb.py \
  --fhir_dir path/to/fhir/bundles \
  --output verifai_fhir.duckdb
```

### Step 2: Build FAISS Index

```bash
python scripts/build_retrieval_index.py \
  --duckdb_path verifai_fhir.duckdb \
  --output_faiss verifai_fhir.faiss \
  --output_mapping verifai_fhir_mapping.json
```

This creates:
- `verifai_fhir.faiss` — Vector index for fast similarity search
- `verifai_fhir_mapping.json` — Maps FAISS IDs to FHIR resource IDs
- `verifai_fhir.duckdb` — Structured patient data for SQL queries

### Step 3: Seed Past Mistakes Database

```bash
python scripts/seed_pb.py
```

Populates the past mistakes memory with sample error cases for the Critic agent.

---

## Frontend Dashboard

### Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page |
| `/diagnose` | Upload X-ray + start workflow |
| `/results/[session_id]` | Live results with SSE agent feed |
| `/observability` | System metrics dashboard |

### Tabs in Results Page

| Tab | Content |
|-----|---------|
| **Visual Proof** | Heatmaps, disease probability bars |
| **Clinical Notes** | Radiologist findings, CheXbert labels, debate summary |
| **Literary** | PubMed citations with relevance scores |
| **Safety** | Safety guardrails report (score, critical findings, red flags) |
| **Audit Trail** | Reproducibility hash (SHA-256) + full execution trace |

### Build for Production

```bash
cd frontend
npm run build
npm start
```

---

---

## Observability & Monitoring

### Metrics Endpoint

```bash
curl http://localhost:8000/api/v1/metrics/summary
```

Returns JSON with:
- `system`: active/total workflows, deferrals, critical findings
- `agents`: per-agent duration, invocation counts, information gain
- `diagnostics`: confidence, uncertainty, debate rounds, safety score
- `safety`: safety flags, errors by component

### Dashboard

Visit `http://localhost:3000/observability` for the visual metrics dashboard.

### Metrics Persistence

When running via `test_workflow.py`, metrics are saved to `metrics_snapshot.json` and automatically picked up by the API dashboard.

---

## Testing

### End-to-End Workflow Test

```bash
# Runs full pipeline on test image
python tests/test_workflow.py
```

### Frontend Build Test

```bash
cd frontend
npx next build
```

### Unit Tests

```bash
pytest tests/ -v
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/workflows/start` | Upload image(s) and start async workflow |
| `GET` | `/api/v1/workflows/{id}/status` | Poll workflow status + results |
| `GET` | `/api/v1/workflows/{id}/stream` | SSE stream for live agent progress |
| `POST` | `/api/v1/workflows/{id}/resume` | Submit doctor feedback (approve/reject) |
| `POST` | `/api/v1/safety/validate` | Run safety guardrails on a diagnosis |
| `GET` | `/api/v1/metrics/summary` | Get observability metrics |
| `GET` | `/api/v1/health` | Server health check |
| `GET` | `/api/v1/tools` | List available tools and agents |

### Start Workflow Request

```bash
curl -X POST http://localhost:8000/api/v1/workflows/start \
  -F "images=@chest_xray_AP.jpg" \
  -F "images=@chest_xray_LAT.jpg" \
  -F "views=AP" \
  -F "views=LATERAL" \
  -F "patient_id=patient-123" \
  -F "fhir_report=@patient_fhir_bundle.json"
```

### Status Response

```json
{
  "session_id": "abc-123-def",
  "status": "completed",
  "final_result": {
    "diagnosis": "Right lower lobe pneumonia with associated pleural effusion",
    "confidence": 0.87,
    "reproducibility_hash": "a3f9c2e1b7d4082f...",
    "evidence_packet": { ... },
    "trace": ["[INIT] Processing...", "[RAD] Findings generated...", ...]
  }
}
```

---

## Configuration Reference

All settings are in `app/config.py` and can be overridden via `.env`:

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDGEMMA_4B_MODEL` | `google/medgemma-1.5-4b-it` | Base MedGemma model |
| `MEDSIGLIP_BASE_MODEL` | `google/medsiglip-448` | Vision encoder for similarity |
| `TEXT_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | KLE embeddings |
| `MOCK_MODELS` | `False` | Skip real models, use mocks |

### Workflow Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBATE_MAX_ROUNDS` | `3` | Maximum debate rounds |
| `DEBATE_CONSENSUS_THRESHOLD` | `0.15` | Max disagreement for consensus |
| `MAX_ROUTING_STEPS` | `5` | Prevent infinite loops |
| `ENABLE_LLM_CRITIC` | `False` | MedGemma-based semantic critic |
| `ENABLE_PAST_MISTAKES_MEMORY` | `True` | Historical error retrieval |
| `ENABLE_DOCTOR_FEEDBACK` | `True` | Feedback reprocessing loop |
| `USE_DEBATE_WORKFLOW` | `True` | Enable multi-agent debate |
| `USE_PARALLEL_AGENTS` | `True` | Parallel agent execution |

### API Keys

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACE_TOKEN` | ✅ | Access to gated models |
| `NCBI_EMAIL` | ✅ | PubMed API policy |
| `NCBI_API_KEY` | Recommended | Higher rate limits (10 req/s) |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Semantic Scholar access |
| `SUPABASE_URL` | Optional | Cloud database URL |
| `SUPABASE_KEY` | Optional | Cloud database anon key |

---

## Technical Design Decisions

### Why Multi-Agent (Not Single Model)?

A single LLM producing diagnoses has no internal checks — it can hallucinate confidently. VERIFAI uses adversarial multi-agent verification:

1. **Radiologist** generates findings (may hallucinate)
2. **CheXbert** independently extracts structured labels (catches text hallucinations)
3. **Critic** adversarially challenges the findings using uncertainty metrics
4. **Debate** forces agents to defend/refute their positions
5. **Validator** applies hard clinical rules before finalizing

### Why MUC (Monotonic Uncertainty Cascade)?

The original system used **KLE (Kernel Language Entropy)** which generated 3-5 model completions and measured semantic spread — making it 5-15x slower and producing a **single, global** uncertainty value at the Radiologist stage that never changed. Even if the Critic flagged dangerous overconfidence, the uncertainty stayed frozen.

MUC replaces this with **Bidirectional Information Gain** where each agent updates a shared system uncertainty:

```
Direction  =  (alignment - 0.5) × 2         →  maps [0, 1] to [-1, +1]
IG         =  agent_confidence × direction × scaling_factor
U_system   =  clamp( U_system - IG,  0.05, 0.95 )
```

- Agent **confirms** the diagnosis (alignment > 0.5) → IG > 0 → uncertainty **decreases**
- Agent **contradicts** (alignment < 0.5) → IG < 0 → uncertainty **increases**
- Agent is **neutral** (alignment = 0.5) → IG = 0 → uncertainty unchanged

The Debate stage uses **Dempster-Shafer evidence fusion** to combine belief mass functions from three agents (Critic, Historian, Literature) before computing IG. When agents violently disagree (K ≥ 0.99), the system correctly remains uncertain rather than picking a winner.

Per-agent scaling factors sum to ~0.95, allowing a full cascade from maximum uncertainty (0.95) down to minimum (0.05 = 95% confidence) when all agents strongly agree. Each factor is grounded in specific research: CheXbert (0.20) from MARS, Debate (0.25) from Dempster-Shafer, etc.

### Why LangGraph (Not LangChain Agents)?

LangGraph provides:
- **Typed state** (TypedDict) shared across all agents — no message passing overhead
- **Checkpointing** — workflow can be interrupted and resumed (critical for human review)
- **Deterministic routing** — graph edges, not LLM-decided next steps
- **Thread safety** — concurrent workflows with isolated state

### Why Reproducibility Hash?

FDA 21 CFR Part 11 requires electronic records to be auditable. The SHA-256 hash proves:
- Which image was analyzed
- What patient context was available
- What model versions were used
- Configuration at time of diagnosis

This is provenance, not exact reproduction (LLMs are stochastic).


---


