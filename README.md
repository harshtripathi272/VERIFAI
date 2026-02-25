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
│  │    Validator (CXR-RePaiR Retrieval + RadGraph NLP + Rules Engine)  │   │
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
| 1 | **Radiologist** | Analyze CXR image, generate findings + impression | MedGemma 4B-IT + MedSigLIP + LRP (Chefer CVPR 2021) | `RadiologistOutput` (findings, impression, disease probabilities, LRP heatmaps) |
| 2 | **CheXbert** | Extract structured pathology labels from report text | CheXbert BERT | `CheXbertOutput` (14 CXR condition labels) |
| 3 | **Historian** | Retrieve patient history from EHR/FHIR records, generate clinical reasoning | DuckDB + FAISS vector search + Clinical Reasoner | `HistorianOutput` (supporting/contradicting facts, clinical summary) |
| 4 | **Literature** | Search PubMed/Europe PMC for evidence (rate-limited) | BioPython + Semantic Scholar API + Rate Limiter | `LiteratureOutput` (citations, evidence strength) |
| 5 | **Critic** | Adversarial verification — detect overconfidence, check past mistakes | MUC Uncertainty + SentenceTransformers | `CriticOutput` (safety score, concern flags) |
| 6 | **Debate** | Multi-round debate between Critic, Historian, and Literature | LangGraph orchestration | `DebateOutput` (consensus, confidence adjustment) |
| 7 | **Validator** | Final quality gate — visual retrieval + NLP entity matching + clinical rules | CXR-RePaiR (MedSigLIP FAISS) + RadGraph entity matching + Rules Engine | Recommendation: `FINALIZE` / `FINALIZE_LOW_CONFIDENCE` / `FLAG_FOR_HUMAN` |
| 8 | **Finalize** | Build final diagnosis with reproducibility hash | SHA-256, Pydantic | `FinalDiagnosis` (diagnosis, confidence, hash) |
| 9 | **Human Review** | Doctor approves/rejects; rejected cases re-enter at Critic | LangGraph `interrupt()` | Approve/Reject + Feedback loop |

---

## Key Features

### Core
- **Multi-Agent Orchestration** — 9 specialized agents coordinated via LangGraph state machine with typed state
- **Multi-View Support** — Accepts multiple X-ray views (AP, PA, Lateral) simultaneously
- **MUC (Monotonic Uncertainty Cascade)** — Bidirectional Information Gain per agent; each agent either confirms (↓ uncertainty) or contradicts (↑ uncertainty) the current diagnosis via log-odds updates. Dempster-Shafer evidence fusion during debate rounds. Grounded in 5 research papers (ICML 2026, ACL 2024, MICCAI 2024, Shafer 1976, arXiv:2601.15703). See [`docs/PRINCIPLED_UNCERTAINTY.md`](docs/PRINCIPLED_UNCERTAINTY.md) for full derivation.
- **Multi-Agent Debate** — Up to 3 rounds of adversarial debate before consensus
- **Human-in-the-Loop** — LangGraph interrupt-based doctor review with feedback reprocessing; rejected diagnoses re-enter at Critic with full context preserved
- **MCP-Style Tool Registry** — Unified tool interface (FHIR, Literature, Vision categories) inspired by Model Context Protocol servers, with tool discovery, invocation tracking, and rate limiting

### Safety & Trust
- **Medical Safety Guardrails** — Rule-based + embedding-based checks for dangerous hallucinations
- **Reproducibility Hash** — SHA-256 fingerprint (image + FHIR + config) for FDA 21 CFR Part 11 audit trail
- **Past Mistakes Memory** — Hybrid retrieval system: DuckDB HNSW vector index (local) + Supabase pgvector (cloud), with **neural re-ranking** (temporal recency weighting, clinical relevance scoring, optional MedGemma semantic analysis) and **automatic mistake detection** (severity classification, error-type taxonomy). Rejected diagnoses from human review are automatically inserted into the database for future reference.
- **CheXbert Cross-Validation** — Structured labels validate free-text radiologist findings
- **Validator Quality Gate** — Three-layer validation: CXR-RePaiR visual retrieval (FAISS similarity search against MIMIC-CXR), RadGraph NLP entity matching (structured clinical entity comparison), and clinical rules engine

### Infrastructure
- **SSE Real-Time Streaming** — Server-Sent Events for live agent progress to frontend (visible during both initial run and reruns)
- **LRP Heatmaps** — Transformer explainability via Chefer et al. (CVPR 2021) Layer-wise Relevance Propagation adapted for SigLIP
- **Observability Dashboard** — Prometheus-style metrics (latency, confidence, safety scores)
- **Evidence Report Generator** — Rich HTML reports with citations, heatmaps, and audit trail
- **Edge Deployable** — Runs on a single consumer GPU (12-16 GB VRAM) with optional 4-bit quantization

---

## Project Structure

```
VERIFAI/
├── agents/                      # AI Agent implementations
│   ├── radiologist/             # MedGemma 4B vision + MedSigLIP classifier + LRP heatmaps
│   │   ├── model.py             # Model loading (4-bit NF4 quantized) + VLM inference
│   │   ├── agent.py             # Radiologist agent logic
│   │   ├── classifier.py        # MedGemmaVisionHead: frozen SigLIP + trainable head
│   │   ├── lrp.py               # Chefer et al. (CVPR 2021) LRP heatmaps for SigLIP
│   │   ├── data.py              # CheXbert class labels + dataset constants
│   │   └── prompts.py           # Structured JSON generation prompts
│   ├── chexbert/                # CheXbert structured labeling
│   │   ├── agent.py             # Extract 14 CXR condition labels
│   │   └── model.py             # CheXbert BERT model wrapper
│   ├── historian/               # FHIR patient history retrieval + clinical reasoning
│   │   ├── agent.py             # DuckDB + FAISS vector search orchestrator
│   │   ├── fhir_client.py       # FHIR R4 resource parser + patient context builder
│   │   ├── reasoner.py          # Clinical reasoning synthesizer
│   │   └── hyp_code_map.py      # ICD/hypothesis code mappings
│   ├── literature/              # PubMed / Europe PMC / Semantic Scholar search
│   │   ├── agent.py             # Literature search orchestrator
│   │   ├── pubmed_entrez.py     # BioPython E-Utilities wrapper
│   │   ├── europe_pmc.py        # Europe PMC REST API
│   │   ├── semantic_scholar.py  # Semantic Scholar API
│   │   ├── rate_limiter.py      # Adaptive rate limiter (NCBI 3/sec, etc.)
│   │   └── prompt.py            # Literature search prompt templates
│   ├── critic/                  # Adversarial verification (4-stage + LLM)
│   │   ├── agent.py             # Overconfidence detection + past mistakes retrieval
│   │   ├── model.py             # Rule-based linguistic certainty + historian/literature challenge
│   │   └── llm_critic.py        # MedGemma semantic critic (gated)
│   ├── debate/                  # Multi-agent debate protocol
│   │   └── agent.py             # 3-round structured debate with Dempster-Shafer fusion
│   ├── validator/               # Final quality gate (3-layer verification)
│   │   ├── agent.py             # Validator orchestrator
│   │   ├── retrieval_tool.py    # CXR-RePaiR: MedSigLIP FAISS visual retrieval
│   │   ├── radgraph_tool.py     # RadGraph NLP entity matching
│   │   └── rules_engine.py      # Clinical rules engine
│   └── feedback/                # Doctor feedback handler
│       └── agent.py             # Process rejection → re-enter pipeline at Critic
│
├── graph/                       # LangGraph workflow
│   ├── state.py                 # VerifaiState TypedDict + Pydantic models
│   ├── workflow.py              # Full graph definition + node wrappers + interrupt()
│   └── router.py                # Uncertainty-based routing logic
│
├── tools/                       # MCP-Style Tool Registry
│   └── registry.py              # Unified tool interface (FHIR, Literature, Vision)
│
├── app/                         # FastAPI backend
│   ├── main.py                  # App entry point + middleware
│   ├── api.py                   # REST endpoints (start, status, resume, SSE)
│   ├── config.py                # Settings (models, thresholds, feature flags)
│   ├── streaming.py             # SSE event bus + streaming endpoint
│   ├── shared_model_loader.py   # Thread-safe MedGemma singleton (27GB→12GB VRAM)
│   └── past_mistakes_routes.py  # Past mistakes CRUD API endpoints
│
├── frontend/                    # Next.js 15 dashboard
│   ├── src/app/
│   │   ├── diagnose/page.tsx    # Upload X-ray + start workflow
│   │   ├── results/[id]/page.tsx # Live results + SSE feed + HITL review
│   │   └── observability/page.tsx # Metrics dashboard
│   └── src/lib/api.ts           # TypeScript API client
│
├── db/                          # Database layer (DuckDB local + Supabase cloud)
│   ├── logger.py                # Session-scoped structured logging
│   ├── supabase_logger.py       # Supabase cloud DB logger
│   ├── connection.py            # DuckDB connection pooling
│   ├── past_mistakes.py         # Past Mistakes DB (DuckDB + HNSW vector index)
│   ├── past_mistakes_repository.py # Supabase pgvector HNSW backend
│   ├── rerank_mistakes.py       # Neural re-ranking (temporal decay + clinical relevance)
│   ├── auto_detect_mistakes.py  # Automatic mistake detection + severity scoring
│   ├── adapter.py               # Database adapter abstraction
│   └── supabase_schema.sql      # Supabase table/function definitions
│
├── uncertainty/                 # MUC framework
│   ├── muc.py                   # Monotonic Uncertainty Cascade (Information Gain)
│   ├── kle.py                   # KL-divergence Epistemic uncertainty
│   └── case_embedding.py        # SentenceTransformers case embeddings for similarity
│
├── utils/                       # Shared utilities
│   ├── evidence_report.py       # Rich HTML evidence report builder
│   └── inference.py             # Robust JSON extraction from LLM output
│
├── safety/                      # Medical safety guardrails
│   └── guardrails.py            # Critical finding detection + hallucination checks
│
├── monitoring/                  # Observability layer
│   └── metrics.py               # Prometheus-style counters + histograms
│
├── tests/                       # Comprehensive test suite (17 test files)
│   ├── test_workflow.py         # End-to-end 9-agent workflow test
│   ├── test_debate.py           # Multi-round debate tests
│   ├── test_past_mistakes.py    # Past mistakes DB + retrieval tests
│   └── ...                      # 14 more test files
│
├── scripts/                     # Utility scripts
│   ├── build_retrieval_index.py # Build FAISS index from MIMIC-CXR
│   ├── install_radgraph_model.py # RadGraph model first-time setup
│   └── seed_pb.py               # Seed patient database
│
├── docs/                        # Documentation
│   └── PRINCIPLED_UNCERTAINTY.md # Full MUC uncertainty derivation (5 papers)
│
├── qlora_medgemma.py            # QLoRA fine-tuning (SFTTrainer + NF4 + LoRA)
├── train_classifier.py          # MedSigLIP classifier training
└── README.md

---

## Deep Dive: Agent Pipeline & Tools

**1. Radiologist** — `MedGemma 4B-IT` (4-bit NF4 quantized) + `MedSigLIP` classifier + `Chefer LRP` heatmaps
Generates structured JSON (`{findings, impression}`) from CXR images using MedGemma VLM with custom `StopOnCloseBrace` stopping criteria. Separately, a fine-tuned MedSigLIP classifier (frozen SigLIP backbone + trainable head) predicts 14 disease probabilities. For classes > 0.5, Layer-wise Relevance Propagation (Chefer CVPR 2021) generates attribution heatmaps overlaid on the CXR. Multi-view support via `<PA>`, `<AP>`, `<LATERAL>` tokens.

**2. CheXbert** — `CheXbert BERT` model
Extracts 14 structured pathology labels (present/absent/uncertain) from the Radiologist's free-text report. Acts as an independent cross-check — catches text hallucinations the VLM might produce.

**3. Historian** — `FHIR R4 client` + `DuckDB` + `FAISS` vector search + `Clinical Reasoner`
Retrieves patient history from FHIR bundles, identifies supporting and contradicting clinical facts, and synthesizes a clinical reasoning summary via shared MedGemma.

**4. Literature** — `PubMed E-Utilities` + `Europe PMC REST` + `Semantic Scholar API` + `Rate Limiter`
Searches 3 databases for relevant evidence. Adaptive rate limiter respects NCBI's 3 req/sec policy. Returns ranked citations with evidence strength via shared MedGemma.

**5. Critic** — `Rule-based linguistic analysis` + `Historian/Literature challenge` + `Past Mistakes DuckDB` + *(optional)* `LLM Semantic Critic`
Runs 4-stage adversarial evaluation: (1) compares linguistic certainty vs. KLE uncertainty, (2) penalizes unaddressed contradicting FHIR facts, (3) flags omitted differentials from literature, (4) retrieves similar past diagnostic errors from DuckDB/Supabase (HNSW vector search + neural re-ranking). Doctor-rejected diagnoses (where agents might have hallucinated) are auto-inserted into this database for future retrieval so that the system learns from its mistakes. Optional 5th stage uses MedGemma for deeper semantic analysis.

**6. Debate** — `Dempster-Shafer fusion` + `LangGraph` orchestration
Up to 3 rounds of structured debate between Critic, Historian, and Literature. Each round adjusts uncertainty via Dempster-Shafer evidence fusion until consensus or max rounds.

**7. Validator** — `CXR-RePaiR` (MedSigLIP FAISS retrieval) + `RadGraph` NLP entity matching + `Rules Engine`
Three-layer quality gate: (1) visual retrieval of similar historical CXRs from MIMIC-CXR FAISS index, (2) RadGraph extracts clinical entities and compares against retrieved reports, (3) clinical rules engine checks for critical finding patterns. Outputs: `FINALIZE` / `FINALIZE_LOW_CONFIDENCE` / `FLAG_FOR_HUMAN`.

**8. Finalize** — `SHA-256` reproducibility hash + `Pydantic` models
Builds final diagnosis with confidence score and a cryptographic hash of (image + FHIR + config) for audit trail.

**9. Human Review** — `LangGraph interrupt()` + `Past Mistakes auto-insertion`
Doctor approves or rejects with feedback. Rejected cases re-enter the pipeline at the Critic node with full context preserved. Rejected diagnoses are automatically saved to the Past Mistakes database for future learning.

**Cross-cutting: Shared Model Loader** — Historian, Literature, and LLM Critic share a single MedGemma 4B instance via thread-safe singleton (`shared_model_loader.py`), reducing VRAM from ~27 GB to ~9 GB.


---

## System Entropy Cascade (Monotonic Uncertainty Cascade — MUC)

Every agent updates a single global value `current_uncertainty ∈ [0.05, 0.95]` using Bayesian log-odds. This is the authoritative system entropy — it is **not** per-agent; it cascades across the entire pipeline.

**Core update rule:**

```
IG(k) = α · confidence(k) + β · alignment(k) + γ · direction(k)
U_sys(k) = clamp( U_sys(k-1) − IG(k), 0.05, 0.95 )
```

`U_sys` starts at 1.0 (maximum uncertainty) and decreases as each agent provides confirming evidence. Contradictions (direction = −1) or low confidence reduce IG, keeping entropy high.

| Agent | Confidence Signal | Alignment Signal | Notes |
|-------|-------------------|------------------|-------|
| Radiologist | Token KLE score | MedSigLIP class prob | Initializes cascade |
| CheXbert | Label Shannon entropy | % matching Radiologist | Label cross-check |
| Historian | FHIR fact consensus | Supporting vs. contradicting | Raises U if contradictions found |
| Literature | Evidence rating | Paper stance vs. impression | Multi-DB citation |
| Critic | Safety score | Certainty-uncertainty gap | Past mistakes penalty |
| Debate | Dempster-Shafer mass | Consensus reached? | DS fusion per round |
| Validator | CXR-RePaiR similarity | RadGraph entity match | Final gate |

**Uncertainty history in LLM prompts** — The last **2** `{agent, system_uncertainty}` values are injected into the LLM Critic and Historian prompts. If the latest value is _higher_ than the previous (i.e. entropy increased), a ⚠️ spike warning is added: `"System uncertainty INCREASED after '{agent}' — investigate contradictions from that stage."` This tells the LLM to look specifically at which agent's output raised doubt.

**Uncertainty graph in UI** — The results page renders a live SVG line chart of `uncertainty_history` (X = agent names, Y = entropy 0–1) so clinicians can see how confidence evolved over the pipeline.

---


| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.10 |
| **CUDA** | 12.1 | 12.6 |
| **GPU VRAM** | 12-16 GB | 24+ GB |
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

This runs the full 9-agent pipeline on a test image and prints:
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

VERIFAI includes a QLoRA fine-tuning pipeline for adapting MedGemma to generate structured radiology reports.

### QLoRA Fine-Tuning (Recommended)

Memory-efficient fine-tuning using 4-bit NF4 quantization + Low-Rank Adaptation via `SFTTrainer`:

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

**Training Details:**
- **LoRA Config:** rank=16, alpha=16, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Quantization:** NF4 double-quantized via BitsAndBytes
- **Optimizer:** `paged_adamw_8bit` with gradient checkpointing
- **Vision tower:** Frozen during training (only language model adapts)
- **Dataset:** MIMIC-CXR JSONL with multi-view support (`<PA>`, `<AP>`, `<LATERAL>` tokens)
- **Output:** Structured JSON (`{"findings": "...", "impression": "..."}`) via chat template

**Requirements:** 12-16 GB VRAM (4-bit base + LoRA adapters + activations)

**Dataset format:** Directory containing subdirectories per patient, each with:
- `study1/` containing `.jpg` chest X-ray images
- `findings.txt` and `impression.txt` (ground truth)

### MedSigLIP Disease Classifier

The disease classifier is a custom `MedGemmaVisionHead` that wraps a frozen MedSigLIP vision encoder (`google/medsiglip-448`) with a trainable classification head:

```
Frozen SigLIP Vision Encoder → MAP Pooled Output → Linear(1152, 1152) → LayerNorm → GELU → Linear(1152, 14)
```

The classification head predicts 14 CheXbert pathology labels. Training only updates the head parameters while keeping the vision backbone frozen. The model must use `attn_implementation="eager"` for LRP gradient extraction.

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

### Database Schema (Supabase)

When `DATABASE_MODE=supabase` or cloud logging is enabled, VERIFAI logs extensive telemetry and workflow states to the following Supabase (PostgreSQL) tables for observability, auditability, and the past-mistakes memory loop.

| Table Name | Description | Key Features |
|------------|-------------|--------------|
| `workflow_sessions` | Core table tracking each full pipeline invocation. | `session_id`, `status`, `final_diagnosis`, doctor feedback flags. |
| `agent_invocations` | Tracks each individual agent call within a session. | Execution duration, input/output summaries. |
| `radiologist_logs` | Radiologist outputs and initial bounds. | Findings, impression, `kle_uncertainty`. |
| `critic_logs` | Adversarial checks and overconfidence detection. | `is_overconfident`, `safety_score`, historical risk. |
| `historian_logs`, `_facts` | FHIR retrieved clinical context and extracted facts. | Supporting vs. contradicting clinical facts. |
| `literature_logs`, `_citations` | PubMed/PMC searches and matched papers. | Evidence strength, relevance summaries, PMIDs. |
| `debate_logs`, `_rounds`, `_arguments`| Full record of the multi-agent debate process. | Rounds, arguments presented, consensus status. |
| `chief_logs` | Final arbitration decisions by the Chief agent. | Calibrated confidence, deferral reasons. |
| `trace_log` | Flat diagnostic audit trail mirroring `state.trace`. | Line-by-line event log for FDA trace compliance. |
| `doctor_feedback` | Captures human-in-the-loop review actions. | Doctor notes, corrected diagnoses, reprocessing links. |
| `past_mistakes` | Memory repository of historical diagnostic errors. | Uses **pgvector** (`VECTOR(384)`) and HNSW index for semantic retrieval. |

See `db/supabase_schema.sql` for the complete schema definitions.

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
| `GET` | `/api/v1/tools` | List available MCP-registered tools |
| `POST` | `/api/past-mistakes/insert` | Insert a validated diagnostic mistake |
| `POST` | `/api/past-mistakes/search` | Search for similar past mistakes (hybrid vector search) |
| `GET` | `/api/past-mistakes/{id}` | Get a specific mistake by ID |
| `DELETE`| `/api/past-mistakes/{id}` | Delete a mistake record |
| `GET` | `/api/past-mistakes/statistics` | Aggregate statistics on past mistakes |

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

### Model Configuration (`app/config.py`)

You can configure model settings and local paths for fine-tuned weights by updating `.env` or conceptually modifying `app/config.py`. All local paths can be set as absolute paths if your models are stored across different directories to manage disk space.

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDGEMMA_4B_MODEL` | `google/medgemma-1.5-4b-it` | Base MedGemma model |
| `MEDSIGLIP_BASE_MODEL` | `google/medsiglip-448` | Vision encoder for similarity |
| `MEDSIGLIP_WEIGHTS_PATH` | `../output/medsiglip_full_model.pt` | Local path to fine-tuned MedSigLIP weights |
| `MEDGEMMA_LORA_ROOT` | `../dataset/med/fine_tuned_model/v1/` | Base directory for MedGemma fine-tuned models |
| `MEDGEMMA_LORA_ADAPTERS`| `../dataset/med/fine_tuned_model/v1/` | Path to load local MedGemma LoRA adapters |
| `TEXT_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | KLE embeddings |
| `RADGRAPH_CACHE_DIR` | `~/elephant_detection/med/dataset/med` | Base directory for RadGraph model |
| `MOCK_MODELS` | `False` | Skip real models, use mocks |

> **Note on Custom Model Paths**: To override default relative paths like `MEDGEMMA_LORA_ADAPTERS` or `MEDSIGLIP_WEIGHTS_PATH`, set them in your `.env` file using absolute paths (e.g., `MEDGEMMA_LORA_ADAPTERS=/path/to/your/custom/adapters`).

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
3. **Critic** adversarially challenges the findings using uncertainty metrics + past mistakes memory
4. **Debate** forces agents to defend/refute their positions with Dempster-Shafer fusion
5. **Validator** applies three-layer verification: CXR-RePaiR visual retrieval, RadGraph NLP entity matching, and clinical rules engine before finalizing

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


