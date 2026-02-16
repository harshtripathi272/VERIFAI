# VERIFAI Doctor Feedback Flow Diagram

## Normal Workflow (First Run)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          NORMAL DIAGNOSTIC FLOW                          │
└──────────────────────────────────────────────────────────────────────────┘

    START
      │
      ▼
┌─────────────┐
│ RADIOLOGIST │ ────┐
│  (vision)   │     │ Generates findings & impression
└─────────────┘     │ + KLE uncertainty score
      │             │
      ▼             │
┌─────────────┐     │
│  CHEXBERT   │     │ Labels pathologies
│  (labeler)  │     │ (present/uncertain only)
└─────────────┘     │
      │             │
      ▼             │
┌─────────────┐     │
│  EVIDENCE   │────┐│ RAG: FHIR + Literature
│  GATHERING  │    ││ (parallel execution)
└─────────────┘    ││
      │            ││
      ▼            ││
┌─────────────┐    ││
│   CRITIC    │◄───┴┴─── Evaluates with full context:
│             │          • Findings/impression text
└─────────────┘          • KLE uncertainty
      │                  • FHIR clinical history
      ▼                  • Literature evidence
┌─────────────┐          • Past mistake patterns
│   DEBATE    │
│ (3 rounds)  │ ───┐
└─────────────┘    │
      │            │ Critic vs Evidence Team
      │            │ Confidence calibration
      ▼            │
    Consensus?     │
    /        \     │
  YES         NO   │
   │           │   │
   │           ▼   │
   │      ┌────────┴──┐
   │      │   CHIEF   │
   │      │ (arbiter) │
   │      └───────────┘
   │           │
   ▼           ▼
┌─────────────────────┐
│      FINALIZE       │ ──► Final Diagnosis Stored
│  (diagnosis ready)  │     in Database (session_id)
└─────────────────────┘

```

---

## Doctor Review & Rejection

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        DOCTOR REVIEW INTERFACE                           │
└──────────────────────────────────────────────────────────────────────────┘

    Doctor views diagnosis:
    
    ╔══════════════════════════════════════════════════════╗
    ║  DIAGNOSIS: Pneumonia, right lower lobe              ║
    ║  CONFIDENCE: 78%                                     ║
    ║                                                      ║
    ║  FINDINGS: Opacity in RLL, air bronchograms...      ║
    ║  IMPRESSION: Consistent with bacterial pneumonia... ║
    ╚══════════════════════════════════════════════════════╝
    
    Doctor disagrees ❌
    
    ┌──────────────────────────────────────────────────────┐
    │ [REJECT] │ [CORRECT] │ [APPROVE]                     │
    └──────────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────────────┐
    │ What's wrong with this diagnosis?                    │
    │ ┌────────────────────────────────────────────────┐   │
    │ │ This is not pneumonia. There is a large        │   │
    │ │ pleural effusion bilaterally that was missed.  │   │
    │ │ The opacity is due to fluid, not consolidation.│   │
    │ │ Patient has CHF history which supports this.   │   │
    │ └────────────────────────────────────────────────┘   │
    │                                                      │
    │ Correct diagnosis: [Bilateral pleural effusion]     │
    │                                                      │
    │ Issues: [✓] Missed finding                          │
    │         [✓] Wrong primary diagnosis                  │
    │         [ ] Wrong severity                           │
    └──────────────────────────────────────────────────────┘
              │
              ▼
    [SUBMIT FEEDBACK] ──► Stored in doctor_feedback table
                          with full context snapshot

```

---

## Feedback-Driven Reprocessing

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     FEEDBACK REPROCESSING FLOW                           │
└──────────────────────────────────────────────────────────────────────────┘

    START (new session_id)
      │
      │ is_feedback_iteration = TRUE
      │ doctor_feedback = {...}
      │
      ▼
   
   ╔═══════════════════════════════════════════════╗
   ║  PRESERVED CONTEXT (from original session)    ║
   ╠═══════════════════════════════════════════════╣
   ║  ✓ Radiologist findings & impression          ║
   ║  ✓ CheXbert labels                            ║
   ║  ✓ Historian FHIR data                        ║
   ║  ✓ Literature citations                       ║
   ║  ✓ KLE uncertainty score                      ║
   ╚═══════════════════════════════════════════════╝
      │
      │ Skip image analysis (already done!)
      │
      ▼
┌─────────────────┐
│  CRITIC         │
│  (enhanced)     │◄────┬─── Preserved context
│                 │     │
│  + FEEDBACK     │◄────┴─── Doctor's notes:
│    INJECTION    │          "Missed pleural effusion..."
└─────────────────┘          "Correct: Bilateral effusion"
      │
      │ Concern flags enriched:
      │ 1. "DOCTOR FEEDBACK: Missed pleural effusion..."
      │ 2. "Doctor's correct diagnosis: Bilateral effusion"
      │ 3. [original concerns...]
      │
      │ Safety score lowered (-0.3)
      │ is_overconfident = TRUE (force reprocessing)
      │
      ▼
┌─────────────────┐
│   DEBATE        │
│   (3 rounds)    │
└─────────────────┘
      │
      │ Round 1:
      │ CRITIC: "Doctor identified missed finding..."
      │ HISTORIAN: "Patient has CHF history, effusion likely"
      │ LITERATURE: "Effusion common in CHF patients..."
      │
      │ Round 2:
      │ CRITIC: "Evidence supports doctor's assessment"
      │ HISTORIAN: "FHIR shows multiple CHF episodes"
      │ LITERATURE: "Strong evidence for cardiogenic effusion"
      │
      │ CONSENSUS REACHED ✓
      │
      ▼
┌─────────────────┐
│   FINALIZE      │
└─────────────────┘
      │
      ▼
┌───────────────────────────────────────────────────┐
│  NEW DIAGNOSIS                                    │
│  ─────────────────────────────────────────────    │
│  Diagnosis: Bilateral pleural effusion,           │
│             likely cardiogenic                     │
│  Confidence: 82%                                  │
│                                                   │
│  Linked to:                                       │
│  • Original session: abc-123                      │
│  • Feedback ID: 456                               │
│  • Improvement: +4% confidence                    │
└───────────────────────────────────────────────────┘

```

---

## Database Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        DATABASE RELATIONSHIPS                            │
└──────────────────────────────────────────────────────────────────────────┘

workflow_sessions                    doctor_feedback
┌────────────────┐                  ┌─────────────────┐
│ session_id (PK)│◄─────────────────│ session_id (FK) │
│ patient_id     │                  │ feedback_id (PK)│
│ final_diagnosis│                  │ doctor_notes    │
│ confidence     │                  │ correct_dx      │
│ has_feedback ──┼──► TRUE          │ reprocessed ────┼──► FALSE → TRUE
│ feedback_count │                  │ reprocess_id    │
└────────────────┘                  └─────────────────┘
        │                                    │
        │                                    │ (after reprocessing)
        │                                    │
        │                                    ▼
        │                            ┌─────────────────┐
        └───────────────────────────►│ session_id (FK) │ New session
                                     │ (reprocess)     │
                                     └─────────────────┘
                                     
                                     Links back to show:
                                     • Original diagnosis (rejected)
                                     • Doctor's feedback
                                     • New diagnosis (corrected)
                                     • Confidence change
```

---

## Benefits Visualization

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     WHY RESTART FROM CRITIC?                             │
└──────────────────────────────────────────────────────────────────────────┘

FULL REPROCESSING (slow):
├─ Radiologist: Re-analyze image           ⏱️  30-60 seconds
├─ CheXbert: Re-label findings             ⏱️  5-10 seconds
├─ Evidence: Re-fetch FHIR + Literature    ⏱️  15-30 seconds
├─ Critic: Evaluate with feedback          ⏱️  5-10 seconds
├─ Debate: 3 rounds                        ⏱️  10-20 seconds
└─ Total: ~70-130 seconds

FEEDBACK REPROCESSING (fast):
├─ [Skip image analysis - use cached]      ⏱️  0 seconds ✓
├─ [Skip CheXbert - use cached]            ⏱️  0 seconds ✓
├─ [Skip Evidence - use cached]            ⏱️  0 seconds ✓
├─ Critic: Evaluate with feedback          ⏱️  5-10 seconds
├─ Debate: 3 rounds                        ⏱️  10-20 seconds
└─ Total: ~15-30 seconds

⚡ 60-80% FASTER!
```

---

## Key Components

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION FILES                             │
└──────────────────────────────────────────────────────────────────────────┘

📁 Database Layer
│
├─ db/supabase_schema.sql
│  └─ PostgreSQL schema with doctor_feedback table
│
├─ db/supabase_connection.py
│  └─ Supabase client connection manager
│
├─ db/supabase_logger.py
│  └─ Cloud-compatible logging with feedback tracking
│
└─ db/adapter.py
   └─ Transparent switch: SQLite ↔ Supabase

📁 Feedback Agent
│
├─ agents/feedback/agent.py
│  ├─ capture_doctor_feedback()
│  ├─ prepare_feedback_for_reprocessing()
│  ├─ create_feedback_enhanced_state()
│  └─ link_feedback_reprocessing_result()
│
└─ agents/feedback/__init__.py

📁 Workflow Updates
│
├─ graph/state.py
│  ├─ DoctorFeedback (BaseModel)
│  └─ VerifaiState.doctor_feedback
│
├─ graph/workflow.py
│  ├─ should_start_from_critic() ──► Routing logic
│  └─ Updated build_workflow() ──► Feedback path
│
└─ agents/critic/agent.py
   └─ Inject doctor feedback into concern_flags

📁 Utilities
│
├─ setup_helper.py
│  ├─ check_database()
│  ├─ migrate_database()
│  ├─ test_feedback_flow()
│  └─ show_stats()
│
└─ DOCTOR_FEEDBACK_AND_CLOUD_DB_GUIDE.md
   └─ Complete setup and usage documentation
```

---

## Decision Points in Workflow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    WORKFLOW ROUTING DECISIONS                            │
└──────────────────────────────────────────────────────────────────────────┘

    START
      │
      ▼
    ┌───────────────────┐
    │ is_feedback_      │
    │ iteration?        │
    └───────────────────┘
       /            \
     NO              YES
      │               │
      ▼               ▼
┌──────────┐    ┌──────────┐
│ Normal   │    │ Skip to  │
│ Flow     │    │ Critic   │
│          │    │ + Feed-  │
│ Radio    │    │ back     │
│ ↓        │    └──────────┘
│ CheX     │           ↓
│ ↓        │         Debate
│ Evid     │           ↓
│ ↓        │       (Consensus?)
│ Critic   │          / \
└──────────┘        NO  YES
      │              │   │
      ▼              ▼   ▼
    Debate      Chief  Finalize
      │
   (Consensus?)
    / \
  NO  YES
   │   │
   ▼   ▼
Chief Finalize

```

This visual guide shows the complete feedback loop architecture!
