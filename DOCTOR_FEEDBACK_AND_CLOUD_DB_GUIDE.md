# Doctor Feedback Loop & Cloud Database Migration Guide

## Overview

This guide covers two major enhancements to VERIFAI:

1. **Cloud Database Migration**: Move from local SQLite to Supabase (cloud PostgreSQL)
2. **Doctor Feedback Loop**: Allow doctors to reject diagnoses and trigger reprocessing

---

## Part 1: Cloud Database with Supabase

### Why Migrate to Supabase?

**SQLite (Local)** → Good for development, single-user testing
- ❌ No cloud access
- ❌ Single-file storage (can be lost)
- ❌ Limited concurrent access
- ❌ Manual backups required

**Supabase (Cloud)** → Better for production, multi-user deployments
- ✅ Cloud-hosted PostgreSQL
- ✅ Automatic backups and replication
- ✅ Real-time subscriptions
- ✅ Built-in authentication
- ✅ Row-level security
- ✅ GraphQL API out of the box

### Setup Instructions

#### Step 1: Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up or log in
3. Click **"New Project"**
   - Choose organization
   - Name: `verifai-production`
   - Database password: *save this securely*
   - Region: Choose closest to your users
4. Wait for project to initialize (~2 minutes)

#### Step 2: Get Connection Credentials

1. In your Supabase dashboard, go to **Settings** → **API**
2. Copy these values:
   - **Project URL** (looks like `https://xxxxxxxxxxxxx.supabase.co`)
   - **anon public** key (safe for client-side use)
   - **service_role** key (keep secret! server-side only)

#### Step 3: Set Up Database Schema

1. In Supabase dashboard, go to **SQL Editor**
2. Open the file `db/supabase_schema.sql` from this project
3. Copy all content and paste into SQL Editor
4. Click **Run**
5. Verify tables were created: Go to **Table Editor** and see 14 tables

#### Step 4: Configure Environment Variables

Update your `.env` file:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_anon_key_here

# Optional: for admin operations
SUPABASE_SERVICE_KEY=your_service_role_key_here

# Set database mode to cloud
DATABASE_MODE=supabase
```

#### Step 5: Install Python Client

```bash
pip install supabase
```

Or update from `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### Step 6: Test Connection

```python
from db.adapter import check_database_health

health = check_database_health()
print(health)
# Should show: {'mode': 'supabase', 'healthy': True, ...}
```

### Migration from SQLite to Supabase

If you have existing data in SQLite:

```python
from db.adapter import migrate_to_cloud

# Migrate all data from local SQLite to Supabase
migrate_to_cloud()

# Or specify custom path
migrate_to_cloud("path/to/verifai_logs.db")
```

**Warning**: This is a one-time operation. Verify your Supabase credentials before running.

### Switching Between SQLite and Supabase

The system automatically uses the correct database based on `DATABASE_MODE` in `.env`:

```bash
# Use cloud database (recommended for production)
DATABASE_MODE=supabase

# Use local SQLite (development/testing)
DATABASE_MODE=sqlite
```

No code changes required! The adapter transparently switches between backends.

---

## Part 2: Doctor Feedback Loop

### Overview

When a doctor reviews a diagnosis and finds it incorrect, they can:
1. **Reject** the diagnosis with notes explaining what's wrong
2. System captures full context (all agent outputs at rejection point)
3. System **restarts from Critic** with doctor's feedback injected
4. New diagnosis is generated with doctor's guidance
5. Results are linked back to show improvement

### Architecture

```
Normal Flow:
START → Radiologist → CheXbert → Evidence → Critic → Debate → Finalize

Feedback Flow (after rejection):
START → [preserved context] → Critic (+ doctor notes) → Debate → New Finalize
                                      ↑
                                Doctor's feedback injected here
```

**Key Insight**: We skip Radiologist/CheXbert/Evidence gathering because:
- The image interpretation hasn't changed
- We preserve all original agent outputs
- Only Critic needs to re-evaluate with doctor's guidance

### Usage Example

#### Step 1: Run Initial Diagnosis

```python
from graph.workflow import app
from graph.state import VerifaiState

# Normal workflow
state = VerifaiState(
    image_path="patient_001_xray.jpg",
    patient_id="P001",
    _session_id="session-abc-123"
)

result = app.invoke(state)
print(f"Diagnosis: {result['final_diagnosis'].diagnosis}")
print(f"Confidence: {result['final_diagnosis'].calibrated_confidence}")
```

#### Step 2: Doctor Reviews and Rejects

```python
from agents.feedback import capture_doctor_feedback

# Doctor finds diagnosis incorrect
feedback_id = capture_doctor_feedback(
    session_id="session-abc-123",
    feedback_type="rejection",
    doctor_notes="Diagnosis missed bilateral pleural effusion. "
                 "Focused only on consolidation but effusion is more significant. "
                 "Also patient has history of CHF which supports effusion diagnosis.",
    correct_diagnosis="Bilateral pleural effusion, likely cardiogenic",
    rejection_reasons=["missed_finding", "incorrect_primary_diagnosis"],
    doctor_id="dr_smith"
)

print(f"Feedback captured: {feedback_id}")
```

#### Step 3: System Reprocesses with Feedback

```python
from agents.feedback import (
    prepare_feedback_for_reprocessing,
    create_feedback_enhanced_state,
    link_feedback_reprocessing_result
)

# Prepare feedback for reprocessing
feedback_input = prepare_feedback_for_reprocessing(feedback_id)

# Create new state with preserved context + doctor feedback
new_state = create_feedback_enhanced_state(
    feedback_input=feedback_input,
    image_path="patient_001_xray.jpg",
    patient_id="P001"
)

# Reprocess (automatically starts from Critic with feedback)
new_result = app.invoke(new_state)

print(f"New diagnosis: {new_result['final_diagnosis'].diagnosis}")
print(f"New confidence: {new_result['final_diagnosis'].calibrated_confidence}")

# Link results
link_feedback_reprocessing_result(
    feedback_id=feedback_id,
    new_session_id=new_state['_session_id'],
    final_diagnosis=new_result['final_diagnosis'].diagnosis,
    final_confidence=new_result['final_diagnosis'].calibrated_confidence
)
```

### Database Schema for Feedback

The `doctor_feedback` table stores:

```sql
CREATE TABLE doctor_feedback (
    feedback_id         SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    original_diagnosis  TEXT,
    original_confidence REAL,
    
    -- Feedback details
    feedback_type       TEXT NOT NULL,  -- 'rejection', 'correction', 'approval'
    doctor_notes        TEXT NOT NULL,
    correct_diagnosis   TEXT,
    rejection_reason    TEXT[],
    
    -- Reprocessing tracking
    reprocessed         BOOLEAN DEFAULT FALSE,
    reprocess_session_id TEXT,
    reprocess_result    TEXT,
    
    -- Metadata
    context_snapshot    JSONB,  -- Full workflow state at rejection
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### How Critic Uses Feedback

When `is_feedback_iteration=True`:

```python
# In critic_node (agents/critic/agent.py)

if is_feedback_iteration and doctor_feedback:
    # Add doctor feedback as high-priority concern
    concern_flags.insert(0, f"DOCTOR FEEDBACK: {doctor_feedback.doctor_notes}")
    
    # Add correct diagnosis if provided
    if doctor_feedback.correct_diagnosis:
        concern_flags.insert(1, f"Doctor's correct diagnosis: {correct_diagnosis}")
    
    # Lower safety score (higher scrutiny)
    safety_score = max(0.1, safety_score - 0.3)
    is_overconfident = True  # Force reprocessing
```

This ensures the debate phase has the doctor's guidance and can correct the error.

### UI Integration Example (Streamlit)

```python
import streamlit as st
from agents.feedback import capture_doctor_feedback

# Show diagnosis
st.header("AI Diagnosis")
st.write(f"**Diagnosis:** {diagnosis}")
st.write(f"**Confidence:** {confidence:.1%}")

# Feedback form
with st.expander("Provide Feedback"):
    feedback_type = st.radio("Feedback Type", ["approval", "correction", "rejection"])
    
    if feedback_type in ["correction", "rejection"]:
        doctor_notes = st.text_area("What's wrong with this diagnosis?")
        correct_dx = st.text_input("Correct diagnosis (optional)")
        
        reasons = st.multiselect(
            "Issue categories",
            ["missed_finding", "incorrect_primary_diagnosis", 
             "wrong_severity", "missed_complication"]
        )
        
        if st.button("Submit Feedback"):
            feedback_id = capture_doctor_feedback(
                session_id=session_id,
                feedback_type=feedback_type,
                doctor_notes=doctor_notes,
                correct_diagnosis=correct_dx,
                rejection_reasons=reasons
            )
            st.success(f"Feedback submitted! ID: {feedback_id}")
            
            # Trigger reprocessing
            if st.button("Reprocess with Feedback"):
                # ... reprocessing code ...
                st.success("Reprocessing complete!")
```

---

## Configuration Reference

### Environment Variables

```bash
# ============================================================
# SUPABASE (Cloud Database)
# ============================================================

# Required for Supabase mode
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your_anon_key

# Optional: Admin operations
SUPABASE_SERVICE_KEY=your_service_role_key

# Database mode selection
DATABASE_MODE=supabase  # or 'sqlite' for local

# ============================================================
# DOCTOR FEEDBACK
# ============================================================

# Enable doctor feedback loop
ENABLE_DOCTOR_FEEDBACK=True

# Automatically restart from critic when feedback provided
FEEDBACK_RESTART_FROM_CRITIC=True
```

### Code Configuration

In `app/config.py`:

```python
class Settings(BaseSettings):
    # Supabase
    SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY")
    DATABASE_MODE: str = os.getenv("DATABASE_MODE", "supabase")
    
    # Doctor Feedback
    ENABLE_DOCTOR_FEEDBACK: bool = True
    FEEDBACK_RESTART_FROM_CRITIC: bool = True
```

---

## Querying Feedback Data

### Get All Feedback for a Session

```python
from db.adapter import get_logger

summary = get_logger.get_session_summary("session-abc-123")
feedback = summary['feedback']

for fb in feedback:
    print(f"Type: {fb['feedback_type']}")
    print(f"Notes: {fb['doctor_notes']}")
    print(f"Reprocessed: {fb['reprocessed']}")
```

### Get Feedback Improvement Metrics

```sql
-- In Supabase SQL Editor

SELECT 
    feedback_type,
    COUNT(*) as total_feedbacks,
    SUM(CASE WHEN reprocessed THEN 1 ELSE 0 END) as reprocessed_count,
    AVG(original_confidence) as avg_original_confidence,
    AVG(reprocess_confidence) as avg_reprocess_confidence,
    AVG(reprocess_confidence - original_confidence) as avg_improvement
FROM doctor_feedback
WHERE feedback_type = 'rejection'
GROUP BY feedback_type;
```

---

## Benefits Summary

### Cloud Database (Supabase)
- ✅ Production-ready scalability
- ✅ Automatic backups
- ✅ Real-time analytics
- ✅ Multi-user collaboration
- ✅ Built-in security

### Doctor Feedback Loop
- ✅ Continuous learning from errors
- ✅ Expert-in-the-loop improvement
- ✅ Audit trail for quality assurance
- ✅ Faster iteration (skip re-analyzing image)
- ✅ Context-aware reprocessing

---

## Troubleshooting

### Supabase Connection Issues

```python
# Check connection
from db.supabase_connection import health_check

if not health_check():
    print("Connection failed!")
    # Check: SUPABASE_URL and SUPABASE_KEY in .env
    # Check: Supabase project is active
    # Check: Database schema is created
```

### Feedback Not Working

1. Verify `ENABLE_DOCTOR_FEEDBACK=True` in `.env`
2. Check `doctor_feedback` table exists in database
3. Ensure original session has complete data

```python
# Verify session exists
summary = get_logger.get_session_summary("session-id")
assert summary is not None, "Session not found"
```

---

## Next Steps

1. **Deploy to Production**
   - Set `DATABASE_MODE=supabase`
   - Configure environment variables in your hosting platform
   - Run initial health check

2. **Integrate with UI**
   - Add feedback form to diagnosis review screen
   - Show reprocessing status
   - Display before/after comparison

3. **Monitor Feedback Metrics**
   - Track feedback frequency
   - Measure confidence improvements
   - Identify common error patterns

4. **Set Up Alerts**
   - Notify relevant team when feedback is submitted
   - Alert on high-severity rejections
   - Track reprocessing queue

---

## Support

For issues or questions:
- Check database connection health first
- Review trace logs in workflow sessions
- Verify environment variables are set correctly
- Check Supabase dashboard for connection status

## Reference Files

- `db/supabase_schema.sql` - Database schema
- `db/supabase_connection.py` - Connection manager
- `db/supabase_logger.py` - Logger implementation
- `db/adapter.py` - Transparent switching layer
- `agents/feedback/agent.py` - Feedback processing
- `graph/workflow.py` - Workflow with feedback routing
