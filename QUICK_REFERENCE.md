# VERIFAI Quick Reference Card

## 🚀 Setup Commands

```bash
# Check database health
python setup_helper.py check-db

# Show statistics
python setup_helper.py stats

# Migrate SQLite → Supabase
python setup_helper.py migrate

# Test feedback system
python setup_helper.py test-feedback
```

## 📝 Environment Configuration

### `.env` File
```bash
# Supabase (Cloud Database)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=your_anon_key_here
DATABASE_MODE=supabase  # or 'sqlite' for local

# Doctor Feedback
ENABLE_DOCTOR_FEEDBACK=True
FEEDBACK_RESTART_FROM_CRITIC=True
```

## 💻 Code Examples

### 1. Normal Diagnosis Workflow
```python
from graph.workflow import app
from graph.state import VerifaiState

state = VerifaiState(
    image_path="xray.jpg",
    patient_id="P001",
    _session_id="session-123"
)

result = app.invoke(state)
diagnosis = result['final_diagnosis'].diagnosis
confidence = result['final_diagnosis'].calibrated_confidence
```

### 2. Capture Doctor Feedback
```python
from agents.feedback import capture_doctor_feedback

feedback_id = capture_doctor_feedback(
    session_id="session-123",
    feedback_type="rejection",  # or 'correction', 'approval'
    doctor_notes="Missed bilateral pleural effusion. Patient has CHF.",
    correct_diagnosis="Bilateral pleural effusion, cardiogenic",
    rejection_reasons=["missed_finding", "wrong_primary_diagnosis"],
    doctor_id="dr_smith"
)
```

### 3. Reprocess with Feedback
```python
from agents.feedback import (
    prepare_feedback_for_reprocessing,
    create_feedback_enhanced_state,
    link_feedback_reprocessing_result
)

# Prepare
feedback_input = prepare_feedback_for_reprocessing(feedback_id)

# Create enhanced state
new_state = create_feedback_enhanced_state(
    feedback_input=feedback_input,
    image_path="xray.jpg",
    patient_id="P001"
)

# Reprocess (starts from critic automatically)
new_result = app.invoke(new_state)

# Link results
link_feedback_reprocessing_result(
    feedback_id=feedback_id,
    new_session_id=new_state['_session_id'],
    final_diagnosis=new_result['final_diagnosis'].diagnosis,
    final_confidence=new_result['final_diagnosis'].calibrated_confidence
)
```

### 4. Database Queries

#### Get Session Summary
```python
from db.adapter import get_logger

summary = get_logger.get_session_summary("session-123")
print(summary['session'])
print(summary['feedback'])
```

#### List Recent Sessions
```python
sessions = get_logger.list_sessions(limit=10, status='completed')
for s in sessions:
    print(f"{s['session_id']}: {s['final_diagnosis']}")
```

#### Get Feedback History
```python
feedback_data = get_logger.get_feedback_for_reprocessing(feedback_id)
print(feedback_data['feedback']['doctor_notes'])
print(feedback_data['original_context'])
```

## 🗄️ Database Schema

### Key Tables

```
workflow_sessions
├─ session_id (PK)
├─ final_diagnosis
├─ has_feedback
└─ feedback_status

doctor_feedback
├─ feedback_id (PK)
├─ session_id (FK)
├─ doctor_notes
├─ correct_diagnosis
├─ reprocessed
└─ reprocess_session_id (FK)

agent_invocations
├─ invocation_id (PK)
├─ session_id (FK)
├─ agent_name
└─ is_feedback_iteration
```

## 🔄 Workflow Paths

### Normal Flow
```
START → Radiologist → CheXbert → Evidence → Critic → Debate → Finalize
```

### Feedback Flow
```
START → [routing] → Critic (+feedback) → Debate → Finalize
```

### Routing Logic
```python
# In workflow
def should_start_from_critic(state):
    is_feedback = state.get("is_feedback_iteration", False)
    return "critic_feedback" if is_feedback else "radiologist"
```

## 📊 SQL Queries (Supabase)

### Feedback Statistics
```sql
SELECT 
    feedback_type,
    COUNT(*) as total,
    SUM(CASE WHEN reprocessed THEN 1 ELSE 0 END) as reprocessed_count,
    AVG(original_confidence) as avg_original_conf,
    AVG(reprocess_confidence) as avg_reprocess_conf,
    AVG(reprocess_confidence - original_confidence) as avg_improvement
FROM doctor_feedback
GROUP BY feedback_type;
```

### Recent Feedback Sessions
```sql
SELECT 
    ws.session_id,
    ws.final_diagnosis,
    df.doctor_notes,
    df.correct_diagnosis,
    df.reprocessed,
    df.created_at
FROM workflow_sessions ws
JOIN doctor_feedback df ON ws.session_id = df.session_id
ORDER BY df.created_at DESC
LIMIT 20;
```

### Sessions Needing Review
```sql
SELECT * FROM workflow_sessions
WHERE has_feedback = FALSE
  AND status = 'completed'
  AND final_confidence < 0.75
ORDER BY started_at DESC;
```

## 🔧 Troubleshooting

### Database Connection Failed
```python
# Check health
from db.adapter import check_database_health
health = check_database_health()
print(health)

# Verify environment
import os
print(f"URL: {os.getenv('SUPABASE_URL')}")
print(f"Key: {os.getenv('SUPABASE_KEY')[:20]}...")
print(f"Mode: {os.getenv('DATABASE_MODE')}")
```

### Feedback Not Captured
```python
# Verify feedback enabled
from app.config import settings
print(f"Enabled: {settings.ENABLE_DOCTOR_FEEDBACK}")

# Check session exists
from db.adapter import get_logger
summary = get_logger.get_session_summary("session-id")
assert summary is not None, "Session not found"
```

### Reprocessing Fails
```python
# Check feedback data
feedback_data = get_logger.get_feedback_for_reprocessing(feedback_id)
assert feedback_data is not None, "Feedback not found"

# Verify context preserved
context = feedback_data['original_context']
assert context['invocations'], "No invocations found"
```

## 📂 Important Files

### Core Implementation
- `db/supabase_schema.sql` - Database schema
- `db/adapter.py` - Database switching layer
- `agents/feedback/agent.py` - Feedback logic
- `graph/workflow.py` - Routing & workflow
- `graph/state.py` - State definitions

### Documentation
- `DOCTOR_FEEDBACK_AND_CLOUD_DB_GUIDE.md` - Complete guide
- `IMPLEMENTATION_SUMMARY.md` - Summary
- `FEEDBACK_FLOW_DIAGRAM.md` - Visual diagrams
- `QUICK_REFERENCE.md` - This file

### Utilities
- `setup_helper.py` - CLI utilities
- `.env.example` - Environment template

## 🎯 Best Practices

### Database
✅ Use Supabase for production
✅ Use SQLite for local development
✅ Switch via `DATABASE_MODE` in `.env`
✅ Run `check-db` before deployment

### Feedback
✅ Capture detailed doctor notes
✅ Include correct diagnosis when known
✅ Tag with rejection reasons
✅ Link reprocessing results

### Testing
✅ Test with `setup_helper.py test-feedback`
✅ Verify connection with `check-db`
✅ Monitor with `stats` command
✅ Check logs in Supabase dashboard

## 🔐 Security

### Supabase Keys
```bash
# .env (safe for client-side)
SUPABASE_KEY=your_anon_key_here

# .env (keep secret! server-side only)
SUPABASE_SERVICE_KEY=your_service_role_key_here
```

### Row Level Security (Optional)
```sql
-- Enable RLS
ALTER TABLE workflow_sessions ENABLE ROW LEVEL SECURITY;

-- Policy example
CREATE POLICY "Users see own sessions"
ON workflow_sessions FOR SELECT
USING (auth.uid()::text = doctor_id);
```

## 📞 Support Checklist

- [ ] Check `.env` has correct values
- [ ] Run `python setup_helper.py check-db`
- [ ] Verify Supabase project is active
- [ ] Confirm schema is created
- [ ] Test with known good session ID
- [ ] Review trace logs in database

## 🎓 Learning Path

1. **Setup**: Create Supabase project, configure `.env`
2. **Test**: Run `check-db` and `test-feedback`
3. **Run**: Execute normal workflow
4. **Feedback**: Capture rejection with feedback
5. **Reprocess**: Run feedback-enhanced workflow
6. **Analyze**: Query results, measure improvement

## 📈 Metrics to Track

```python
from db.adapter import get_logger

stats = get_logger.get_diagnosis_stats()

print(f"Total sessions: {stats['total_sessions']}")
print(f"Average confidence: {stats['avg_confidence']:.2%}")
print(f"Debate consensus rate: {stats['debate_consensus_rate']:.1%}")

# Custom query for feedback impact
# (see SQL examples above)
```

---

**Quick Start**: 
1. `cp .env.example .env` 
2. Add Supabase credentials
3. `python setup_helper.py check-db`
4. Start using feedback loop!
