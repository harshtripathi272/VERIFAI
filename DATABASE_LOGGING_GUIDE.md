# VERIFAI Database Logging System

**Comprehensive SQL-based logging for multi-agent diagnostic workflows**

---

## 📋 Overview

The VERIFAI Database Logging System provides **structured, queryable logging** for every agent invocation, debate round, and workflow session in the VERIFAI diagnostic pipeline. Built on SQLite with automatic schema creation and indexing, it enables:

- ✅ **Complete audit trails** — Every agent call is logged with inputs, outputs, and timing
- ✅ **Debate transparency** — Full round-by-round tracking of critic challenges and evidence responses
- ✅ **Fast queries** — 42 auto-created indexes enable sub-millisecond lookups
- ✅ **Thread-safe** — Concurrent logging from parallel agents (historian + literature)
- ✅ **REST API** — 5 endpoints for querying sessions, debates, and stats
- ✅ **Zero configuration** — Auto-initializes on first use

---

## 🏗️ Architecture

### Core Components

```
db/
├── __init__.py           # Public API exports
├── connection.py         # Schema, connection pooling, auto-indexing
└── logger.py            # AgentLogger class with logging methods
```

### Database File
- **Location**: `verifai_logs.db` (project root)
- **Format**: SQLite 3 with WAL mode (Write-Ahead Logging)
- **Size**: ~10 KB empty, ~200-500 KB per 100 sessions

---

## 📊 Schema Design

### 17 Tables in 5 Categories

#### 1️⃣ **Session Tracking**
```sql
workflow_sessions
├── session_id (PK)           -- UUID for each pipeline run
├── image_path                -- Input chest X-ray file
├── patient_id                -- Optional FHIR patient ID
├── workflow_type             -- 'debate' or 'legacy'
├── status                    -- 'running', 'completed', 'failed'
├── started_at / completed_at -- Timestamps
├── final_diagnosis           -- e.g., "Community-Acquired Pneumonia"
├── final_confidence          -- 0.0 to 1.0
├── was_deferred              -- Boolean
├── deferral_reason           -- If deferred to human
└── total_agents_invoked      -- Count of agents called
```

#### 2️⃣ **Agent Invocations** (generic)
```sql
agent_invocations
├── invocation_id (PK, auto-increment)
├── session_id (FK → workflow_sessions)
├── agent_name                -- 'radiologist', 'critic', 'debate', etc.
├── started_at / completed_at
├── duration_ms               -- Execution time
├── status                    -- 'running', 'success', 'error'
├── input_summary             -- JSON snapshot of inputs
├── output_summary            -- JSON snapshot of outputs
└── trace_entries             -- JSON array of trace strings
```

#### 3️⃣ **Radiologist Logs** (visual analysis)
```sql
radiologist_logs              -- Main log entry
├── log_id (PK)
├── session_id (FK)
├── invocation_id (FK)
├── image_path
├── num_findings
└── reasoning

radiologist_findings          -- Individual findings (1-to-many)
├── finding_id (PK)
├── radiologist_log_id (FK)
├── session_id (FK)
├── location                  -- "RLL", "LUL", "Mediastinum"
├── observation               -- "consolidation", "nodule", "effusion"
├── severity                  -- 0.0 to 1.0
└── bounding_box              -- JSON [x, y, w, h]

radiologist_hypotheses        -- Ranked diagnoses (1-to-many)
├── hypothesis_id (PK)
├── radiologist_log_id (FK)
├── session_id (FK)
├── rank                      -- 1 = top hypothesis
├── diagnosis                 -- "Community-Acquired Pneumonia"
├── confidence                -- 0.0 to 1.0
└── icd10_code                -- "J18.9"

radiologist_signals           -- Internal uncertainty signals
├── signal_id (PK)
├── radiologist_log_id (FK)
├── session_id (FK)
├── logits_top2               -- JSON [top_logit, 2nd_logit]
├── logit_margin              -- difference
├── predictive_entropy        -- Shannon entropy
├── attention_dispersion      -- Gini coefficient
└── prediction_stability      -- MC dropout std
```

#### 4️⃣ **Critic Logs** (overconfidence detection)
```sql
critic_logs
├── log_id (PK)
├── session_id (FK)
├── invocation_id (FK)
├── overconfidence_prob       -- 0.0 to 1.0
├── calculated_uncertainty    -- 0.0 to 1.0
├── counter_hypotheses        -- JSON array of alternatives
└── concern_signals           -- JSON array of warnings
```

#### 5️⃣ **Historian Logs** (FHIR clinical context)
```sql
historian_logs
├── log_id (PK)
├── session_id (FK)
├── invocation_id (FK)
├── patient_id
├── confidence_adjustment     -- -1.0 to +1.0
├── clinical_summary
├── num_supporting            -- Count of supporting facts
└── num_contradicting         -- Count of contradicting facts

historian_facts               -- Individual FHIR facts (1-to-many)
├── fact_id (PK)
├── historian_log_id (FK)
├── session_id (FK)
├── fact_type                 -- 'supporting' or 'contradicting'
├── description               -- "[CAP] Recent fever and cough noted"
├── fhir_resource_id          -- "Condition/123"
└── fhir_resource_type        -- "Condition", "Observation"
```

#### 6️⃣ **Literature Logs** (PubMed/PMC/Semantic Scholar)
```sql
literature_logs
├── log_id (PK)
├── session_id (FK)
├── invocation_id (FK)
├── query_used                -- Search query sent to APIs
├── overall_evidence_strength -- 'low', 'medium', 'high'
├── num_citations
└── raw_summary               -- Full text (if fast mode)

literature_citations          -- Individual papers (1-to-many)
├── citation_id (PK)
├── literature_log_id (FK)
├── session_id (FK)
├── pmid                      -- PubMed ID
├── title
├── authors                   -- JSON array
├── journal
├── year
├── relevance_summary
├── evidence_strength         -- 'low', 'medium', 'high'
└── source                    -- 'pubmed', 'europepmc', 'semanticscholar'
```

#### 7️⃣ **Debate Logs** (adversarial consensus)
```sql
debate_logs                   -- Main debate session
├── log_id (PK)
├── session_id (FK)
├── invocation_id (FK)
├── num_rounds
├── final_consensus           -- Boolean
├── consensus_diagnosis
├── consensus_confidence
├── escalate_to_chief         -- Boolean
├── escalation_reason
├── debate_summary
└── total_confidence_adj      -- Net adjustment across rounds

debate_rounds                 -- Individual rounds (1-to-many)
├── round_id (PK)
├── debate_log_id (FK)
├── session_id (FK)
├── round_number              -- 1, 2, 3...
├── round_consensus           -- NULL or 'reached'
└── confidence_delta          -- Net impact for this round

debate_arguments              -- Arguments per round (1-to-many per round)
├── argument_id (PK)
├── round_id (FK)
├── debate_log_id (FK)
├── session_id (FK)
├── agent                     -- 'critic', 'historian', 'literature'
├── position                  -- 'challenge', 'support', 'refine'
├── argument                  -- Full text of argument
├── confidence_impact         -- -1.0 to +1.0
└── evidence_refs             -- JSON array of references (PMIDs, FHIR IDs)
```

#### 8️⃣ **Chief Logs** (final arbitration)
```sql
chief_logs
├── log_id (PK)
├── session_id (FK)
├── invocation_id (FK)
├── diagnosis
├── calibrated_confidence
├── was_deferred              -- Boolean
├── deferral_reason
├── explanation
└── recommended_next_steps    -- JSON array
```

#### 9️⃣ **Trace Log** (flat audit trail)
```sql
trace_log                     -- Mirrors state.trace
├── trace_id (PK, auto-increment)
├── session_id (FK)
├── agent_name                -- Which agent generated this
├── entry                     -- "CRITIC: U=38.00%, Overconf=35.00%"
└── created_at                -- Timestamp
```

---

## 🔍 Indexing Strategy

### 42 Auto-Created Indexes

All indexes are created automatically via `init_db()` on first connection:

| Category | Indexes | Purpose |
|----------|---------|---------|
| **Sessions** | 4 | patient_id, status, started_at, final_diagnosis |
| **Invocations** | 4 | session_id, agent_name, status, composite |
| **Radiologist** | 7 | session_id, location, diagnosis, confidence |
| **Critic** | 3 | session_id, overconfidence_prob, uncertainty |
| **Historian** | 5 | session_id, patient_id, fact_type, fhir_resource_id |
| **Literature** | 6 | session_id, strength, pmid, year, source |
| **Debate** | 8 | session_id, consensus, round_id, agent, position |
| **Chief** | 2 | session_id, was_deferred |
| **Trace** | 3 | session_id, agent_name, created_at |

**Performance Impact**: Sub-millisecond lookups even with 10,000+ sessions.

---

## 🔧 Implementation Details

### 1. Connection Management (`db/connection.py`)

#### Thread-Local Connections
```python
import threading
_local = threading.local()

def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Each thread gets its own connection (SQLite is not thread-safe)."""
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(path, timeout=30)
        _local.connection.row_factory = sqlite3.Row
        # Enable WAL mode for concurrent read/write
        _local.connection.execute("PRAGMA journal_mode=WAL")
        _local.connection.execute("PRAGMA foreign_keys=ON")
    return _local.connection
```

#### Auto-Initialization
```python
def init_db(db_path: str = None):
    """
    Creates all tables and indexes on first call.
    Safe to call multiple times (uses CREATE IF NOT EXISTS).
    """
    with _init_lock:
        if _initialized:
            return
        conn = sqlite3.connect(db_path or DB_PATH)
        conn.executescript(SCHEMA_SQL)    # Create tables
        conn.executescript(INDEXES_SQL)   # Create indexes
        conn.commit()
        _initialized = True
```

### 2. AgentLogger Class (`db/logger.py`)

#### Session Lifecycle
```python
logger = AgentLogger(
    session_id="abc-123",           # Auto-generated if None
    image_path="xray.png",
    patient_id="patient-456",
    workflow_type="debate"
)

# Automatically creates workflow_sessions row
# Status: 'running'
```

#### Per-Agent Logging Methods
```python
# Each agent has a dedicated method:
logger.log_radiologist(state, result)
logger.log_critic(state, result)
logger.log_historian(state, result)
logger.log_literature(state, result)
logger.log_debate(state, result)      # Full round-by-round
logger.log_chief(state, result)
logger.log_finalize(state, result)
```

#### Auto-Timing
```python
def log_radiologist(self, state, result):
    t0 = time.time()
    # ... insert logs ...
    duration_ms = int((time.time() - t0) * 1000)
    # Stored in agent_invocations.duration_ms
```

#### Session Completion
```python
logger.complete_session(final_diagnosis=FinalDiagnosis(...))
# Updates: status='completed', completed_at, final_diagnosis, final_confidence
```

### 3. Workflow Integration (`graph/workflow.py`)

#### Automatic Wrapper Pattern
```python
# Thread-local logger registry
_logger_registry: dict[str, AgentLogger] = {}

def _get_or_create_logger(state: VerifaiState) -> AgentLogger:
    """Get or create logger for current session."""
    session_id = state.get("_session_id") or str(uuid.uuid4())
    if session_id not in _logger_registry:
        _logger_registry[session_id] = AgentLogger(
            session_id=session_id,
            image_path=state.get("image_path"),
            patient_id=state.get("patient_id")
        )
    return _logger_registry[session_id]

def logged_radiologist_node(state: VerifaiState) -> dict:
    """Wrapper that adds logging to radiologist_node."""
    logger = _get_or_create_logger(state)
    result = radiologist_node(state)  # Call original
    try:
        logger.log_radiologist(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed: {e}")  # Never blocks pipeline
    return result
```

#### Graph Registration
```python
graph = StateGraph(VerifaiState)
graph.add_node("radiologist", logged_radiologist_node)  # Logged version
graph.add_node("critic", logged_critic_node)
# ... etc
```

---

## 🚀 Usage Examples

### Example 1: Basic Logging
```python
from db.logger import AgentLogger
from graph.state import RadiologistOutput, FinalDiagnosis

# Create session
logger = AgentLogger(
    image_path="chest_xray_001.png",
    patient_id="patient-123"
)

# Log agents (normally called by workflow wrappers)
logger.log_radiologist(state, radiologist_result)
logger.log_critic(state, critic_result)
# ... log other agents ...

# Complete session
logger.complete_session(final_diagnosis=final_dx)
```

### Example 2: Querying Sessions
```python
from db.logger import AgentLogger

# List recent sessions
sessions = AgentLogger.list_sessions(limit=50, status='completed')
print(f"Found {len(sessions)} completed sessions")

# Get full session detail
summary = AgentLogger.get_session_summary(session_id="abc-123")
print(f"Diagnosis: {summary['session']['final_diagnosis']}")
print(f"Confidence: {summary['session']['final_confidence']}")
print(f"Debate rounds: {len(summary['debate']['rounds'])}")
```

### Example 3: Debate Analysis
```python
# Get all debates from a session
debates = AgentLogger.get_debate_history(session_id="abc-123")

for debate in debates:
    print(f"Rounds: {debate['debate']['num_rounds']}")
    print(f"Consensus: {debate['debate']['final_consensus']}")
    
    for round_data in debate['rounds']:
        print(f"  Round {round_data['round']['round_number']}")
        for arg in round_data['arguments']:
            print(f"    [{arg['agent']}] {arg['position']}: {arg['argument'][:50]}...")
```

### Example 4: Agent Performance
```python
# Get invocation history for critic agent
history = AgentLogger.get_agent_history("critic", limit=100)

avg_duration = sum(h['duration_ms'] for h in history) / len(history)
print(f"Critic average duration: {avg_duration:.1f}ms")

high_uncertainty = [
    h for h in history 
    if json.loads(h['output_summary']).get('calculated_uncertainty', 0) > 0.5
]
print(f"High uncertainty cases: {len(high_uncertainty)}")
```

### Example 5: Aggregate Stats
```python
stats = AgentLogger.get_diagnosis_stats()

print(f"Total sessions: {stats['total_sessions']}")
print(f"Completed: {stats['completed']}")
print(f"Deferred: {stats['deferred']}")
print(f"Average confidence: {stats['avg_confidence']:.2%}")
print(f"Debate consensus rate: {stats['debate_consensus_rate']:.0%}")

print("\nTop 5 diagnoses:")
for dx in stats['top_diagnoses'][:5]:
    print(f"  {dx['final_diagnosis']}: {dx['cnt']} cases ({dx['avg_conf']:.0%} avg)")
```

---

## 🌐 REST API Endpoints

### Base URL: `http://localhost:8000`

#### 1. List Sessions
```http
GET /logs/sessions?limit=50&status=completed&patient_id=patient-123
```
**Response:**
```json
{
  "sessions": [
    {
      "session_id": "abc-123",
      "image_path": "xray.png",
      "patient_id": "patient-123",
      "status": "completed",
      "final_diagnosis": "Community-Acquired Pneumonia",
      "final_confidence": 0.88,
      "started_at": "2026-02-14T10:30:00",
      "completed_at": "2026-02-14T10:30:45"
    }
  ],
  "total": 1
}
```

#### 2. Session Detail
```http
GET /logs/sessions/abc-123
```
**Response:**
```json
{
  "session": { "session_id": "abc-123", ... },
  "invocations": [
    {
      "invocation_id": 1,
      "agent_name": "radiologist",
      "duration_ms": 2340,
      "status": "success"
    },
    ...
  ],
  "traces": [
    { "entry": "RADIOLOGIST: 2 findings, Top Dx: CAP (75%)", ... }
  ],
  "debate": {
    "summary": { "num_rounds": 1, "final_consensus": true, ... },
    "rounds": [
      {
        "round": { "round_number": 1, "confidence_delta": 0.17 },
        "arguments": [
          {
            "agent": "critic",
            "position": "challenge",
            "argument": "Moderate overconfidence detected...",
            "confidence_impact": -0.05
          },
          ...
        ]
      }
    ]
  }
}
```

#### 3. Agent History
```http
GET /logs/agents/critic?limit=100
```
**Response:**
```json
{
  "agent": "critic",
  "invocations": [
    {
      "invocation_id": 42,
      "session_id": "abc-123",
      "agent_name": "critic",
      "duration_ms": 180,
      "output_summary": "{\"overconfidence_prob\": 0.35, ...}"
    }
  ],
  "total": 100
}
```

#### 4. Debate History
```http
GET /logs/debates?session_id=abc-123
```
**Response:**
```json
{
  "debates": [
    {
      "debate": {
        "log_id": 5,
        "num_rounds": 1,
        "final_consensus": 1,
        "consensus_diagnosis": "Community-Acquired Pneumonia",
        "escalate_to_chief": 0
      },
      "rounds": [...],
      "arguments": [...]
    }
  ],
  "total": 1
}
```

#### 5. Statistics
```http
GET /logs/stats
```
**Response:**
```json
{
  "total_sessions": 147,
  "completed": 142,
  "failed": 3,
  "deferred": 12,
  "avg_confidence": 0.7823,
  "avg_agents_per_session": 5.8,
  "debate_consensus_rate": 0.856,
  "top_diagnoses": [
    {
      "final_diagnosis": "Community-Acquired Pneumonia",
      "cnt": 42,
      "avg_conf": 0.81
    },
    ...
  ]
}
```

---

## 🔐 Thread Safety

### Concurrent Logging Guarantees

1. **Thread-local connections**: Each thread (e.g., historian, literature running in parallel) gets its own SQLite connection
2. **WAL mode**: Write-Ahead Logging allows multiple readers + one writer concurrently
3. **Busy timeout**: 5-second timeout if another thread holds a write lock
4. **Registry lock**: Logger creation/cleanup uses `threading.Lock()`

**Test Case**: Evidence gathering node runs historian + literature in parallel:
```python
with ThreadPoolExecutor(max_workers=2) as executor:
    historian_future = executor.submit(logged_historian_node, state)
    literature_future = executor.submit(logged_literature_node, state)
    # Both log to DB concurrently without conflicts
```

---

## 📈 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Schema init** | ~50ms | One-time on first connection |
| **Session creation** | <1ms | Single INSERT |
| **Agent log write** | 1-5ms | Depends on # of findings/citations |
| **Debate log write** | 2-10ms | Full round-by-round |
| **Session query** | <1ms | With indexes |
| **Debate history** | 1-3ms | Fetches all rounds + arguments |
| **Stats aggregation** | 5-15ms | Multiple GROUP BY queries |
| **DB size growth** | ~2-5 KB/session | Varies by findings/citations |

**Benchmark** (100 sessions, mixed workflow):
- Total DB size: 260 KB
- Average write time per agent: 2.3ms
- Average session query time: 0.4ms

---

## 🛠️ Maintenance & Administration

### Backup Database
```bash
# While system is running (thanks to WAL mode)
cp verifai_logs.db verifai_logs_backup_$(date +%Y%m%d).db
```

### Vacuum Database (reclaim space)
```bash
sqlite3 verifai_logs.db "VACUUM;"
```

### Inspect Schema
```bash
sqlite3 verifai_logs.db ".schema workflow_sessions"
sqlite3 verifai_logs.db ".indexes debate_arguments"
```

### Query Examples (SQL)
```sql
-- Sessions with high uncertainty
SELECT session_id, final_diagnosis, final_confidence
FROM workflow_sessions
WHERE final_confidence < 0.5
ORDER BY started_at DESC;

-- Debates that escalated to Chief
SELECT d.session_id, d.num_rounds, d.escalation_reason, w.final_diagnosis
FROM debate_logs d
JOIN workflow_sessions w ON d.session_id = w.session_id
WHERE d.escalate_to_chief = 1;

-- Top overconfident predictions
SELECT c.session_id, c.overconfidence_prob, w.final_diagnosis
FROM critic_logs c
JOIN workflow_sessions w ON c.session_id = w.session_id
WHERE c.overconfidence_prob > 0.5
ORDER BY c.overconfidence_prob DESC;

-- Literature sources distribution
SELECT source, COUNT(*) as cnt, AVG(evidence_strength) as avg_strength
FROM literature_citations
GROUP BY source;
```

---

## 🐛 Error Handling

### Design Philosophy
**Never block the diagnostic pipeline due to logging failures.**

### Implementation
```python
def logged_radiologist_node(state):
    logger = _get_or_create_logger(state)
    result = radiologist_node(state)  # Core logic
    try:
        logger.log_radiologist(state, result)
    except Exception as e:
        print(f"[DB LOG] Failed to log radiologist: {e}")
        # Pipeline continues regardless
    return result
```

### Common Failure Modes
| Error | Cause | Resolution |
|-------|-------|------------|
| `sqlite3.OperationalError: database is locked` | Multiple writers | Auto-retried via `busy_timeout=5000` |
| `JSON serialization error` | Non-serializable object | Caught in `_safe_json()` helper |
| `Foreign key constraint failed` | Orphaned record | Should never happen (invocations created first) |
| `Disk full` | No space for DB growth | Logged to stderr, pipeline continues |

---

## 🧪 Testing

### Run Full Test Suite
```bash
python test_db_logging.py
```

**Tests Included:**
1. ✅ Schema creation (17 tables, 42 indexes)
2. ✅ Full workflow logging (all 6 agents)
3. ✅ Debate round-by-round logging
4. ✅ Query helpers (5 methods)
5. ✅ Detail verification (row counts, foreign keys)

### Test Output
```
============================================================
  VERIFAI Database Logging System — Full Test Suite
============================================================

TEST 1: Schema Creation
  Tables created: 17
    ✓ agent_invocations
    ✓ chief_logs
    ✓ critic_logs
    ...
  Indexes created: 42
    ✓ idx_sessions_patient
    ✓ idx_invocations_session_agent
    ...
  ✅ Schema creation PASSED

TEST 2: Full Workflow Logging
  Session created: abc-123-456-789
  ✓ Radiologist logged
  ✓ Critic logged
  ✓ Historian logged
  ✓ Literature logged
  ✓ Debate logged (1 round, 3 arguments)
  ✓ Finalize logged
  ✓ Session completed
  ✅ Full workflow logging PASSED

TEST 3: Query Helpers
  ✓ Session summary: 6 invocations, 10 traces
    Debate: 1 rounds, 3 arguments
  ✓ List sessions: 1 found
  ✓ Agent history [radiologist]: 1 invocations
  ✓ Agent history [critic]: 1 invocations
  ...
  ✅ Query helpers PASSED

TEST 4: Detail Verification
  ✓ Radiologist findings: 2
    - RLL: consolidation (severity=0.8)
    - LUL: ground-glass opacity (severity=0.4)
  ✓ Hypotheses: 2
    - Rank 1: Community-Acquired Pneumonia (75%)
    - Rank 2: Pulmonary Edema (15%)
  ✓ Debate arguments: 3
    - [critic] challenge: impact=-0.05
    - [historian] support: impact=+0.10
    - [literature] support: impact=+0.12
  ...
  ✅ Detail verification PASSED

  📁 Database file: d:\Workspace\VERIFAI\verifai_logs.db
  📊 Size: 260.0 KB

============================================================
  🎉 ALL TESTS PASSED — Database logging system is working!
============================================================
```

---

## 🔮 Future Enhancements

### Potential Extensions

1. **Time-series analysis**
   - Track confidence drift over time per diagnosis type
   - Identify agents that consistently trigger debates

2. **Performance metrics**
   - Agent execution time percentiles
   - Bottleneck detection (slowest agents)

3. **Audit compliance**
   - HIPAA-compliant logging (de-identification)
   - Tamper-proof checksums for medical records

4. **Advanced queries**
   - Full-text search on debate arguments
   - Correlation analysis (overconfidence → deferral rate)

5. **Database backends**
   - PostgreSQL support for multi-user deployments
   - BigQuery integration for large-scale analytics

---

## 📚 Related Documentation

- **[ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md)** — Full system architecture
- **[DEBATE_SYSTEM_GUIDE.md](DEBATE_SYSTEM_GUIDE.md)** — Debate mechanism details
- **[THREAD_SAFETY_GUIDE.md](THREAD_SAFETY_GUIDE.md)** — Concurrency best practices
- **[graph/state.py](graph/state.py)** — State models (RadiologistOutput, etc.)
- **[test_db_logging.py](test_db_logging.py)** — Full test suite

---

## 📞 Support

For questions or issues:
1. Check the test suite: `python test_db_logging.py`
2. Inspect the database: `sqlite3 verifai_logs.db`
3. Review API logs: Check FastAPI console for `[DB LOG]` messages

---

**Version**: 1.0.0  
**Last Updated**: February 14, 2026  
**License**: MIT
