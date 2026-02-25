"""
VERIFAI Database Connection Manager

SQLite-based structured logging with automatic schema creation and indexing.
All tables are created on first connection with proper indexes for fast queries.
"""

import sqlite3
import threading
import os
from contextlib import contextmanager
from datetime import datetime

# Thread-local storage for connections (SQLite is not thread-safe by default)
_local = threading.local()

# Default database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "verifai_logs.db")

# Lock for schema initialization
_init_lock = threading.Lock()
_initialized = False


# =============================================================================
# SCHEMA DEFINITION
# =============================================================================

SCHEMA_SQL = """
-- ============================================================
-- 1. WORKFLOW SESSIONS — one row per full pipeline invocation
-- ============================================================
CREATE TABLE IF NOT EXISTS workflow_sessions (
    session_id          TEXT PRIMARY KEY,
    image_path          TEXT NOT NULL,
    patient_id          TEXT,
    workflow_type       TEXT DEFAULT 'debate',   -- 'debate' or 'legacy'
    status              TEXT DEFAULT 'running',  -- 'running', 'completed', 'failed'
    started_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at        TIMESTAMP,
    final_diagnosis     TEXT,
    final_confidence    REAL,
    was_deferred        INTEGER DEFAULT 0,
    deferral_reason     TEXT,
    total_agents_invoked INTEGER DEFAULT 0,
    error_message       TEXT,
    uncertainty_cascade TEXT                     -- JSON array: [{agent, system_uncertainty}, ...] full MUC cascade audit trail
);

-- ============================================================
-- 2. AGENT INVOCATIONS — one row per agent call within a session
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_invocations (
    invocation_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    agent_name          TEXT NOT NULL,            -- 'radiologist', 'critic', 'historian', 'literature', 'debate', 'chief', 'finalize'
    started_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at        TIMESTAMP,
    duration_ms         INTEGER,
    status              TEXT DEFAULT 'running',   -- 'running', 'success', 'error'
    error_message       TEXT,
    input_summary       TEXT,                     -- JSON summary of inputs
    output_summary      TEXT,                     -- JSON summary of outputs
    trace_entries       TEXT,                     -- JSON array of trace strings
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id)
);

-- ============================================================
-- 3. RADIOLOGIST LOGS — plain-text findings & impression + KLE uncertainty
-- ============================================================
CREATE TABLE IF NOT EXISTS radiologist_logs (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_path          TEXT,
    findings_text       TEXT,                     -- Full FINDINGS section (plain text)
    impression_text     TEXT,                     -- Full IMPRESSION section (plain text)
    kle_uncertainty     REAL,                     -- KLE epistemic uncertainty score
    num_samples         INTEGER DEFAULT 1,        -- Number of samples used for KLE
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id),
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id)
);

-- ============================================================
-- 4. CRITIC LOGS — overconfidence detection via KLE consistency
-- ============================================================
CREATE TABLE IF NOT EXISTS critic_logs (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_overconfident    INTEGER DEFAULT 0,        -- boolean: text assertiveness vs KLE uncertainty
    safety_score        REAL,                     -- 0.0 to 1.0, overall safety/appropriateness
    concern_flags       TEXT,                     -- JSON array of specific concern strings
    recommended_hedging TEXT,                     -- Suggested rephrasing (nullable)
    kle_uncertainty_input REAL,                   -- KLE score that critic evaluated against
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id),
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id)
);

-- ============================================================
-- 5. HISTORIAN LOGS — FHIR-based clinical context
-- ============================================================
CREATE TABLE IF NOT EXISTS historian_logs (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    patient_id          TEXT,
    confidence_adjustment REAL,
    clinical_summary    TEXT,
    num_supporting      INTEGER DEFAULT 0,
    num_contradicting   INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id),
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id)
);

CREATE TABLE IF NOT EXISTS historian_facts (
    fact_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    historian_log_id    INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    fact_type           TEXT NOT NULL,             -- 'supporting' or 'contradicting'
    description         TEXT NOT NULL,
    fhir_resource_id    TEXT,
    fhir_resource_type  TEXT,
    FOREIGN KEY (historian_log_id) REFERENCES historian_logs(log_id),
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id)
);

-- ============================================================
-- 6. LITERATURE LOGS — PubMed/PMC/SemanticScholar results
-- ============================================================
CREATE TABLE IF NOT EXISTS literature_logs (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query_used          TEXT,
    overall_evidence_strength TEXT,                -- 'low', 'medium', 'high'
    num_citations       INTEGER DEFAULT 0,
    raw_summary         TEXT,                     -- Full text summary if fast mode
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id),
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id)
);

CREATE TABLE IF NOT EXISTS literature_citations (
    citation_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    literature_log_id   INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    pmid                TEXT,
    title               TEXT,
    authors             TEXT,                     -- JSON array
    journal             TEXT,
    year                INTEGER,
    relevance_summary   TEXT,
    evidence_strength   TEXT,                     -- 'low', 'medium', 'high'
    source              TEXT,                     -- 'pubmed', 'europepmc', 'semanticscholar'
    FOREIGN KEY (literature_log_id) REFERENCES literature_logs(log_id),
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id)
);

-- ============================================================
-- 7. DEBATE LOGS — full debate rounds and arguments
-- ============================================================
CREATE TABLE IF NOT EXISTS debate_logs (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    num_rounds          INTEGER DEFAULT 0,
    final_consensus     INTEGER DEFAULT 0,        -- boolean
    consensus_diagnosis TEXT,
    consensus_confidence REAL,
    escalate_to_chief   INTEGER DEFAULT 0,        -- boolean
    escalation_reason   TEXT,
    debate_summary      TEXT,
    total_confidence_adj REAL DEFAULT 0.0,
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id),
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id)
);

CREATE TABLE IF NOT EXISTS debate_rounds (
    round_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_log_id       INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    round_number        INTEGER NOT NULL,
    round_consensus     TEXT,                     -- NULL if no consensus, 'reached' if yes
    confidence_delta    REAL DEFAULT 0.0,
    FOREIGN KEY (debate_log_id) REFERENCES debate_logs(log_id),
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS debate_arguments (
    argument_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id            INTEGER NOT NULL,
    debate_log_id       INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    agent               TEXT NOT NULL,             -- 'critic', 'historian', 'literature'
    position            TEXT NOT NULL,             -- 'challenge', 'support', 'refine'
    argument            TEXT NOT NULL,
    confidence_impact   REAL DEFAULT 0.0,
    evidence_refs       TEXT,                     -- JSON array of reference strings
    FOREIGN KEY (round_id) REFERENCES debate_rounds(round_id),
    FOREIGN KEY (debate_log_id) REFERENCES debate_logs(log_id),
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id)
);

-- ============================================================
-- 8. CHIEF LOGS — final arbitration decisions
-- ============================================================
CREATE TABLE IF NOT EXISTS chief_logs (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    diagnosis           TEXT,
    calibrated_confidence REAL,
    was_deferred        INTEGER DEFAULT 0,
    deferral_reason     TEXT,
    explanation         TEXT,
    recommended_next_steps TEXT,                  -- JSON array
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id),
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id)
);

-- ============================================================
-- 9. TRACE LOG — flat audit trail (mirrors state.trace)
-- ============================================================
CREATE TABLE IF NOT EXISTS trace_log (
    trace_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    agent_name          TEXT,
    entry               TEXT NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id)
);
"""

# =============================================================================
# INDEX DEFINITIONS — auto-created for fast lookups
# =============================================================================

INDEXES_SQL = """
-- Session indexes
CREATE INDEX IF NOT EXISTS idx_sessions_patient       ON workflow_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status        ON workflow_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_started       ON workflow_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_diagnosis     ON workflow_sessions(final_diagnosis);

-- Agent invocation indexes
CREATE INDEX IF NOT EXISTS idx_invocations_session    ON agent_invocations(session_id);
CREATE INDEX IF NOT EXISTS idx_invocations_agent      ON agent_invocations(agent_name);
CREATE INDEX IF NOT EXISTS idx_invocations_status     ON agent_invocations(status);
CREATE INDEX IF NOT EXISTS idx_invocations_session_agent ON agent_invocations(session_id, agent_name);

-- Radiologist indexes
CREATE INDEX IF NOT EXISTS idx_rad_logs_session       ON radiologist_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_rad_logs_kle           ON radiologist_logs(kle_uncertainty);

-- Critic indexes
CREATE INDEX IF NOT EXISTS idx_critic_session         ON critic_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_critic_overconf        ON critic_logs(is_overconfident);
CREATE INDEX IF NOT EXISTS idx_critic_safety          ON critic_logs(safety_score);

-- Historian indexes
CREATE INDEX IF NOT EXISTS idx_historian_session      ON historian_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_historian_patient      ON historian_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_historian_facts_session ON historian_facts(session_id);
CREATE INDEX IF NOT EXISTS idx_historian_facts_type   ON historian_facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_historian_facts_fhir   ON historian_facts(fhir_resource_id);

-- Literature indexes
CREATE INDEX IF NOT EXISTS idx_lit_session            ON literature_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_lit_strength           ON literature_logs(overall_evidence_strength);
CREATE INDEX IF NOT EXISTS idx_lit_citations_session  ON literature_citations(session_id);
CREATE INDEX IF NOT EXISTS idx_lit_citations_pmid     ON literature_citations(pmid);
CREATE INDEX IF NOT EXISTS idx_lit_citations_year     ON literature_citations(year);
CREATE INDEX IF NOT EXISTS idx_lit_citations_source   ON literature_citations(source);

-- Debate indexes
CREATE INDEX IF NOT EXISTS idx_debate_session         ON debate_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_debate_consensus       ON debate_logs(final_consensus);
CREATE INDEX IF NOT EXISTS idx_debate_rounds_session  ON debate_rounds(session_id);
CREATE INDEX IF NOT EXISTS idx_debate_rounds_log      ON debate_rounds(debate_log_id);
CREATE INDEX IF NOT EXISTS idx_debate_args_round      ON debate_arguments(round_id);
CREATE INDEX IF NOT EXISTS idx_debate_args_session    ON debate_arguments(session_id);
CREATE INDEX IF NOT EXISTS idx_debate_args_agent      ON debate_arguments(agent);
CREATE INDEX IF NOT EXISTS idx_debate_args_position   ON debate_arguments(position);

-- Chief indexes
CREATE INDEX IF NOT EXISTS idx_chief_session          ON chief_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_chief_deferred         ON chief_logs(was_deferred);

-- Trace indexes
CREATE INDEX IF NOT EXISTS idx_trace_session          ON trace_log(session_id);
CREATE INDEX IF NOT EXISTS idx_trace_agent            ON trace_log(agent_name);
CREATE INDEX IF NOT EXISTS idx_trace_created          ON trace_log(created_at);
"""


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """
    Get a thread-local SQLite connection.
    
    Each thread gets its own connection to avoid threading issues.
    Connections are reused within the same thread.
    """
    path = db_path or DB_PATH
    
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(path, timeout=30)
        _local.connection.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read/write performance
        _local.connection.execute("PRAGMA journal_mode=WAL")
        _local.connection.execute("PRAGMA foreign_keys=ON")
        _local.connection.execute("PRAGMA busy_timeout=5000")
    
    return _local.connection


@contextmanager
def get_db(db_path: str = None):
    """
    Context manager for database operations with automatic commit/rollback.
    
    Usage:
        with get_db() as conn:
            conn.execute("INSERT INTO ...")
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(db_path: str = None):
    """
    Initialize the database schema and indexes.
    
    Safe to call multiple times — uses CREATE IF NOT EXISTS.
    Called automatically on first AgentLogger instantiation.
    """
    global _initialized
    
    with _init_lock:
        if _initialized:
            return
        
        path = db_path or DB_PATH
        conn = sqlite3.connect(path, timeout=30)
        
        try:
            # Create all tables
            conn.executescript(SCHEMA_SQL)
            
            # Create all indexes
            conn.executescript(INDEXES_SQL)

            # ── Migrations for existing DBs ──────────────────────────────
            # Idempotent: ALTER TABLE is wrapped in try/except since SQLite
            # has no IF NOT EXISTS for columns prior to v3.37.
            try:
                conn.execute(
                    "ALTER TABLE workflow_sessions ADD COLUMN "
                    "uncertainty_cascade TEXT"
                )
                conn.commit()
                print("[VERIFAI DB] Migration: added 'uncertainty_cascade' column")
            except Exception:
                pass  # column already exists — no-op
            
            conn.commit()
            print(f"[VERIFAI DB] Database initialized at: {path}")
            print(f"[VERIFAI DB] Schema: 14 tables, 35+ indexes created")
            _initialized = True
        except Exception as e:
            print(f"[VERIFAI DB] ERROR initializing database: {e}")
            raise
        finally:
            conn.close()
