-- VERIFAI Supabase Schema (PostgreSQL)
-- Migration from SQLite to cloud-based Supabase
-- 
-- To apply this schema:
-- 1. Create a Supabase project at https://supabase.com
-- 2. Go to SQL Editor and run this script
-- 3. Update .env with your Supabase credentials

-- 1. WORKFLOW SESSIONS — one row per full pipeline invocation
CREATE TABLE IF NOT EXISTS workflow_sessions (
    session_id          TEXT PRIMARY KEY,
    image_path          TEXT NOT NULL,
    patient_id          TEXT,
    workflow_type       TEXT DEFAULT 'debate',
    status              TEXT DEFAULT 'running',
    started_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at        TIMESTAMP WITH TIME ZONE,
    final_diagnosis     TEXT,
    final_confidence    REAL,
    was_deferred        BOOLEAN DEFAULT FALSE,
    deferral_reason     TEXT,
    total_agents_invoked INTEGER DEFAULT 0,
    error_message       TEXT,
    
    -- NEW: Doctor feedback tracking
    has_feedback        BOOLEAN DEFAULT FALSE,
    feedback_status     TEXT,  -- 'approved', 'rejected', 'pending_review'
    feedback_count      INTEGER DEFAULT 0
);

-- 2. AGENT INVOCATIONS — one row per agent call within a session
CREATE TABLE IF NOT EXISTS agent_invocations (
    invocation_id       SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    agent_name          TEXT NOT NULL,
    started_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at        TIMESTAMP WITH TIME ZONE,
    duration_ms         INTEGER,
    status              TEXT DEFAULT 'running',
    error_message       TEXT,
    input_summary       JSONB,
    output_summary      JSONB,
    trace_entries       JSONB,
    
    -- NEW: Feedback loop tracking
    is_feedback_iteration BOOLEAN DEFAULT FALSE,
    parent_invocation_id INTEGER,  -- Links to original invocation before feedback
    
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE
);

-- ============================================================
-- 3. RADIOLOGIST LOGS — plain-text findings & impression + KLE uncertainty
-- ============================================================
CREATE TABLE IF NOT EXISTS radiologist_logs (
    log_id              SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    image_path          TEXT,
    findings_text       TEXT,
    impression_text     TEXT,
    kle_uncertainty     REAL,
    num_samples         INTEGER DEFAULT 1,
    
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id) ON DELETE CASCADE
);

-- 4. CRITIC LOGS — overconfidence detection via KLE consistency
CREATE TABLE IF NOT EXISTS critic_logs (
    log_id              SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_overconfident    BOOLEAN DEFAULT FALSE,
    safety_score        REAL,
    concern_flags       JSONB,
    recommended_hedging TEXT,
    kle_uncertainty_input REAL,
    similar_mistakes_count INTEGER DEFAULT 0,
    historical_risk_level TEXT,
    
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id) ON DELETE CASCADE
);

-- 5. HISTORIAN LOGS — FHIR-based clinical context
CREATE TABLE IF NOT EXISTS historian_logs (
    log_id              SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    patient_id          TEXT,
    confidence_adjustment REAL,
    clinical_summary    TEXT,
    num_supporting      INTEGER DEFAULT 0,
    num_contradicting   INTEGER DEFAULT 0,
    
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS historian_facts (
    fact_id             SERIAL PRIMARY KEY,
    historian_log_id    INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    fact_type           TEXT NOT NULL,
    description         TEXT NOT NULL,
    fhir_resource_id    TEXT,
    fhir_resource_type  TEXT,
    
    FOREIGN KEY (historian_log_id) REFERENCES historian_logs(log_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE
);

-- 6. LITERATURE LOGS — PubMed/PMC/SemanticScholar results
CREATE TABLE IF NOT EXISTS literature_logs (
    log_id              SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    query_used          TEXT,
    overall_evidence_strength TEXT,
    num_citations       INTEGER DEFAULT 0,
    raw_summary         TEXT,
    
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS literature_citations (
    citation_id         SERIAL PRIMARY KEY,
    literature_log_id   INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    pmid                TEXT,
    title               TEXT,
    authors             JSONB,
    journal             TEXT,
    year                INTEGER,
    relevance_summary   TEXT,
    evidence_strength   TEXT,
    source              TEXT,
    
    FOREIGN KEY (literature_log_id) REFERENCES literature_logs(log_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE
);

-- 7. DEBATE LOGS — full debate rounds and arguments
CREATE TABLE IF NOT EXISTS debate_logs (
    log_id              SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    num_rounds          INTEGER DEFAULT 0,
    final_consensus     BOOLEAN DEFAULT FALSE,
    consensus_diagnosis TEXT,
    consensus_confidence REAL,
    escalate_to_chief   BOOLEAN DEFAULT FALSE,
    escalation_reason   TEXT,
    debate_summary      TEXT,
    total_confidence_adj REAL DEFAULT 0.0,
    
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS debate_rounds (
    round_id            SERIAL PRIMARY KEY,
    debate_log_id       INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    round_number        INTEGER NOT NULL,
    round_consensus     TEXT,
    confidence_delta    REAL DEFAULT 0.0,
    
    FOREIGN KEY (debate_log_id) REFERENCES debate_logs(log_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS debate_arguments (
    argument_id         SERIAL PRIMARY KEY,
    round_id            INTEGER NOT NULL,
    debate_log_id       INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    agent               TEXT NOT NULL,
    position            TEXT NOT NULL,
    argument            TEXT NOT NULL,
    confidence_impact   REAL DEFAULT 0.0,
    evidence_refs       JSONB,
    
    FOREIGN KEY (round_id) REFERENCES debate_rounds(round_id) ON DELETE CASCADE,
    FOREIGN KEY (debate_log_id) REFERENCES debate_logs(log_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE
);

-- 8. CHIEF LOGS — final arbitration decisions
CREATE TABLE IF NOT EXISTS chief_logs (
    log_id              SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    invocation_id       INTEGER NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    diagnosis           TEXT,
    calibrated_confidence REAL,
    was_deferred        BOOLEAN DEFAULT FALSE,
    deferral_reason     TEXT,
    explanation         TEXT,
    recommended_next_steps JSONB,
    
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (invocation_id) REFERENCES agent_invocations(invocation_id) ON DELETE CASCADE
);

-- 9. TRACE LOG — flat audit trail (mirrors state.trace)
CREATE TABLE IF NOT EXISTS trace_log (
    trace_id            SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    agent_name          TEXT,
    entry               TEXT NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE
);

-- 10. DOCTOR FEEDBACK — NEW: Captures doctor's input when rejecting diagnosis

CREATE TABLE IF NOT EXISTS doctor_feedback (
    feedback_id         SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    original_diagnosis  TEXT,
    original_confidence REAL,
    
    -- Feedback details
    feedback_type       TEXT NOT NULL,  -- 'rejection', 'correction', 'approval'
    doctor_notes        TEXT NOT NULL,  -- Doctor's explanation of what's wrong
    correct_diagnosis   TEXT,           -- What doctor believes is correct
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    doctor_id           TEXT,           -- Optional: track which doctor gave feedback
    
    -- Reprocessing tracking
    reprocessed         BOOLEAN DEFAULT FALSE,
    reprocess_session_id TEXT,          -- Links to new session created from feedback
    reprocess_result    TEXT,           -- Final outcome after reprocessing
    reprocess_confidence REAL,
    
    -- Metadata
    rejection_reason    TEXT[],         -- List of reasons (e.g., ['wrong_pathology', 'missed_finding'])
    context_snapshot    JSONB,          -- Store full workflow state at rejection point
    
    FOREIGN KEY (session_id) REFERENCES workflow_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (reprocess_session_id) REFERENCES workflow_sessions(session_id) ON DELETE SET NULL
);


-- INDEXES FOR PERFORMANCE


-- Session indexes
CREATE INDEX IF NOT EXISTS idx_sessions_patient ON workflow_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON workflow_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON workflow_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_diagnosis ON workflow_sessions(final_diagnosis);
CREATE INDEX IF NOT EXISTS idx_sessions_feedback ON workflow_sessions(has_feedback);

-- Agent invocation indexes
CREATE INDEX IF NOT EXISTS idx_invocations_session ON agent_invocations(session_id);
CREATE INDEX IF NOT EXISTS idx_invocations_agent ON agent_invocations(agent_name);
CREATE INDEX IF NOT EXISTS idx_invocations_status ON agent_invocations(status);
CREATE INDEX IF NOT EXISTS idx_invocations_feedback ON agent_invocations(is_feedback_iteration);

-- Radiologist indexes
CREATE INDEX IF NOT EXISTS idx_rad_logs_session ON radiologist_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_rad_logs_kle ON radiologist_logs(kle_uncertainty);

-- Critic indexes
CREATE INDEX IF NOT EXISTS idx_critic_session ON critic_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_critic_overconf ON critic_logs(is_overconfident);
CREATE INDEX IF NOT EXISTS idx_critic_safety ON critic_logs(safety_score);

-- Historian indexes
CREATE INDEX IF NOT EXISTS idx_historian_session ON historian_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_historian_patient ON historian_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_historian_facts_session ON historian_facts(session_id);

-- Literature indexes
CREATE INDEX IF NOT EXISTS idx_lit_session ON literature_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_lit_citations_session ON literature_citations(session_id);
CREATE INDEX IF NOT EXISTS idx_lit_citations_pmid ON literature_citations(pmid);

-- Debate indexes
CREATE INDEX IF NOT EXISTS idx_debate_session ON debate_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_debate_rounds_session ON debate_rounds(session_id);
CREATE INDEX IF NOT EXISTS idx_debate_args_session ON debate_arguments(session_id);

-- Trace indexes
CREATE INDEX IF NOT EXISTS idx_trace_session ON trace_log(session_id);
CREATE INDEX IF NOT EXISTS idx_trace_agent ON trace_log(agent_name);

-- NEW: Doctor feedback indexes
CREATE INDEX IF NOT EXISTS idx_feedback_session ON doctor_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON doctor_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_reprocessed ON doctor_feedback(reprocessed);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON doctor_feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_doctor ON doctor_feedback(doctor_id);

-- ============================================================
-- ROW LEVEL SECURITY (OPTIONAL - Enable if needed)
-- ============================================================

-- Enable RLS on all tables (recommended for multi-tenant setups)
-- ALTER TABLE workflow_sessions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE agent_invocations ENABLE ROW LEVEL SECURITY;
-- ... (repeat for all tables)

-- Example policy: Allow authenticated users to see their own data
-- CREATE POLICY "Users can view their own sessions" 
-- ON workflow_sessions FOR SELECT 
-- USING (auth.uid()::text = doctor_id);

-- ============================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================

COMMENT ON TABLE doctor_feedback IS 'Stores doctor feedback when diagnoses are rejected, enabling feedback loop reprocessing';
COMMENT ON COLUMN doctor_feedback.context_snapshot IS 'Full workflow state (radiologist, critic, historian, literature outputs) at rejection point';
COMMENT ON COLUMN doctor_feedback.reprocess_session_id IS 'Links to the new session created when reprocessing with doctor feedback';
COMMENT ON COLUMN agent_invocations.is_feedback_iteration IS 'TRUE if this agent run was triggered by doctor feedback loop';

-- ============================================================
-- HELPER FUNCTIONS (OPTIONAL)
-- ============================================================

-- Function to retrieve full context for a feedback session
CREATE OR REPLACE FUNCTION get_feedback_context(p_session_id TEXT)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'session', (SELECT row_to_json(workflow_sessions.*) FROM workflow_sessions WHERE session_id = p_session_id),
        'radiologist', (SELECT row_to_json(radiologist_logs.*) FROM radiologist_logs WHERE session_id = p_session_id ORDER BY created_at DESC LIMIT 1),
        'critic', (SELECT row_to_json(critic_logs.*) FROM critic_logs WHERE session_id = p_session_id ORDER BY created_at DESC LIMIT 1),
        'historian', (SELECT row_to_json(historian_logs.*) FROM historian_logs WHERE session_id = p_session_id ORDER BY created_at DESC LIMIT 1),
        'literature', (SELECT row_to_json(literature_logs.*) FROM literature_logs WHERE session_id = p_session_id ORDER BY created_at DESC LIMIT 1),
        'debate', (SELECT row_to_json(debate_logs.*) FROM debate_logs WHERE session_id = p_session_id ORDER BY created_at DESC LIMIT 1)
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_feedback_context IS 'Retrieves complete workflow context for a session to support feedback loop reprocessing';


-- ============================================================
-- PAST MISTAKES — pgvector HNSW (run before enabling USE_CLOUD_VECTOR_DB)
-- ============================================================

-- 1. Enable pgvector extension (requires Supabase pgvector addon or PG>=15 self-hosted)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Past-mistakes table with a native pgvector column
--    (Case embeddings are 384-dim from sentence-transformers/all-MiniLM-L6-v2)
CREATE TABLE IF NOT EXISTS past_mistakes (
    mistake_id           TEXT PRIMARY KEY,
    session_id           TEXT NOT NULL,
    image_path           TEXT,
    created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    original_diagnosis   TEXT NOT NULL,
    corrected_diagnosis  TEXT NOT NULL,
    disease_type         TEXT NOT NULL,
    error_type           TEXT NOT NULL,
    severity_level       INTEGER NOT NULL CHECK (severity_level BETWEEN 1 AND 5),
    kle_uncertainty      REAL,
    safety_score         REAL,
    chexbert_labels      TEXT,   -- JSON-encoded dict
    clinical_summary     TEXT,
    debate_summary       TEXT,
    case_embedding       VECTOR(384) NOT NULL
);

-- 3. Structured indexes for fast metadata pre-filtering
CREATE INDEX IF NOT EXISTS idx_pm_disease     ON past_mistakes (disease_type);
CREATE INDEX IF NOT EXISTS idx_pm_error_type  ON past_mistakes (error_type);
CREATE INDEX IF NOT EXISTS idx_pm_severity    ON past_mistakes (severity_level);
CREATE INDEX IF NOT EXISTS idx_pm_kle         ON past_mistakes (kle_uncertainty);
CREATE INDEX IF NOT EXISTS idx_pm_created_at  ON past_mistakes (created_at);
CREATE INDEX IF NOT EXISTS idx_pm_session     ON past_mistakes (session_id);

-- 4. HNSW index for approximate nearest-neighbour cosine similarity
--    Must be created BEFORE the match_mistakes function is called in production.
CREATE INDEX IF NOT EXISTS idx_pm_case_embedding_hnsw
    ON past_mistakes
    USING hnsw (case_embedding vector_cosine_ops);

-- 5. RPC function called by SupabasePastMistakesRepository
--    Performs cosine similarity search ordered by ascending distance (HNSW optimal).
--    Returns similarity = 1 - cosine_distance so callers use standard [0, 1] scoring.
CREATE OR REPLACE FUNCTION match_mistakes(
    query_embedding  VECTOR(384),
    disease_type     TEXT,
    kle_min          FLOAT,
    kle_max          FLOAT,
    severity_min     INT,
    top_k            INT
)
RETURNS TABLE (
    mistake_id           TEXT,
    session_id           TEXT,
    image_path           TEXT,
    original_diagnosis   TEXT,
    corrected_diagnosis  TEXT,
    disease_type         TEXT,
    error_type           TEXT,
    severity_level       INT,
    kle_uncertainty      FLOAT,
    safety_score         FLOAT,
    chexbert_labels      TEXT,
    clinical_summary     TEXT,
    debate_summary       TEXT,
    created_at           TIMESTAMP WITH TIME ZONE,
    similarity           FLOAT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        pm.mistake_id,
        pm.session_id,
        pm.image_path,
        pm.original_diagnosis,
        pm.corrected_diagnosis,
        pm.disease_type,
        pm.error_type,
        pm.severity_level,
        pm.kle_uncertainty,
        pm.safety_score,
        pm.chexbert_labels,
        pm.clinical_summary,
        pm.debate_summary,
        pm.created_at,
        -- similarity: 1 - cosine_distance (higher = more similar)
        1.0 - (pm.case_embedding <=> query_embedding) AS similarity
    FROM past_mistakes pm
    WHERE pm.disease_type     = match_mistakes.disease_type
      AND pm.kle_uncertainty  BETWEEN kle_min AND kle_max
      AND pm.severity_level  >= severity_min
    ORDER BY pm.case_embedding <=> query_embedding  -- ascending distance exercised by HNSW
    LIMIT top_k * 2;   -- over-fetch; Python layer applies similarity threshold & top_k cap
$$;

COMMENT ON FUNCTION match_mistakes IS
    'pgvector HNSW cosine similarity search over past_mistakes. '
    'Used by SupabasePastMistakesRepository when USE_CLOUD_VECTOR_DB=True.';
