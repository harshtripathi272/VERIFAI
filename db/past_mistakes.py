"""
Past Mistakes Memory Database

Persistent storage for validated diagnostic errors with hybrid indexing:
- Structured indexes for fast metadata filtering (disease_type, error_type, severity, KLE buckets)
- Vector similarity index (HNSW) for semantic case search

Uses DuckDB VSS extension for efficient approximate nearest neighbor search.
"""

import os
import json
import uuid
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager

import duckdb
import numpy as np

from app.config import settings

# Thread-local storage for connections
_local = threading.local()

# Database path
DB_PATH = getattr(settings, 'PAST_MISTAKES_DB_PATH', 
                  os.path.join(os.path.dirname(os.path.dirname(__file__)), "verifai_past_mistakes.duckdb"))

# Lock for initialization
_init_lock = threading.Lock()
_initialized = False


# SCHEMA DEFINITION

SCHEMA_SQL = """
-- Past Mistakes table with all required fields
CREATE TABLE IF NOT EXISTS past_mistakes (
    mistake_id VARCHAR PRIMARY KEY,
    
    -- Session tracking
    session_id VARCHAR NOT NULL,
    image_path VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Diagnosis information
    original_diagnosis VARCHAR NOT NULL,
    corrected_diagnosis VARCHAR NOT NULL,
    disease_type VARCHAR NOT NULL,  -- Primary pathology category (pneumonia, effusion, etc.)
    
    -- Error classification
    error_type VARCHAR NOT NULL,  -- overconfidence, misdiagnosis, missed_differential, calibration_error
    severity_level INTEGER NOT NULL CHECK (severity_level BETWEEN 1 AND 5),
    
    -- Uncertainty and safety metrics
    kle_uncertainty REAL,
    safety_score REAL,
    
    -- Clinical data (JSON strings)
    chexbert_labels VARCHAR,  -- JSON dict of CheXpert labels
    clinical_summary TEXT,
    debate_summary TEXT,
    
    -- Semantic embedding for similarity search (384-dim vector from sentence-transformers)
    case_embedding FLOAT[384] NOT NULL
);
"""

INDEXES_SQL = """
-- Structured indexes for fast metadata filtering
CREATE INDEX IF NOT EXISTS idx_pm_disease_type ON past_mistakes(disease_type);
CREATE INDEX IF NOT EXISTS idx_pm_error_type ON past_mistakes(error_type);
CREATE INDEX IF NOT EXISTS idx_pm_original_diagnosis ON past_mistakes(original_diagnosis);
CREATE INDEX IF NOT EXISTS idx_pm_corrected_diagnosis ON past_mistakes(corrected_diagnosis);
CREATE INDEX IF NOT EXISTS idx_pm_severity ON past_mistakes(severity_level);
CREATE INDEX IF NOT EXISTS idx_pm_kle_bucket ON past_mistakes(CAST(kle_uncertainty * 10 AS INTEGER));
CREATE INDEX IF NOT EXISTS idx_pm_created_at ON past_mistakes(created_at);
CREATE INDEX IF NOT EXISTS idx_pm_session ON past_mistakes(session_id);
"""

# HNSW index creation (separate, requires VSS extension)
HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_pm_case_embedding 
    ON past_mistakes 
    USING HNSW (case_embedding)
    WITH (metric = 'cosine');
"""



# CONNECTION MANAGEMENT

def get_connection(db_path: str = None) -> duckdb.DuckDBPyConnection:
    """
    Get a thread-local DuckDB connection with VSS extension loaded.
    
    Each thread gets its own connection to avoid threading issues.
    Connections are reused within the same thread.
    """
    path = db_path or DB_PATH
    
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = duckdb.connect(path)
        
        # Load VSS extension for HNSW indexing
        try:
            _local.connection.execute("INSTALL vss")
            _local.connection.execute("LOAD vss")
        except Exception as e:
            print(f"[PAST_MISTAKES] Warning: Failed to load VSS extension: {e}")
            print("[PAST_MISTAKES] Vector similarity search will not be available")
    
    return _local.connection


@contextmanager
def get_db(db_path: str = None):
    conn = get_connection(db_path)
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except:
            pass
        raise



def init_past_mistakes_db(db_path: str = None):
    """
    Initialize the past mistakes database schema and indexes.
    
    Safe to call multiple times — uses CREATE IF NOT EXISTS.
    Automatically called on first database access.
    """
    global _initialized
    
    with _init_lock:
        if _initialized:
            return
        
        path = db_path or DB_PATH
        
        try:
            with get_db(path) as conn:
                # Create table
                conn.execute(SCHEMA_SQL)
                
                # Create structured indexes
                conn.execute(INDEXES_SQL)
                
                # Try to create HNSW index (requires VSS extension)
                try:
                    conn.execute(HNSW_INDEX_SQL)
                    print(f"[PAST_MISTAKES] Database initialized at: {path}")
                    print("[PAST_MISTAKES] Schema: past_mistakes table with 8 structured indexes + HNSW vector index")
                except Exception as e:
                    print(f"[PAST_MISTAKES] Warning: HNSW index creation failed: {e}")
                    print("[PAST_MISTAKES] Database initialized with structured indexes only")
                
                _initialized = True
        except Exception as e:
            print(f"[PAST_MISTAKES] ERROR initializing database: {e}")
            raise


# =============================================================================
# CRUD OPERATIONS
# =============================================================================

def insert_validated_mistake(
    session_id: str,
    image_path: str,
    original_diagnosis: str,
    corrected_diagnosis: str,
    disease_type: str,
    error_type: str,
    severity_level: int,
    case_embedding: np.ndarray,
    kle_uncertainty: Optional[float] = None,
    safety_score: Optional[float] = None,
    chexbert_labels: Optional[Dict[str, str]] = None,
    clinical_summary: Optional[str] = None,
    debate_summary: Optional[str] = None,
    db_path: str = None
) -> str:
    """
    Insert a validated diagnostic mistake into the database.
    
    Args:
        session_id: Original session ID where mistake occurred
        image_path: Path to the X-ray image
        original_diagnosis: Incorrect diagnosis that was made
        corrected_diagnosis: Validated correct diagnosis
        disease_type: Primary pathology category (e.g., 'pneumonia', 'effusion')
        error_type: Type of error (overconfidence, misdiagnosis, missed_differential, calibration_error)
        severity_level: Error severity from 1 (minor) to 5 (critical)
        case_embedding: 384-dim numpy array semantic embedding of the case
        kle_uncertainty: KLE uncertainty score at time of mistake (optional)
        safety_score: Safety score at time of mistake (optional)
        chexbert_labels: Dict of CheXpert labels (optional)
        clinical_summary: Clinical context summary (optional)
        debate_summary: Debate/reasoning summary (optional)
        db_path: Database path (optional, uses config default)
    
    Returns:
        Mistake ID (UUID)
    
    Raises:
        ValueError: If required fields are invalid
    """
    # Validate inputs
    if not original_diagnosis or not corrected_diagnosis:
        raise ValueError("original_diagnosis and corrected_diagnosis are required")
    
    if not disease_type:
        raise ValueError("disease_type is required")
    
    if error_type not in ['overconfidence', 'misdiagnosis', 'missed_differential', 'calibration_error']:
        raise ValueError(f"Invalid error_type: {error_type}")
    
    if not (1 <= severity_level <= 5):
        raise ValueError(f"severity_level must be between 1 and 5, got {severity_level}")
    
    if case_embedding.shape != (384,):
        raise ValueError(f"case_embedding must be 384-dim array, got shape {case_embedding.shape}")
    
    # Initialize DB if needed
    init_past_mistakes_db(db_path)
    
    # Generate mistake ID
    mistake_id = str(uuid.uuid4())
    
    # Convert embedding to list for DuckDB ARRAY type
    embedding_list = case_embedding.tolist()
    
    # Serialize JSON fields
    chexbert_json = json.dumps(chexbert_labels) if chexbert_labels else None
    
    with get_db(db_path) as conn:
        conn.execute("""
            INSERT INTO past_mistakes (
                mistake_id, session_id, image_path, created_at,
                original_diagnosis, corrected_diagnosis, disease_type,
                error_type, severity_level,
                kle_uncertainty, safety_score,
                chexbert_labels, clinical_summary, debate_summary,
                case_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            mistake_id, session_id, image_path, datetime.now(),
            original_diagnosis, corrected_diagnosis, disease_type,
            error_type, severity_level,
            kle_uncertainty, safety_score,
            chexbert_json, clinical_summary, debate_summary,
            embedding_list
        ])
    
    return mistake_id


def retrieve_similar_mistakes(
    disease_type: str,
    embedding: np.ndarray,
    kle_uncertainty_range: Optional[Tuple[float, float]] = None,
    error_types: Optional[List[str]] = None,
    severity_min: int = 1,
    top_k: int = 5,
    similarity_threshold: float = 0.75,
    db_path: str = None
) -> List[Dict[str, Any]]:
    """
    Retrieve historically similar mistakes using hybrid search.
    
    Pipeline:
    1. Structured filtering by disease_type, KLE range, error types, severity
    2. Vector similarity search using HNSW index (approximate nearest neighbors)
    3. Filter by similarity threshold
    4. Return top-K most similar cases
    
    Args:
        disease_type: Exact match on disease category (e.g., 'pneumonia')
        embedding: 384-dim query embedding for semantic similarity
        kle_uncertainty_range: Optional (min, max) KLE range filter
        error_types: Optional list of error types to filter by
        severity_min: Minimum severity level (1-5)
        top_k: Number of similar cases to retrieve
        similarity_threshold: Minimum cosine similarity (0-1) to include
        db_path: Database path (optional)
    
    Returns:
        List of dictionaries containing mistake records with similarity scores
    """
    init_past_mistakes_db(db_path)
    
    if embedding.shape != (384,):
        raise ValueError(f"embedding must be 384-dim array, got shape {embedding.shape}")
    
    embedding_list = embedding.tolist()
    
    # Build WHERE clause for structured filtering
    where_clauses = ["disease_type = ?"]
    params = [disease_type]
    
    if kle_uncertainty_range:
        where_clauses.append("kle_uncertainty BETWEEN ? AND ?")
        params.extend(kle_uncertainty_range)
    
    if error_types:
        placeholders = ','.join(['?'] * len(error_types))
        where_clauses.append(f"error_type IN ({placeholders})")
        params.extend(error_types)
    
    where_clauses.append("severity_level >= ?")
    params.append(severity_min)
    
    where_clause = " AND ".join(where_clauses)
    
    # Query with vector similarity using array_cosine_distance
    # Note: DuckDB VSS uses distance metrics, so lower is more similar
    # For cosine similarity, we compute 1 - cosine_distance
    query = f"""
        SELECT 
            mistake_id,
            session_id,
            image_path,
            original_diagnosis,
            corrected_diagnosis,
            disease_type,
            error_type,
            severity_level,
            kle_uncertainty,
            safety_score,
            chexbert_labels,
            clinical_summary,
            debate_summary,
            created_at,
            (1.0 - array_cosine_distance(case_embedding, ?::FLOAT[384])) AS similarity
        FROM past_mistakes
        WHERE {where_clause}
        ORDER BY array_cosine_distance(case_embedding, ?::FLOAT[384]) ASC
        LIMIT ?
    """
    
    # Add embedding twice (for SELECT and ORDER BY) plus other params
    all_params = [embedding_list] + params + [embedding_list, top_k * 2]  # Fetch 2x for filtering
    
    with get_db(db_path) as conn:
        results = conn.execute(query, all_params).fetchall()
    
    # Convert to list of dicts and filter by similarity threshold
    similar_cases = []
    for row in results:
        # Convert row to dict
        case = {
            'mistake_id': row[0],
            'session_id': row[1],
            'image_path': row[2],
            'original_diagnosis': row[3],
            'corrected_diagnosis': row[4],
            'disease_type': row[5],
            'error_type': row[6],
            'severity_level': row[7],
            'kle_uncertainty': row[8],
            'safety_score': row[9],
            'chexbert_labels': json.loads(row[10]) if row[10] else {},
            'clinical_summary': row[11],
            'debate_summary': row[12],
            'created_at': row[13],
            'similarity': row[14]
        }
        
        # Filter by similarity threshold
        if case['similarity'] >= similarity_threshold:
            similar_cases.append(case)
        
        # Stop if we have enough
        if len(similar_cases) >= top_k:
            break
    
    return similar_cases


def get_mistake_by_id(mistake_id: str, db_path: str = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single mistake by its ID.
    
    Args:
        mistake_id: UUID of the mistake
        db_path: Database path (optional)
    
    Returns:
        Mistake record as dict, or None if not found
    """
    init_past_mistakes_db(db_path)
    
    with get_db(db_path) as conn:
        result = conn.execute("""
            SELECT 
                mistake_id, session_id, image_path, created_at,
                original_diagnosis, corrected_diagnosis, disease_type,
                error_type, severity_level,
                kle_uncertainty, safety_score,
                chexbert_labels, clinical_summary, debate_summary
            FROM past_mistakes
            WHERE mistake_id = ?
        """, [mistake_id]).fetchone()
    
    if not result:
        return None
    
    return {
        'mistake_id': result[0],
        'session_id': result[1],
        'image_path': result[2],
        'created_at': result[3],
        'original_diagnosis': result[4],
        'corrected_diagnosis': result[5],
        'disease_type': result[6],
        'error_type': result[7],
        'severity_level': result[8],
        'kle_uncertainty': result[9],
        'safety_score': result[10],
        'chexbert_labels': json.loads(result[11]) if result[11] else {},
        'clinical_summary': result[12],
        'debate_summary': result[13]
    }


def delete_mistake(mistake_id: str, db_path: str = None) -> bool:
    """
    Delete a mistake from the database.
    
    Args:
        mistake_id: UUID of the mistake to delete
        db_path: Database path (optional)
    
    Returns:
        True if deleted, False if not found
    """
    init_past_mistakes_db(db_path)
    
    with get_db(db_path) as conn:
        cursor = conn.execute("DELETE FROM past_mistakes WHERE mistake_id = ?", [mistake_id])
        return cursor.rowcount > 0


def get_statistics(db_path: str = None) -> Dict[str, Any]:
    """
    Get aggregate statistics about past mistakes.
    
    Returns:
        Dict with statistics by disease type, error type, severity, etc.
    """
    init_past_mistakes_db(db_path)
    
    with get_db(db_path) as conn:
        # Total count
        total = conn.execute("SELECT COUNT(*) FROM past_mistakes").fetchone()[0]
        
        # By disease type
        by_disease = conn.execute("""
            SELECT disease_type, COUNT(*) as cnt, AVG(severity_level) as avg_severity
            FROM past_mistakes
            GROUP BY disease_type
            ORDER BY cnt DESC
        """).fetchall()
        
        # By error type
        by_error = conn.execute("""
            SELECT error_type, COUNT(*) as cnt, AVG(severity_level) as avg_severity
            FROM past_mistakes
            GROUP BY error_type
            ORDER BY cnt DESC
        """).fetchall()
        
        # By severity
        by_severity = conn.execute("""
            SELECT severity_level, COUNT(*) as cnt
            FROM past_mistakes
            GROUP BY severity_level
            ORDER BY severity_level
        """).fetchall()
    
    return {
        'total_mistakes': total,
        'by_disease_type': [{'disease_type': r[0], 'count': r[1], 'avg_severity': r[2]} for r in by_disease],
        'by_error_type': [{'error_type': r[0], 'count': r[1], 'avg_severity': r[2]} for r in by_error],
        'by_severity': [{'severity': r[0], 'count': r[1]} for r in by_severity]
    }
