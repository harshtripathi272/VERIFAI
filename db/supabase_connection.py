"""
VERIFAI Supabase Connection Manager

Cloud-based PostgreSQL database via Supabase for structured logging.
Replaces local SQLite with scalable cloud storage.

Features:
- Connection pooling for better performance
- JSONB support for flexible data storage
- Timestamp with timezone for global consistency
- Row-level security ready
"""

import os
import json
from contextlib import contextmanager
from typing import Optional, Any, Dict, List
from datetime import datetime
import threading

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("[WARNING] supabase-py not installed. Run: pip install supabase")

# Thread-local storage for connections
_local = threading.local()

# Connection settings
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_KEY")

# Initialize flag
_initialized = False
_init_lock = threading.Lock()


def get_client() -> Optional[Client]:
    """
    Get a thread-local Supabase client using the anon/public key.

    Used for general structured logging (workflow_sessions, agent_invocations, …).
    Each thread gets its own client instance to avoid threading issues.
    """
    if not SUPABASE_AVAILABLE:
        raise ImportError("supabase-py is not installed. Run: pip install supabase")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in environment variables. "
            "Check your .env file."
        )

    # Create a fresh client instead of caching to avoid httpx ConnectError
    # (Errno 104 Connection reset by peer) when connections go idle during long-running nodes
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_service_client() -> Optional[Client]:
    """
    Get a thread-local Supabase client using the service-role key.

    Required for past-mistakes memory operations — the service-role key bypasses
    Row Level Security on the ``past_mistakes`` table so the repository can
    read and upsert records without user context.

    Raises RuntimeError if SUPABASE_URL or SUPABASE_SERVICE_KEY are not set.
    """
    if not SUPABASE_AVAILABLE:
        raise ImportError("supabase-py is not installed. Run: pip install supabase")

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "[SUPABASE] SUPABASE_URL and SUPABASE_SERVICE_KEY must be set. "
            "These are required for the past-mistakes memory backend. "
            "Check your .env file."
        )

    # Create a fresh client instead of caching to avoid httpx ConnectError
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


@contextmanager
def get_db():
    """
    Context manager for database operations.
    
    Unlike SQLite, Supabase handles commits automatically via REST API.
    This context manager provides a consistent interface with the old SQLite code.
    
    Usage:
        with get_db() as db:
            result = db.table('workflow_sessions').insert({...}).execute()
    """
    client = get_client()
    try:
        yield client
        # No explicit commit needed - Supabase auto-commits
    except Exception as e:
        print(f"[Supabase Error] {e}")
        raise


def init_db():
    """
    Initialize database connection and verify schema.
    
    Note: Schema must be created manually in Supabase dashboard using supabase_schema.sql
    This function only verifies connectivity.
    """
    global _initialized
    
    with _init_lock:
        if _initialized:
            return
        
        try:
            client = get_client()
            
            # Test connection by querying workflow_sessions table
            result = client.table('workflow_sessions').select('session_id').limit(1).execute()
            
            print(f"[VERIFAI DB] Connected to Supabase at: {SUPABASE_URL}")
            print(f"[VERIFAI DB] Schema verification: SUCCESS")
            _initialized = True
            
        except Exception as e:
            print(f"[VERIFAI DB] ERROR connecting to Supabase: {e}")
            print(f"[VERIFAI DB] Please ensure:")
            print(f"  1. SUPABASE_URL and SUPABASE_KEY are set in .env")
            print(f"  2. Database schema has been created using db/supabase_schema.sql")
            print(f"  3. Tables have proper permissions")
            # We don't raise here to keep the logger completely non-blocking
            _initialized = True  # Prevent repeated crash attempts on startup

 

def serialize_to_jsonb(obj: Any) -> Any:
    """
    Convert Python object to PostgreSQL JSONB-compatible format.
    
    Supabase automatically handles JSONB, so we just need to ensure
    the data is JSON-serializable.
    """
    if obj is None:
        return None
    
    # Handle Pydantic models
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    
    # Handle lists
    if isinstance(obj, list):
        return [serialize_to_jsonb(item) for item in obj]
    
    # Handle dicts
    if isinstance(obj, dict):
        return {k: serialize_to_jsonb(v) for k, v in obj.items()}
    
    # Handle datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Return as-is for primitives
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Fallback: convert to string
    return str(obj)


def prepare_insert_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare data for Supabase insert/update operations.
    
    - Converts Python objects to JSONB-compatible format
    - Handles datetime serialization
    - Removes None values (optional)
    """
    prepared = {}
    
    for key, value in data.items():
        if value is None:
            # Option 1: Keep None values (will be stored as NULL)
            prepared[key] = None
            
            # Option 2: Skip None values (uncomment if preferred)
            # continue
        elif isinstance(value, (list, dict)):
            # Convert to JSONB-compatible format
            prepared[key] = serialize_to_jsonb(value)
        elif isinstance(value, datetime):
            # Convert datetime to ISO format with timezone
            prepared[key] = value.isoformat()
        else:
            prepared[key] = value
    
    return prepared


def query_builder_helper(table_name: str, filters: Dict[str, Any] = None) -> Any:
    """
    Helper for building Supabase queries with common patterns.
    
    Example:
        query = query_builder_helper('workflow_sessions', {'patient_id': 'P001'})
        results = query.select('*').execute()
    """
    client = get_client()
    query = client.table(table_name)
    
    if filters:
        for key, value in filters.items():
            query = query.eq(key, value)
    
    return query


# =============================================================================
# MIGRATION UTILITIES (Optional - for moving data from SQLite to Supabase)
# =============================================================================

def migrate_from_sqlite(sqlite_db_path: str):
    """
    Migrate data from local SQLite database to Supabase.
    
    WARNING: This is a one-time migration tool. Use with caution.
    
    Args:
        sqlite_db_path: Path to the SQLite .db file
    """
    import sqlite3
    
    print(f"[MIGRATION] Starting migration from {sqlite_db_path}")
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_db_path)
    sqlite_conn.row_factory = sqlite3.Row
    
    # Get Supabase client
    supabase = get_client()
    
    # Define tables to migrate (in order due to foreign keys)
    tables = [
        'workflow_sessions',
        'agent_invocations',
        'radiologist_logs',
        'critic_logs',
        'historian_logs',
        'historian_facts',
        'literature_logs',
        'literature_citations',
        'debate_logs',
        'debate_rounds',
        'debate_arguments',
        'chief_logs',
        'trace_log'
    ]
    
    try:
        for table in tables:
            print(f"[MIGRATION] Migrating table: {table}")
            
            # Fetch all rows from SQLite
            cursor = sqlite_conn.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            
            if not rows:
                print(f"[MIGRATION] {table}: No data to migrate")
                continue
            
            # Convert to dict and insert into Supabase
            for row in rows:
                row_dict = dict(row)
                
                # Convert INTEGER booleans to actual booleans for PostgreSQL
                for key, value in row_dict.items():
                    if isinstance(value, int) and key in [
                        'was_deferred', 'is_overconfident', 'final_consensus',
                        'escalate_to_chief', 'is_feedback_iteration', 'has_feedback',
                        'reprocessed'
                    ]:
                        row_dict[key] = bool(value)
                
                # Prepare data
                prepared_data = prepare_insert_data(row_dict)
                
                # Insert into Supabase
                supabase.table(table).insert(prepared_data).execute()
            
            print(f"[MIGRATION] {table}: Migrated {len(rows)} rows")
        
        print("[MIGRATION] Completed successfully!")
        
    except Exception as e:
        print(f"[MIGRATION] ERROR: {e}")
        raise
    finally:
        sqlite_conn.close()


# =============================================================================
# BACKWARDS COMPATIBILITY LAYER
# =============================================================================

def execute_query(query: str, params: tuple = None) -> List[Dict]:
    """
    Execute a raw SQL query (for backwards compatibility).
    
    Note: Supabase prefers using its query builder, but this allows
    running raw SQL for complex queries.
    """
    # Supabase doesn't support raw SQL via the Python client
    # You'd need to use the PostgREST API or pg library directly
    raise NotImplementedError(
        "Raw SQL queries not supported via Supabase client. "
        "Use query builder methods instead."
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================

def health_check() -> bool:
    """
    Verify database connection and basic operations.
    
    Returns True if healthy, False otherwise.
    """
    try:
        client = get_client()
        
        # Try a simple query
        result = client.table('workflow_sessions').select('session_id').limit(1).execute()
        
        return True
    except Exception as e:
        print(f"[HEALTH CHECK] Failed: {e}")
        return False
