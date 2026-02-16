"""
VERIFAI Database Adapter

Unified interface that automatically switches between SQLite and Supabase
based on DATABASE_MODE configuration.

This allows seamless transition from local development to cloud production.

Usage (automatically uses correct backend):
    from db.adapter import get_logger
    
    logger = get_logger(session_id="abc-123")
    logger.log_radiologist(state, result)
"""

import os
from typing import Optional
from app.config import settings


def get_logger(session_id: str = None, image_path: str = "", patient_id: str = None, workflow_type: str = "debate"):
    """
    Get the appropriate logger based on DATABASE_MODE setting.
    
    Returns either SQLite logger or Supabase logger transparently.
    Both have identical APIs, so code using them doesn't need to change.
    
    Args:
        session_id: Unique session ID (auto-generated if not provided)
        image_path: Path to input X-ray image
        patient_id: Optional FHIR patient ID
        workflow_type: 'debate' or 'legacy'
    
    Returns:
        AgentLogger instance (SQLite or Supabase based on config)
    """
    database_mode = getattr(settings, 'DATABASE_MODE', 'sqlite').lower()
    
    if database_mode == 'supabase':
        # Use cloud-based Supabase logger
        try:
            from db.supabase_logger import AgentLogger
            print(f"[DB Adapter] Using Supabase (cloud) for session: {session_id}")
            return AgentLogger(session_id, image_path, patient_id, workflow_type)
        except ImportError as e:
            print(f"[DB Adapter] ERROR: Supabase not available: {e}")
            print(f"[DB Adapter] Falling back to SQLite")
            from db.logger import AgentLogger
            return AgentLogger(session_id, image_path, patient_id, workflow_type)
    else:
        # Use local SQLite logger (default for development)
        from db.logger import AgentLogger
        print(f"[DB Adapter] Using SQLite (local) for session: {session_id}")
        return AgentLogger(session_id, image_path, patient_id, workflow_type)


def check_database_health() -> dict:
    """
    Check database connection health for current mode.
    
    Returns:
        Dictionary with health status and details
    """
    database_mode = getattr(settings, 'DATABASE_MODE', 'sqlite').lower()
    
    result = {
        'mode': database_mode,
        'healthy': False,
        'details': {}
    }
    
    try:
        if database_mode == 'supabase':
            from db.supabase_connection import health_check, SUPABASE_URL
            result['healthy'] = health_check()
            result['details'] = {
                'url': SUPABASE_URL,
                'connection': 'OK' if result['healthy'] else 'FAILED'
            }
        else:
            from db.connection import DB_PATH, init_db
            import os
            init_db()
            result['healthy'] = os.path.exists(DB_PATH)
            result['details'] = {
                'path': DB_PATH,
                'exists': result['healthy']
            }
    except Exception as e:
        result['healthy'] = False
        result['details']['error'] = str(e)
    
    return result


def migrate_to_cloud(sqlite_db_path: str = None):
    """
    Migrate data from local SQLite to Supabase cloud.
    
    This is a one-time operation when moving from development to production.
    
    Args:
        sqlite_db_path: Path to SQLite database (auto-detected if not provided)
    """
    from db.supabase_connection import migrate_from_sqlite
    from db.connection import DB_PATH
    
    db_path = sqlite_db_path or DB_PATH
    
    print(f"[Migration] Starting migration from SQLite to Supabase")
    print(f"[Migration] Source: {db_path}")
    print(f"[Migration] Target: {getattr(settings, 'SUPABASE_URL', 'Not configured')}")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    
    if not getattr(settings, 'SUPABASE_URL') or not getattr(settings, 'SUPABASE_KEY'):
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    
    confirm = input("This will copy all data from SQLite to Supabase. Continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("[Migration] Cancelled")
        return
    
    migrate_from_sqlite(db_path)
    print("[Migration] Complete! You can now set DATABASE_MODE=supabase in .env")
