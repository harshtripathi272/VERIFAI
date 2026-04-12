"""
VERIFAI Database Adapter

Provides the unified Supabase cloud database interface for the workflow.
The SQLite fallback has been completely removed for production mode.

Usage:
    from db.adapter import get_logger
    
    logger = get_logger(session_id="abc-123")
    logger.log_radiologist(state, result)
"""

import os
from typing import Optional
from app.config import settings


def get_logger(session_id: str = None, image_paths: list = None, views: list = None, patient_id: str = None, workflow_type: str = "debate"):
    """
    Get the appropriate logger instance (Supabase or SQLite).
    
    Args:
        session_id: Unique session ID (auto-generated if not provided)
        image_paths: List of paths to input X-ray images
        views: List of corresponding view names
        patient_id: Optional FHIR patient ID
        workflow_type: 'debate' or 'legacy'
    
    Returns:
        AgentLogger instance
    """
    db_mode = getattr(settings, "DATABASE_MODE", "supabase").lower()
    
    if db_mode == "sqlite":
        try:
            from db.logger import AgentLogger
            return AgentLogger(session_id, image_paths, views, patient_id, workflow_type)
        except ImportError as e:
            print(f"[DB Adapter] ERROR: SQLite logger not available: {e}")
            raise
    else:
        try:
            from db.supabase_logger import AgentLogger
            return AgentLogger(session_id, image_paths, views, patient_id, workflow_type)
        except ImportError as e:
            print(f"[DB Adapter] ERROR: Supabase not available: {e}")
            raise RuntimeError("Supabase must be installed and configured for cloud mode.")


def check_database_health() -> dict:
    """
    Check Supabase database connection health.
    
    Returns:
        Dictionary with health status and details
    """
    result = {
        'mode': 'supabase',
        'healthy': False,
        'details': {}
    }
    
    try:
        from db.supabase_connection import health_check, SUPABASE_URL
        result['healthy'] = health_check()
        result['details'] = {
            'url': SUPABASE_URL,
            'connection': 'OK' if result['healthy'] else 'FAILED'
        }
    except Exception as e:
        result['healthy'] = False
        result['details']['error'] = str(e)
    
    return result


def migrate_to_cloud(sqlite_db_path: str = None):
    """
    Migrate data from local SQLite to Supabase cloud.
    
    [DELETED IN PRODUCTION]
    """
    print("[Migration] SQLite fallback has been removed. Migration is disabled.")
    return
