"""
VERIFAI Database Module

Structured SQL logging for all agent interactions, debates, and workflow sessions.
"""

from db.connection import get_db, init_db
from db.logger import AgentLogger

__all__ = ["get_db", "init_db", "AgentLogger"]
