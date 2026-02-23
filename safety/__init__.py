"""
Medical Safety Guardrails Module

Production-grade safety layer for VERIFAI diagnostic system.
Runs as a final checkpoint before any diagnosis is released.
"""

from .guardrails import run_safety_check, SafetyReport

__all__ = ["run_safety_check", "SafetyReport"]
