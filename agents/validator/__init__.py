"""
Validator Agent Module

Provides three tools for validation when debate fails to reach consensus:
1. CXR-RePaiR Retrieval: Find similar historical cases
2. RadGraph Entity Matching: Verify clinical facts
3. Clinical Rules Engine: Check for contradictions
"""

from .agent import validator_node, initialize_validator_tools

__all__ = ["validator_node", "initialize_validator_tools"]
