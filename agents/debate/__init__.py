"""
Debate Agent Module

Implements adversarial debate between Critic and (Historian + Literature) agents.
"""

from .agent import debate_node, DebateRound, DebateOutput

__all__ = ["debate_node", "DebateRound", "DebateOutput"]
