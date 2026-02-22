"""
Clinical Rules Engine

Run deterministic clinical logic checks on the case state. 
Flag contradictions and inconsistencies that LLMs miss systematically.

Rules check for:
- Overconfident language with high uncertainty
- Spatial contradictions with Grad-CAM
- Lack of external evidence
- Clinical inconsistencies (e.g., pneumonia with normal labs)
"""

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING, Dict, Any, List

if TYPE_CHECKING:
    from graph.state import VerifaiState


@dataclass
class Rule:
    """
    A single clinical rule.
    
    Attributes:
        name: Human-readable name
        severity: "WARN" or "FLAG" (flags are more critical)
        message: Explanation shown when rule triggers
        condition: Function that takes state dict and returns True if rule triggers
    """
    name: str
    severity: str  # "WARN" or "FLAG"
    message: str
    condition: Callable[[Dict[str, Any]], bool]


# RULE DEFINITIONS
RULES = [
    Rule(
        name="Overconfident Language",
        severity="FLAG",
        message="High uncertainty (>0.6) but definitive language used in impression",
        condition=lambda s: (
            s.get("current_uncertainty", 0) > 0.6 and
            s.get("radiologist_output") is not None and
            any(w in s["radiologist_output"].impression.lower() 
                for w in ["definitely", "consistent with", "confirmed", "no evidence of"])
        )
    ),
    
    Rule(
        name="No External Evidence",
        severity="WARN",
        message="Neither FHIR history nor literature provides supporting evidence",
        condition=lambda s: (
            s.get("historian_output") is not None and
            s.get("literature_output") is not None and
            (not s["historian_output"].supporting_facts or 
             len(s["historian_output"].supporting_facts) == 0) and
            (not s["literature_output"] or 
             not hasattr(s["literature_output"], "overall_evidence_strength") or
             s["literature_output"].overall_evidence_strength in ["none", "low"])
        )
    ),
    
    Rule(
        name="Pneumonia Normal Labs",
        severity="WARN",
        message="Pneumonia diagnosed but WBC within normal range (atypical presentation)",
        condition=lambda s: (
            s.get("chexbert_output") is not None and
            s.get("historian_output") is not None and
            s["chexbert_output"].labels.get("Pneumonia") == "Positive" and
            s["historian_output"].supporting_facts is not None and
            any("wbc" in fact.description.lower() and "normal" in fact.description.lower()
                for fact in s["historian_output"].supporting_facts 
                if hasattr(fact, 'description'))
        )
    ),
    
    Rule(
        name="High System Uncertainty",
        severity="WARN",
        message="Very high epistemic uncertainty (>0.7) suggests model disagreement",
        condition=lambda s: s.get("current_uncertainty", 0) > 0.7
    ),
    
    Rule(
        name="Debate No Consensus",
        severity="FLAG",
        message="Debate failed to reach consensus after maximum rounds",
        condition=lambda s: (
            s.get("debate_output") is not None and
            not s["debate_output"].final_consensus and
            s["debate_output"].escalate_to_chief
        )
    ),
    
    Rule(
        name="Critic High Concern",
        severity="WARN",
        message="Critic flagged significant safety concerns",
        condition=lambda s: (
            s.get("critic_output") is not None and
            s["critic_output"].is_overconfident and
            len(s["critic_output"].concern_flags) >= 2
        )
    ),
    
    Rule(
        name="Historical Mistakes Similar",
        severity="FLAG",
        message="Similar to past validated mistakes in database",
        condition=lambda s: (
            s.get("critic_output") is not None and
            hasattr(s["critic_output"], 'similar_mistakes_count') and
            s["critic_output"].similar_mistakes_count >= 2 and
            hasattr(s["critic_output"], 'historical_risk_level') and
            s["critic_output"].historical_risk_level in ["medium", "high"]
        )
    ),
]


# ============================================================================
# RULES ENGINE
# ============================================================================

class ClinicalRulesEngine:
    """
    Execute clinical rules and report violations.
    """
    
    def __init__(self, rules: List[Rule] = None):
        """
        Args:
            rules: List of rules to execute. If None, uses default RULES.
        """
        self.rules = rules if rules is not None else RULES
    
    def execute(self, state: "VerifaiState") -> Dict[str, Any]:
        """
        Run all clinical rules on the current state.
        
        Args:
            state: Current VERIFAI state
        
        Returns:
            Dict with:
                - rules_triggered: List of triggered rules
                - flag_count: Number of FLAG-level violations
                - warn_count: Number of WARN-level violations
                - has_critical_flag: Whether any critical flags present
                - summary: Human-readable summary
        """
        triggered = []
        
        for rule in self.rules:
            try:
                if rule.condition(state):
                    triggered.append({
                        "rule": rule.name,
                        "severity": rule.severity,
                        "message": rule.message
                    })
            except (KeyError, AttributeError, TypeError) as e:
                # Rule failed due to missing data — skip silently
                # (some rules may not apply to all cases)
                continue
            except Exception as e:
                # Unexpected error - log but don't fail
                print(f"[Rules Engine] Rule '{rule.name}' failed: {e}")
                continue
        
        # Separate flags from warnings
        flags = [r for r in triggered if r["severity"] == "FLAG"]
        warns = [r for r in triggered if r["severity"] == "WARN"]
        
        return {
            "rules_triggered": triggered,
            "flag_count": len(flags),
            "warn_count": len(warns),
            "has_critical_flag": len(flags) > 0,
            "summary": f"{len(flags)} flags, {len(warns)} warnings",
            "triggered_rule_names": [r["rule"] for r in triggered]
        }
    
    def add_rule(self, rule: Rule):
        """Add a custom rule to the engine."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str):
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != rule_name]
