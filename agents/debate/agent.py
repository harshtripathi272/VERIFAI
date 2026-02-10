"""
Debate Agent Node

Implements adversarial debate between:
- CRITIC: Challenges the diagnosis, looks for overconfidence
- EVIDENCE TEAM (Historian + Literature): Defends/refines with clinical context and literature

The debate runs for multiple rounds until consensus or escalation to Chief.
"""

import json
from typing import List, Optional
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from graph.state import DebateArgument, DebateRound, DebateOutput
import re

from app.config import settings

class DebateOrchestrator:
    """
    Orchestrates the debate between Critic and Evidence Team.
    
    Debate Flow:
    1. Critic raises challenges based on overconfidence signals
    2. Historian responds with clinical context
    3. Literature responds with evidence
    4. Evaluate if consensus reached
    5. Repeat or escalate
    """
    
    def __init__(self, max_rounds: int = 3, consensus_threshold: float = 0.15):
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold  # Max disagreement for consensus
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def _extract_primary_diagnosis(self, impression: str) -> str:
        """
        Extract the primary diagnosis from the IMPRESSION text.
        
        Simple heuristic: Look for phrases after "consistent with", "suggestive of", etc.
        """
        if not impression:
            return "Unknown"
        
        impression_lower = impression.lower()
        
        # Try to extract diagnosis from common patterns
        patterns = [
            r'consistent with ([^.;]+)',
            r'suggestive of ([^.;]+)',
            r'diagnosis[:\s]+([^.;]+)',
            r'impression[:\s]+([^.;]+)',
            r'findings.{0,30}(?:raise concern for|concerning for) ([^.;]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, impression_lower)
            if match:
                diagnosis = match.group(1).strip()
                # Clean up and capitalize
                diagnosis = diagnosis.split(',')[0]  # Take first part before comma
                diagnosis = ' '.join(word.capitalize() for word in diagnosis.split())
                return diagnosis
        
        # Fallback: extract first sentence
        first_sentence = impression.split('.')[0].strip()
        if len(first_sentence) < 100:
            return first_sentence
        
        return "Complex diagnostic impression"
    
    def _generate_critic_challenge(
        self,
        radiologist_output,
        critic_output,
        round_num: int,
        previous_rounds: List[DebateRound]
    ) -> DebateArgument:
        """Generate critic's challenge for this round."""
        
        # First round: Use initial concerns
        if round_num == 1:
            concerns = critic_output.concern_flags if critic_output else []
            
            if concerns and critic_output.is_overconfident:
                challenge_text = f"Safety concern detected: {'; '.join(concerns[:2])}"
                if critic_output.recommended_hedging:
                    challenge_text += f" Suggestion: {critic_output.recommended_hedging[:100]}"
            elif concerns:
                challenge_text = f"Moderate concerns: {'; '.join(concerns[:2])}"
            else:
                challenge_text = "No significant concerns identified. Requesting evidence validation."
            
            confidence_impact = -0.05 if critic_output.is_overconfident else 0.0
            
            return DebateArgument(
                agent="critic",
                position="challenge",
                argument=challenge_text,
                confidence_impact=confidence_impact,
                evidence_refs=[f"safety_score={critic_output.safety_score:.2f}" if critic_output else ""]
            )
        
        # Subsequent rounds: Challenge based on previous responses
        last_round = previous_rounds[-1] if previous_rounds else None
        if last_round:
            # Check if historian/literature provided strong evidence
            hist_impact = last_round.historian_response.confidence_impact if last_round.historian_response else 0
            lit_impact = last_round.literature_response.confidence_impact if last_round.literature_response else 0
            
            if hist_impact + lit_impact > 0.1:
                # Evidence was strong, reduce challenge intensity
                return DebateArgument(
                    agent="critic",
                    position="challenge",
                    argument="Evidence appears supportive. Verifying consistency with imaging findings.",
                    confidence_impact=-0.02,
                    evidence_refs=["reduced_challenge_intensity"]
                )
            else:
                # Evidence was weak, maintain challenge
                return DebateArgument(
                    agent="critic",
                    position="challenge",
                    argument="Evidence insufficient to resolve uncertainty. Recommend additional validation.",
                    confidence_impact=-0.08,
                    evidence_refs=["maintained_challenge"]
                )
        
        return DebateArgument(
            agent="critic",
            position="challenge",
            argument="Continuing evaluation.",
            confidence_impact=0.0
        )
    
    def _generate_historian_response(
        self,
        historian_output,
        critic_challenge: DebateArgument,
        radiologist_output
    ) -> DebateArgument:
        """Generate historian's response to critic's challenge."""
        
        if not historian_output:
            return DebateArgument(
                agent="historian",
                position="refine",
                argument="No clinical history available to support or refute.",
                confidence_impact=0.0
            )
        
        supporting = historian_output.supporting_facts
        contradicting = historian_output.contradicting_facts
        
        # Build response based on evidence balance
        if len(supporting) > len(contradicting):
            # Strong clinical support
            support_text = "; ".join([f.description for f in supporting[:3]])
            return DebateArgument(
                agent="historian",
                position="support",
                argument=f"Clinical history supports diagnosis: {support_text}",
                confidence_impact=min(0.15, len(supporting) * 0.05),
                evidence_refs=[f.fhir_resource_id for f in supporting[:3]]
            )
        elif len(contradicting) > len(supporting):
            # Clinical concerns
            contra_text = "; ".join([f.description for f in contradicting[:2]])
            return DebateArgument(
                agent="historian",
                position="refine",
                argument=f"Clinical history raises concerns: {contra_text}. Consider differential.",
                confidence_impact=-min(0.10, len(contradicting) * 0.04),
                evidence_refs=[f.fhir_resource_id for f in contradicting[:2]]
            )
        else:
            # Mixed evidence
            return DebateArgument(
                agent="historian",
                position="refine",
                argument=f"Clinical history is mixed. {historian_output.clinical_summary[:200]}",
                confidence_impact=historian_output.confidence_adjustment,
                evidence_refs=[]
            )
    
    def _generate_literature_response(
        self,
        literature_output,
        critic_challenge: DebateArgument,
        radiologist_output
    ) -> DebateArgument:
        """Generate literature agent's response to critic's challenge."""
        
        if not literature_output:
            return DebateArgument(
                agent="literature",
                position="refine",
                argument="No literature evidence retrieved.",
                confidence_impact=0.0
            )
        
        # Handle string output (from optimized agent)
        if isinstance(literature_output, str):
            # Parse the summary
            if "No relevant literature" in literature_output:
                return DebateArgument(
                    agent="literature",
                    position="refine",
                    argument="Literature search found no directly relevant studies.",
                    confidence_impact=0.0
                )
            else:
                # Assume positive evidence
                return DebateArgument(
                    agent="literature",
                    position="support",
                    argument=f"Literature evidence: {literature_output[:300]}",
                    confidence_impact=0.08,
                    evidence_refs=["literature_search"]
                )
        
        # Handle structured output
        citations = literature_output.citations if hasattr(literature_output, 'citations') else []
        evidence_strength = getattr(literature_output, 'overall_evidence_strength', 'low')
        
        if not citations:
            return DebateArgument(
                agent="literature",
                position="refine",
                argument="No relevant literature citations found.",
                confidence_impact=0.0
            )
        
        # Build response based on evidence strength
        high_evidence = [c for c in citations if c.evidence_strength == "high"]
        
        if evidence_strength == "high" or len(high_evidence) >= 2:
            cite_text = "; ".join([f"{c.authors[0] if c.authors else 'Unknown'} et al. ({c.year})" for c in citations[:3]])
            return DebateArgument(
                agent="literature",
                position="support",
                argument=f"Strong literature support: {cite_text}. {citations[0].relevance_summary[:150] if citations else ''}",
                confidence_impact=0.12,
                evidence_refs=[c.pmid for c in citations[:3]]
            )
        elif evidence_strength == "medium":
            return DebateArgument(
                agent="literature",
                position="support",
                argument=f"Moderate literature support from {len(citations)} studies.",
                confidence_impact=0.06,
                evidence_refs=[c.pmid for c in citations[:3]]
            )
        else:
            return DebateArgument(
                agent="literature",
                position="refine",
                argument=f"Limited literature evidence. Only {len(citations)} marginally relevant studies found.",
                confidence_impact=0.02,
                evidence_refs=[c.pmid for c in citations[:2]]
            )
    
    def _check_consensus(
        self,
        critic_arg: DebateArgument,
        historian_arg: DebateArgument,
        literature_arg: DebateArgument
    ) -> tuple[bool, float]:
        """
        Check if the round reached consensus.
        
        Returns: (consensus_reached, net_confidence_delta)
        """
        # Calculate net impact
        total_impact = (
            critic_arg.confidence_impact +
            historian_arg.confidence_impact +
            literature_arg.confidence_impact
        )
        
        # Check for strong disagreement
        positions = [critic_arg.position, historian_arg.position, literature_arg.position]
        
        # Consensus if all support or all refine in same direction
        if historian_arg.position == "support" and literature_arg.position == "support":
            if critic_arg.confidence_impact > -0.05:  # Critic not strongly challenging
                return True, total_impact
        
        # Check if confidence impacts are aligned
        impacts = [critic_arg.confidence_impact, historian_arg.confidence_impact, literature_arg.confidence_impact]
        
        # If all positive or all negative (aligned)
        if all(i >= 0 for i in impacts) or all(i <= 0 for i in impacts):
            return True, total_impact
        
        # Check disagreement magnitude
        max_disagreement = max(impacts) - min(impacts)
        if max_disagreement <= self.consensus_threshold:
            return True, total_impact
        
        return False, total_impact
    
    def run_debate(
        self,
        radiologist_output,
        critic_output,
        historian_output,
        literature_output
    ) -> DebateOutput:
        """
        Run the full debate process.
        
        Returns DebateOutput with consensus or escalation decision.
        """
        rounds: List[DebateRound] = []
        total_adjustment = 0.0
        
        # Initial confidence: Use inverse of KLE uncertainty as proxy
        # or extract from text if possible
        initial_confidence = 0.5
        
        if radiologist_output:
            # Try to get KLE uncertainty from somewhere (would be passed in state)
            # For now, use a default or try to infer from critic
            if critic_output and hasattr(critic_output, 'safety_score'):
                # Use safety score as proxy for confidence
                initial_confidence = critic_output.safety_score
            else:
                # Moderate default
                initial_confidence = 0.6
        
        current_confidence = initial_confidence
        
        for round_num in range(1, self.max_rounds + 1):
            # 1. Critic challenge
            critic_challenge = self._generate_critic_challenge(
                radiologist_output, critic_output, round_num, rounds
            )
            
            # 2. Evidence team responds (in parallel)
            historian_future = self.executor.submit(
                self._generate_historian_response,
                historian_output, critic_challenge, radiologist_output
            )
            literature_future = self.executor.submit(
                self._generate_literature_response,
                literature_output, critic_challenge, radiologist_output
            )
            
            historian_response = historian_future.result(timeout=5)
            literature_response = literature_future.result(timeout=5)
            
            # 3. Check consensus
            consensus_reached, round_delta = self._check_consensus(
                critic_challenge, historian_response, literature_response
            )
            
            total_adjustment += round_delta
            current_confidence = max(0.0, min(0.99, initial_confidence + total_adjustment))
            
            # Record round
            debate_round = DebateRound(
                round_number=round_num,
                critic_challenge=critic_challenge,
                historian_response=historian_response,
                literature_response=literature_response,
                round_consensus="reached" if consensus_reached else None,
                confidence_delta=round_delta
            )
            rounds.append(debate_round)
            
            # If consensus reached, stop
            if consensus_reached:
                # Extract diagnosis from impression text
                diagnosis = None
                if radiologist_output:
                    diagnosis = self._extract_primary_diagnosis(radiologist_output.impression)
                
                return DebateOutput(
                    rounds=rounds,
                    final_consensus=True,
                    consensus_diagnosis=diagnosis,
                    consensus_confidence=current_confidence,
                    escalate_to_chief=False,
                    debate_summary=f"Consensus reached in round {round_num}. Final confidence: {current_confidence:.2%}",
                    total_confidence_adjustment=total_adjustment
                )
        
        # No consensus after max rounds -> escalate to Chief
        diagnosis = None
        if radiologist_output:
            diagnosis = self._extract_primary_diagnosis(radiologist_output.impression)
        
        return DebateOutput(
            rounds=rounds,
            final_consensus=False,
            consensus_diagnosis=diagnosis,
            consensus_confidence=current_confidence,
            escalate_to_chief=True,
            escalation_reason=f"No consensus after {self.max_rounds} debate rounds. Confidence adjustment: {total_adjustment:+.2f}",
            debate_summary=f"Debate inconclusive. Escalating to Chief for final arbitration.",
            total_confidence_adjustment=total_adjustment
        )


def debate_node(state) -> dict:
    """
    Debate node for LangGraph workflow.
    
    Runs adversarial debate between Critic and Evidence Team (Historian + Literature).
    Consensus is determined solely by DebateOrchestrator's confidence impact heuristics.
    """
    orchestrator = DebateOrchestrator(
        max_rounds=settings.DEBATE_MAX_ROUNDS if hasattr(settings, 'DEBATE_MAX_ROUNDS') else 3,
        consensus_threshold=settings.DEBATE_CONSENSUS_THRESHOLD if hasattr(settings, 'DEBATE_CONSENSUS_THRESHOLD') else 0.15
    )
    
    debate_output = orchestrator.run_debate(
        radiologist_output=state.get("radiologist_output"),
        critic_output=state.get("critic_output"),
        historian_output=state.get("historian_output"),
        literature_output=state.get("literature_output")
    )
    
    # Build trace
    trace_entries = [
        f"DEBATE: {len(debate_output.rounds)} rounds completed",
        f"DEBATE: Consensus={'YES' if debate_output.final_consensus else 'NO'}",
        f"DEBATE: Confidence adjustment={debate_output.total_confidence_adjustment:+.2%}"
    ]
    
    if debate_output.escalate_to_chief:
        trace_entries.append(f"DEBATE: Escalating to Chief - {debate_output.escalation_reason}")
    
    # Update routing decision based on debate outcome
    routing = "finalize" if debate_output.final_consensus else "chief"
    
    # Update uncertainty based on debate
    new_uncertainty = state.get("current_uncertainty", 0.5)
    if debate_output.final_consensus:
        new_uncertainty = max(0.1, new_uncertainty - 0.2)  # Reduce uncertainty on consensus
    else:
        new_uncertainty = min(0.9, new_uncertainty + 0.1)  # Increase on no consensus
    
    return {
        "debate_output": debate_output,
        "routing_decision": routing,
        "current_uncertainty": new_uncertainty,
        "trace": trace_entries
    }


