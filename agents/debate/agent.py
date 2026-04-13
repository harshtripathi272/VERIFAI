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
                confidence_impact = -0.20
            elif concerns:
                challenge_text = f"Moderate concerns: {'; '.join(concerns[:2])}"
                confidence_impact = -0.15
            else:
                challenge_text = "No significant concerns identified. Requesting evidence validation."
                confidence_impact = 0.0
            
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
            hist_arg = getattr(last_round.historian_response, "argument", "") if last_round.historian_response else ""
            hist_snippet = hist_arg[:80].strip() if hist_arg else "No preceding clinical history"
            
            # Check if historian/literature provided strong evidence
            hist_impact = last_round.historian_response.confidence_impact if last_round.historian_response else 0
            lit_impact = last_round.literature_response.confidence_impact if last_round.literature_response else 0
            
            # Use the round number to cycle through different types of follow-up challenges
            round_num = len(previous_rounds) + 1
            
            if hist_impact + lit_impact > 0.1:
                # Evidence was strong, reduce challenge intensity but ask for specificity
                if round_num % 2 == 0:
                    challenge_msg = f"Historian noted: '{hist_snippet}...'. While supportive, please verify if secondary disease patterns in the patient's history uniquely confirm this."
                else:
                    challenge_msg = f"The historical evidence ('{hist_snippet[:40]}...') aligns well. Literature, are there any recent studies documenting rare contra-indications for this specific presentation?"
                
                return DebateArgument(
                    agent="critic",
                    position="challenge",
                    argument=challenge_msg,
                    confidence_impact=-0.02,
                    evidence_refs=["reduced_challenge_intensity"]
                )
            else:
                # Evidence was weak, maintain challenge and push back
                if round_num % 2 == 0:
                    challenge_msg = f"Historian's previous claim ('{hist_snippet}...') is insufficient to resolve uncertainty. Are there definitive prior reports or contradicting differentials?"
                else:
                    challenge_msg = f"The provided evidence remains weak. We must consider alternative diagnoses. What other pathologies present with these exact visual findings?"
                    
                return DebateArgument(
                    agent="critic",
                    position="challenge",
                    argument=challenge_msg,
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
        radiologist_output,
        round_num: int
    ) -> DebateArgument:
        """Generate historian's response to critic's challenge."""
        
        prefix = f"[Round {round_num}] Replying to Critic ('{critic_challenge.argument[:50]}...'): " if round_num > 1 else ""

        if not historian_output:
            return DebateArgument(
                agent="historian",
                position="refine",
                argument=f"{prefix}No clinical history available to support or refute.",
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
                argument=f"{prefix}Clinical history supports diagnosis: {support_text}",
                confidence_impact=min(0.15, len(supporting) * 0.05),
                evidence_refs=[f.fhir_resource_id for f in supporting[:3]]
            )
        elif len(contradicting) > len(supporting):
            # Clinical concerns
            contra_text = "; ".join([f.description for f in contradicting[:2]])
            return DebateArgument(
                agent="historian",
                position="refine",
                argument=f"{prefix}Clinical history raises concerns: {contra_text}. Consider differential.",
                confidence_impact=-min(0.10, len(contradicting) * 0.04),
                evidence_refs=[f.fhir_resource_id for f in contradicting[:2]]
            )
        else:
            # Mixed evidence
            return DebateArgument(
                agent="historian",
                position="refine",
                argument=f"{prefix}Clinical history is mixed. {historian_output.clinical_summary[:200]}",
                confidence_impact=historian_output.confidence_adjustment,
                evidence_refs=[]
            )
    
    def _generate_literature_response(
        self,
        literature_output,
        critic_challenge: DebateArgument,
        radiologist_output,
        round_num: int
    ) -> DebateArgument:
        """Generate literature agent's response to critic's challenge."""
        
        prefix = f"[Round {round_num}] Addressing Critic ('{critic_challenge.argument[:50]}...'): " if round_num > 1 else ""

        if not literature_output:
            return DebateArgument(
                agent="literature",
                position="refine",
                argument=f"{prefix}No literature evidence retrieved.",
                confidence_impact=0.0
            )
        
        # Handle string output (from refactored parallel-API + MedGemma synthesis agent)
        if isinstance(literature_output, str):
            # No results at all
            if "No relevant literature" in literature_output:
                return DebateArgument(
                    agent="literature",
                    position="refine",
                    argument=f"{prefix}Literature search found no directly relevant studies.",
                    confidence_impact=0.0
                )
            
            # Parse the synthesis text for evidence-strength signals
            output_lower = literature_output.lower()
            
            if any(kw in output_lower for kw in ["strongly support", "high evidence", "robust evidence", "consistent with the literature"]):
                confidence_impact = 0.12
                position = "support"
            elif any(kw in output_lower for kw in ["contradicts", "does not support", "inconsistent with", "argues against"]):
                confidence_impact = -0.05
                position = "refine"
            elif any(kw in output_lower for kw in ["limited evidence", "weak evidence", "insufficient", "no directly relevant"]):
                confidence_impact = 0.02
                position = "refine"
            else:
                # Default: moderate support
                confidence_impact = 0.08
                position = "support"
            
            return DebateArgument(
                agent="literature",
                position=position,
                argument=f"{prefix}Literature evidence: {literature_output[:300]}",
                confidence_impact=confidence_impact,
                evidence_refs=["literature_synthesis"]
            )
        
        # Handle structured output
        citations = literature_output.citations if hasattr(literature_output, 'citations') else []
        evidence_strength = getattr(literature_output, 'overall_evidence_strength', 'low')
        
        if not citations:
            return DebateArgument(
                agent="literature",
                position="refine",
                argument=f"{prefix}No relevant literature citations found.",
                confidence_impact=0.0
            )
        
        # Build response based on evidence strength
        high_evidence = [c for c in citations if c.evidence_strength == "high"]
        
        if evidence_strength == "high" or len(high_evidence) >= 2:
            cite_text = "; ".join([f"{c.authors[0] if c.authors else 'Unknown'} et al. ({c.year})" for c in citations[:3]])
            if round_num > 1:
                base_msg = f"{prefix}Reaffirming strong literature support: {cite_text}. No contra-indications found in the retrieved cohort."
            else:
                base_msg = f"{prefix}Strong literature support: {cite_text}. {citations[0].relevance_summary[:150] if citations else ''}"
                
            return DebateArgument(
                agent="literature",
                position="support",
                argument=base_msg,
                confidence_impact=0.12,
                evidence_refs=[c.pmid for c in citations[:3]]
            )
        elif evidence_strength == "medium":
            if round_num > 1:
                base_msg = f"{prefix}The {len(citations)} retrieved studies continue to offer moderate support, but lack definitive randomized trial data for this edge case."
            else:
                base_msg = f"{prefix}Moderate literature support from {len(citations)} studies."
                
            return DebateArgument(
                agent="literature",
                position="support",
                argument=base_msg,
                confidence_impact=0.06,
                evidence_refs=[c.pmid for c in citations[:3]]
            )
        else:
            if round_num > 1:
                base_msg = f"{prefix}Literature evidence remains limited. The queried references do not strongly distinguish between the primary and secondary differentials."
            else:
                base_msg = f"{prefix}Limited literature evidence. Only {len(citations)} marginally relevant studies found."
                
            return DebateArgument(
                agent="literature",
                position="refine",
                argument=base_msg,
                confidence_impact=0.02,
                evidence_refs=[c.pmid for c in citations[:3]]
            )
    
    def _check_consensus(
        self,
        critic_arg: DebateArgument,
        historian_arg: DebateArgument,
        literature_arg: DebateArgument
    ) -> tuple[bool, float]:
        """
        Check if the round reached consensus.

        Consensus rules (priority order):
        1. Guard: all-zero/neutral round -> NOT consensus (forces substantive debate)
        2. Evidence team both support AND critic not strongly challenging -> consensus
        3. All impacts aligned (all >= 0 or all <= 0) AND |total| > MIN_IMPACT -> consensus
        4. Max disagreement between any two agents <= consensus_threshold -> consensus

        Returns: (consensus_reached, net_confidence_delta)
        """
        impacts = [
            critic_arg.confidence_impact,
            historian_arg.confidence_impact,
            literature_arg.confidence_impact,
        ]
        total_impact = sum(impacts)

        # Rule 0 (Guard): pure-neutral round -> never consensus
        MIN_SUBSTANTIVE_IMPACT = 0.02
        if all(abs(i) < MIN_SUBSTANTIVE_IMPACT for i in impacts):
            return False, total_impact

        # Rule 1: Evidence team both support AND critic conceding
        if historian_arg.position == "support" and literature_arg.position == "support":
            if critic_arg.confidence_impact > -0.05:
                return True, total_impact

        # Rule 2: All aligned AND meaningful net movement
        if all(i >= 0 for i in impacts) or all(i <= 0 for i in impacts):
            if abs(total_impact) >= MIN_SUBSTANTIVE_IMPACT:
                return True, total_impact

        # Rule 3: Disagreement within threshold
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
        
        # Initial confidence: use critic's safety score as proxy
        initial_confidence = 0.5
        
        if radiologist_output:
            if critic_output and hasattr(critic_output, 'safety_score'):
                # Use safety score as proxy for confidence
                initial_confidence = critic_output.safety_score
            else:
                # Moderate default
                initial_confidence = 0.6
        
        current_confidence = initial_confidence
        print(f"\n[DEBATE] Starting debate — initial confidence: {initial_confidence:.2%}")
        
        for round_num in range(1, self.max_rounds + 1):
            # 1. Critic challenge
            critic_challenge = self._generate_critic_challenge(
                radiologist_output, critic_output, round_num, rounds
            )
            
            # 2. Evidence team responds (in parallel)
            historian_future = self.executor.submit(
                self._generate_historian_response,
                historian_output, critic_challenge, radiologist_output, round_num
            )
            literature_future = self.executor.submit(
                self._generate_literature_response,
                literature_output, critic_challenge, radiologist_output, round_num
            )
            
            historian_response = historian_future.result(timeout=5)
            literature_response = literature_future.result(timeout=5)
            
            # 3. Check consensus
            consensus_reached, round_delta = self._check_consensus(
                critic_challenge, historian_response, literature_response
            )
            
            total_adjustment += round_delta
            current_confidence = max(0.0, min(0.99, initial_confidence + total_adjustment))
            
            # --- Per-round uncertainty tracking ---
            system_uncertainty = max(0.01, 1.0 - current_confidence)
            print(f"\n[DEBATE] ─── Round {round_num} ───")
            print(f"  Critic       : Δ = {critic_challenge.confidence_impact:+.3f}  ({critic_challenge.position})")
            print(f"  Historian    : Δ = {historian_response.confidence_impact:+.3f}  ({historian_response.position})")
            print(f"  Literature   : Δ = {literature_response.confidence_impact:+.3f}  ({literature_response.position})")
            print(f"  Round net    : Δ = {round_delta:+.3f}")
            print(f"  Confidence   : {current_confidence:.2%}  (cumulative Δ = {total_adjustment:+.3f})")
            print(f"  Uncertainty  : {system_uncertainty:.2%}")
            print(f"  Consensus    : {'YES ✓' if consensus_reached else 'NO — continuing'}")
            
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
    Uses Dempster-Shafer evidence fusion to combine agent beliefs and compute
    the debate's Information Gain via the MUC framework.
    """
    from uncertainty.muc import compute_ig, compute_debate_ds_fusion

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
    
    # === MUC: Dempster-Shafer Fusion for Debate IG ===
    # Build per-agent confidence/alignment from debate rounds
    critic_conf, hist_conf, lit_conf = 0.5, 0.5, 0.5
    critic_align, hist_align, lit_align = 0.5, 0.5, 0.5
    
    if debate_output.rounds:
        # Aggregate across all rounds
        total_critic_impact = 0.0
        total_hist_impact = 0.0
        total_lit_impact = 0.0
        
        for rnd in debate_output.rounds:
            total_critic_impact += rnd.critic_challenge.confidence_impact
            total_hist_impact += rnd.historian_response.confidence_impact if rnd.historian_response else 0
            total_lit_impact += rnd.literature_response.confidence_impact if rnd.literature_response else 0
        
        n_rounds = len(debate_output.rounds)
        
        # Convert impacts to confidence (how sure each agent is about its position)
        critic_conf = max(0.1, min(0.9, 0.5 + abs(total_critic_impact)))
        hist_conf = max(0.1, min(0.9, 0.5 + abs(total_hist_impact)))
        lit_conf = max(0.1, min(0.9, 0.5 + abs(total_lit_impact)))
        
        # Convert impacts to alignment (positive impact = supports diagnosis)
        critic_align = max(0.05, min(0.95, 0.5 + total_critic_impact * 2))
        hist_align = max(0.05, min(0.95, 0.5 + total_hist_impact * 2))
        lit_align = max(0.05, min(0.95, 0.5 + total_lit_impact * 2))
    
    # Dempster-Shafer fusion
    fused_alignment, fused_uncertainty, conflict_K = compute_debate_ds_fusion(
        critic_confidence=critic_conf,
        critic_alignment=critic_align,
        historian_confidence=hist_conf,
        historian_alignment=hist_align,
        literature_confidence=lit_conf,
        literature_alignment=lit_align,
    )
    
    # Apply IG formula for the debate node
    ig_result = compute_ig(
        agent_name="debate",
        agent_uncertainty=fused_uncertainty,
        alignment_score=fused_alignment,
        system_uncertainty=state.get("current_uncertainty", 0.5),
    )
    
    # Build trace
    trace_entries = [
        f"DEBATE: {len(debate_output.rounds)} rounds completed",
        f"DEBATE: Consensus={'YES' if debate_output.final_consensus else 'NO'}",
        f"DEBATE: Confidence adjustment={debate_output.total_confidence_adjustment:+.3f}",
        f"DEBATE MUC: DS-fusion alignment={fused_alignment:.3f}, uncertainty={fused_uncertainty:.3f}, "
        f"conflict_K={conflict_K:.3f}, IG={ig_result.information_gain:.4f}",
    ]
    
    if debate_output.escalate_to_chief:
        trace_entries.append(f"DEBATE: Escalating to Chief - {debate_output.escalation_reason}")
    
    # Update routing decision based on debate outcome
    routing = "finalize" if debate_output.final_consensus else "chief"
    
    return {
        "debate_output": debate_output,
        "routing_decision": routing,
        "current_uncertainty": ig_result.system_uncertainty_after,
        "trace": trace_entries
    }


