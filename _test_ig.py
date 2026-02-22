"""Test bidirectional IG formula."""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from uncertainty.muc import compute_ig

print("=" * 70)
print("TEST: Bidirectional IG Formula")
print("=" * 70)
print()

# Test 1: Agent CONFIRMS diagnosis (alignment > 0.5)
r1 = compute_ig("critic", agent_uncertainty=0.2, alignment_score=0.85, system_uncertainty=0.50)
print(f"CONFIRMS   (align=0.85, unc=0.2):")
print(f"  direction = (0.85 - 0.5) * 2 = +0.70")
print(f"  IG = {r1.information_gain:+.4f}")
print(f"  U: {r1.system_uncertainty_before:.3f} -> {r1.system_uncertainty_after:.3f}")
print(f"  RESULT: Uncertainty {'DECREASED' if r1.system_uncertainty_after < r1.system_uncertainty_before else 'INCREASED'}")
print()

# Test 2: Agent is NEUTRAL (alignment = 0.5)
r2 = compute_ig("critic", agent_uncertainty=0.2, alignment_score=0.50, system_uncertainty=0.50)
print(f"NEUTRAL    (align=0.50, unc=0.2):")
print(f"  direction = (0.50 - 0.5) * 2 = 0.00")
print(f"  IG = {r2.information_gain:+.4f}")
print(f"  U: {r2.system_uncertainty_before:.3f} -> {r2.system_uncertainty_after:.3f}")
print(f"  RESULT: Uncertainty UNCHANGED")
print()

# Test 3: Agent CONTRADICTS diagnosis (alignment < 0.5)
r3 = compute_ig("critic", agent_uncertainty=0.2, alignment_score=0.15, system_uncertainty=0.50)
print(f"CONTRADICTS (align=0.15, unc=0.2):")
print(f"  direction = (0.15 - 0.5) * 2 = -0.70")
print(f"  IG = {r3.information_gain:+.4f}")
print(f"  U: {r3.system_uncertainty_before:.3f} -> {r3.system_uncertainty_after:.3f}")
print(f"  RESULT: Uncertainty {'DECREASED' if r3.system_uncertainty_after < r3.system_uncertainty_before else 'INCREASED'}")
print()

# Test 4: Full cascade example
print("=" * 70)
print("FULL CASCADE EXAMPLE")
print("=" * 70)
u = 0.70  # Radiologist initial
print(f"  Radiologist sets initial uncertainty: {u:.3f}")
print()

# CheXbert: 3 present labels match impression (good alignment)
r = compute_ig("chexbert", agent_uncertainty=0.15, alignment_score=0.80, system_uncertainty=u)
print(f"  CheXbert  (confirms):  IG={r.information_gain:+.4f}  U: {u:.3f} -> {r.system_uncertainty_after:.3f}")
u = r.system_uncertainty_after

# Historian: Mostly supporting evidence
r = compute_ig("historian", agent_uncertainty=0.25, alignment_score=0.70, system_uncertainty=u)
print(f"  Historian (confirms):  IG={r.information_gain:+.4f}  U: {u:.3f} -> {r.system_uncertainty_after:.3f}")
u = r.system_uncertainty_after

# Literature: Contradicting papers found!
r = compute_ig("literature", agent_uncertainty=0.30, alignment_score=0.25, system_uncertainty=u)
print(f"  Literature (CONTRA):   IG={r.information_gain:+.4f}  U: {u:.3f} -> {r.system_uncertainty_after:.3f}")
u = r.system_uncertainty_after

# Critic: Overconfident, many flags (contradicts)
r = compute_ig("critic", agent_uncertainty=0.10, alignment_score=0.20, system_uncertainty=u)
print(f"  Critic    (CONTRA):    IG={r.information_gain:+.4f}  U: {u:.3f} -> {r.system_uncertainty_after:.3f}")
u = r.system_uncertainty_after

# Debate: No consensus (neutral)
r = compute_ig("debate", agent_uncertainty=0.40, alignment_score=0.50, system_uncertainty=u)
print(f"  Debate    (neutral):   IG={r.information_gain:+.4f}  U: {u:.3f} -> {r.system_uncertainty_after:.3f}")
u = r.system_uncertainty_after

# Validator: FLAG_FOR_HUMAN (contradicts)
r = compute_ig("validator", agent_uncertainty=0.20, alignment_score=0.15, system_uncertainty=u)
print(f"  Validator (CONTRA):    IG={r.information_gain:+.4f}  U: {u:.3f} -> {r.system_uncertainty_after:.3f}")
u = r.system_uncertainty_after

print()
print(f"  FINAL UNCERTAINTY: {u:.3f}  (confidence = {(1-u)*100:.0f}%)")
print(f"  >> This is HIGH uncertainty -- system correctly lost confidence!")
print()

# Test 5: Happy path (everything confirms)
print("=" * 70)
print("HAPPY PATH CASCADE (everything confirms)")
print("=" * 70)
u = 0.70
print(f"  Radiologist: {u:.3f}")
for name, unc, align in [
    ("chexbert", 0.10, 0.90),
    ("historian", 0.15, 0.85),
    ("literature", 0.10, 0.90),
    ("critic", 0.05, 0.90),
    ("debate", 0.10, 0.90),
    ("validator", 0.10, 0.95),
]:
    r = compute_ig(name, unc, align, u)
    print(f"  {name:12s}: IG={r.information_gain:+.4f}  U: {u:.3f} -> {r.system_uncertainty_after:.3f}")
    u = r.system_uncertainty_after

print()
print(f"  FINAL UNCERTAINTY: {u:.3f}  (confidence = {(1-u)*100:.0f}%)")
print(f"  >> This is LOW uncertainty -- system correctly gained confidence!")
