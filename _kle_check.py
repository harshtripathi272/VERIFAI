"""
KLE Reference Audit Script
Scans all .py files for KLE references, excluding:
- uncertainty/kle.py (intentionally kept as embedding utility)
- venv / __pycache__
- .git
"""
import pathlib, re, sys, collections

SKIP_DIRS  = {"venv", "__pycache__", ".git", ".gemini", "build", "dist", "node_modules"}
SKIP_FILES = {
    "uncertainty/kle.py",
    "_kle_check.py",
}
KLE_PATTERN = re.compile(r'kle', re.IGNORECASE)

root = pathlib.Path(".")
# file -> [(lineno, text)]
hits = collections.defaultdict(list)

for p in sorted(root.rglob("*.py")):
    parts = set(p.parts)
    if parts & SKIP_DIRS:
        continue
    rel = str(p).replace("\\", "/")
    if any(rel.endswith(s) for s in SKIP_FILES):
        continue

    for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        if KLE_PATTERN.search(line):
            hits[rel].append((i, line.strip()[:100]))

total = sum(len(v) for v in hits.values())
if not hits:
    print("=" * 70)
    print("  CLEAN - zero KLE references found outside kle.py")
    print("=" * 70)
else:
    print(f"{'='*70}")
    print(f"  FOUND {total} KLE reference(s) across {len(hits)} file(s):")
    print(f"{'='*70}")
    for f in sorted(hits):
        print(f"\n  [{f}]")
        for ln, txt in hits[f]:
            print(f"    L{ln:4d}: {txt}")

n_files = sum(1 for p in root.rglob("*.py") if not (set(p.parts) & SKIP_DIRS))
print(f"\nTotal .py files scanned: {n_files}")
