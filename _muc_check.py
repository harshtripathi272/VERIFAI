"""Syntax check all MUC-modified files."""
import ast
import sys

files = [
    "uncertainty/muc.py",
    "uncertainty/__init__.py",
    "agents/radiologist/agent.py",
    "agents/chexbert/agent.py",
    "agents/critic/agent.py",
    "agents/debate/agent.py",
    "graph/workflow.py",
]

errors = []
for f in files:
    try:
        with open(f, encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"{f}: OK")
    except SyntaxError as e:
        print(f"{f}: SYNTAX ERROR — {e}")
        errors.append(f)

if errors:
    print(f"\nFAILED: {errors}")
    sys.exit(1)
else:
    print("\nAll syntax checks passed!")
