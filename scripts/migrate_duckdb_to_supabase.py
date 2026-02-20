"""
migrate_duckdb_to_supabase.py
=============================
One-time migration: copy all rows from the local DuckDB past_mistakes table
into Supabase (Postgres + pgvector).

Usage
-----
    # From the VERIFAI repo root:
    python scripts/migrate_duckdb_to_supabase.py

    # Override DuckDB path:
    DUCKDB_PATH=/path/to/verifai_past_mistakes.duckdb python scripts/migrate_duckdb_to_supabase.py

Environment variables required
-------------------------------
    SUPABASE_URL          Supabase project REST URL
    SUPABASE_SERVICE_KEY  Service-role key (bypasses RLS on past_mistakes)

Notes
-----
- Uses upsert (conflict on mistake_id) so the script is safe to re-run.
- FLOAT[384] columns returned by DuckDB are already plain Python lists.
- The script NEVER modifies or deletes the source DuckDB file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so we can import from db.*
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Load .env before importing anything from the project
from dotenv import load_dotenv  # noqa: E402
load_dotenv(_REPO_ROOT / ".env")

import duckdb  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DUCKDB_PATH: str = os.environ.get(
    "DUCKDB_PATH",
    str(_REPO_ROOT / "verifai_past_mistakes.duckdb"),
)
BATCH_SIZE: int = 100

# ---------------------------------------------------------------------------
# Helper: convert a DuckDB row tuple to a Supabase-ready dict
# ---------------------------------------------------------------------------

COLUMN_NAMES = [
    "mistake_id",
    "session_id",
    "image_path",
    "created_at",
    "original_diagnosis",
    "corrected_diagnosis",
    "disease_type",
    "error_type",
    "severity_level",
    "kle_uncertainty",
    "safety_score",
    "chexbert_labels",
    "clinical_summary",
    "debate_summary",
    "case_embedding",
]


def _row_to_dict(row: tuple) -> dict:
    """Convert a DuckDB fetchall() row into a Supabase insert dict."""
    d = dict(zip(COLUMN_NAMES, row))

    # case_embedding: DuckDB returns FLOAT[384] as a Python list already.
    # Supabase vector(384) accepts a plain Python list via the REST API.
    emb = d.get("case_embedding")
    if emb is None:
        raise ValueError(f"NULL embedding for mistake_id={d.get('mistake_id')!r} — skipping.")
    d["case_embedding"] = list(emb)

    # chexbert_labels: stored as JSON string in DuckDB; keep as string for
    # Supabase TEXT column (parsed on read by the repository).
    chexbert = d.get("chexbert_labels")
    if chexbert is not None and not isinstance(chexbert, str):
        d["chexbert_labels"] = json.dumps(chexbert)

    # created_at: DuckDB returns datetime objects; convert to ISO string.
    created_at = d.get("created_at")
    if created_at is not None and hasattr(created_at, "isoformat"):
        d["created_at"] = created_at.isoformat()

    return d


# ---------------------------------------------------------------------------
# Main migration logic
# ---------------------------------------------------------------------------

def migrate() -> None:
    # ------------------------------------------------------------------
    # 1. Validate Supabase credentials (fail fast)
    # ------------------------------------------------------------------
    supabase_url = os.environ.get("SUPABASE_URL", "")
    service_key = os.environ.get("SUPABASE_SERVICE_KEY", "")

    missing = []
    if not supabase_url:
        missing.append("SUPABASE_URL")
    if not service_key:
        missing.append("SUPABASE_SERVICE_KEY")
    if missing:
        print(
            f"[ERROR] Missing required environment variables: {', '.join(missing)}\n"
            "Set them in your .env file or as shell exports before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Connect to Supabase
    # ------------------------------------------------------------------
    from supabase import create_client

    print(f"[MIGRATION] Connecting to Supabase: {supabase_url}")
    supabase = create_client(supabase_url, service_key)

    # ------------------------------------------------------------------
    # 3. Open DuckDB in read-only mode
    # ------------------------------------------------------------------
    if not Path(DUCKDB_PATH).exists():
        print(f"[ERROR] DuckDB file not found: {DUCKDB_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"[MIGRATION] Opening DuckDB (read-only): {DUCKDB_PATH}")
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)

    # Count total rows
    total: int = conn.execute("SELECT COUNT(*) FROM past_mistakes").fetchone()[0]
    print(f"[MIGRATION] Found {total} row(s) in DuckDB past_mistakes table.")

    if total == 0:
        print("[MIGRATION] Nothing to migrate. Exiting.")
        conn.close()
        return

    # ------------------------------------------------------------------
    # 4. Stream rows and upsert into Supabase in batches
    # ------------------------------------------------------------------
    select_columns = [
    col if col != "case_embedding"
    else "case_embedding::FLOAT[] AS case_embedding"
    for col in COLUMN_NAMES
]

    rows = conn.execute(
        f"SELECT {', '.join(select_columns)} FROM past_mistakes"
    ).fetchall()


    migrated = 0
    skipped = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_rows = rows[batch_start : batch_start + BATCH_SIZE]
        batch_dicts: list[dict] = []

        for row in batch_rows:
            try:
                batch_dicts.append(_row_to_dict(row))
            except ValueError as exc:
                print(f"[MIGRATION] WARNING — {exc}")
                skipped += 1
                continue

        if not batch_dicts:
            continue

        # Upsert: on_conflict="mistake_id" makes re-runs idempotent.
        supabase.table("past_mistakes").upsert(
            batch_dicts,
            on_conflict="mistake_id",
        ).execute()

        migrated += len(batch_dicts)
        print(f"[MIGRATION] Migrated {migrated}/{total} rows.")

    # ------------------------------------------------------------------
    # 5. Verify: query Supabase row count
    # ------------------------------------------------------------------
    print("\n[MIGRATION] Verifying Supabase row count …")
    result = supabase.table("past_mistakes").select("mistake_id", count="exact").execute()
    supabase_count: int = result.count if result.count is not None else -1

    print(
        f"[MIGRATION] Complete!\n"
        f"  DuckDB rows read    : {total}\n"
        f"  Rows upserted       : {migrated}\n"
        f"  Rows skipped (error): {skipped}\n"
        f"  Supabase total rows : {supabase_count}"
    )

    if supabase_count < migrated:
        print(
            "[MIGRATION] WARNING: Supabase count is lower than expected. "
            "Check for upsert conflicts or RLS policy issues.",
            file=sys.stderr,
        )
    else:
        print("[MIGRATION] ✓ All rows confirmed in Supabase.")


if __name__ == "__main__":
    migrate()
