"""
Past Mistakes Repository — Supabase-Only Backend

Provides a single concrete implementation backed by Supabase pgvector HNSW:
  - SupabasePastMistakesRepository  : pgvector cosine search via HNSW over RPC

Usage
-----
    from db.past_mistakes_repository import get_past_mistakes_repository

    repo = get_past_mistakes_repository()          # always Supabase-backed
    results = repo.retrieve_similar_mistakes(...)  # raises RuntimeError if creds missing

Re-ranking is NOT done inside this layer — callers pass raw results to
db.rerank_mistakes.rerank_mistakes() unchanged.

Environment variables required
------------------------------
    SUPABASE_URL          Supabase project REST URL
    SUPABASE_SERVICE_KEY  Service-role key (bypasses RLS on past_mistakes table)

The application will fail fast at startup if either variable is absent.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class PastMistakesRepository(ABC):
    """
    Abstract interface for past-mistake vector retrieval.

    Implementations must return a list of dicts with these mandatory keys
    (so callers and re-rankers can rely on a stable schema):

        mistake_id, session_id, image_path,
        original_diagnosis, corrected_diagnosis, disease_type,
        error_type, severity_level,
        kle_uncertainty, safety_score,
        chexbert_labels, clinical_summary, debate_summary,
        created_at,
        similarity          ← cosine similarity (0-1, higher = more similar)
    """

    @abstractmethod
    def retrieve_similar_mistakes(
        self,
        disease_type: str,
        embedding: np.ndarray,
        kle_uncertainty_range: Optional[Tuple[float, float]] = None,
        error_types: Optional[List[str]] = None,
        severity_min: int = 1,
        top_k: int = 5,
        similarity_threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Supabase Implementation
# ---------------------------------------------------------------------------

# SQL DDL that must be run once in the Supabase SQL editor / migrations:
#
#   -- Enable pgvector
#   create extension if not exists vector;
#
#   -- Add pgvector column to past_mistakes (run only if not already done)
#   -- alter table past_mistakes add column if not exists case_embedding vector(384);
#
#   -- HNSW index for cosine similarity
#   create index if not exists idx_pm_case_embedding_hnsw
#       on past_mistakes
#       using hnsw (case_embedding vector_cosine_ops);
#
#   -- RPC function used by SupabasePastMistakesRepository
#   create or replace function match_mistakes(
#       query_embedding vector(384),
#       disease_type    text,
#       kle_min         float,
#       kle_max         float,
#       severity_min    int,
#       top_k           int
#   )
#   returns table (
#       mistake_id           text,
#       session_id           text,
#       image_path           text,
#       original_diagnosis   text,
#       corrected_diagnosis  text,
#       disease_type         text,
#       error_type           text,
#       severity_level       int,
#       kle_uncertainty      float,
#       safety_score         float,
#       chexbert_labels      text,
#       clinical_summary     text,
#       debate_summary       text,
#       created_at           timestamptz,
#       similarity           float
#   )
#   language sql stable
#   as $$
#       select
#           mistake_id,
#           session_id,
#           image_path,
#           original_diagnosis,
#           corrected_diagnosis,
#           disease_type,
#           error_type,
#           severity_level,
#           kle_uncertainty,
#           safety_score,
#           chexbert_labels::text,
#           clinical_summary,
#           debate_summary,
#           created_at,
#           1 - (case_embedding <=> query_embedding) as similarity
#       from past_mistakes
#       where disease_type = match_mistakes.disease_type
#         and kle_uncertainty between kle_min and kle_max
#         and severity_level   >= severity_min
#       order by case_embedding <=> query_embedding  -- ascending distance = most similar first
#       limit top_k * 2   -- over-fetch so caller can threshold-filter
#   $$;
#
# IMPORTANT: the index must exist BEFORE running the function or searches
# will fall back to sequential scan (still correct, just slower).

_SUPABASE_DDL_SNIPPET = """-- See db/past_mistakes_repository.py module docstring for full DDL."""


class SupabasePastMistakesRepository(PastMistakesRepository):
    """
    Retrieves similar past mistakes from Supabase using pgvector HNSW cosine search.

    Calls the ``match_mistakes`` SQL RPC function which uses the
    ``<=>`` (pgvector cosine distance) operator ordered ascending so that
    the HNSW index is exercised.  Similarity returned = 1 - distance.
    """

    backend_name = "supabase_hnsw"

    def __init__(self) -> None:
        from db.supabase_connection import get_service_client
        self._get_client = get_service_client  # service-role key bypasses RLS on past_mistakes

    def retrieve_similar_mistakes(
        self,
        disease_type: str,
        embedding: np.ndarray,
        kle_uncertainty_range: Optional[Tuple[float, float]] = None,
        error_types: Optional[List[str]] = None,
        severity_min: int = 1,
        top_k: int = 5,
        similarity_threshold: float = 0.75,
    ) -> List[Dict[str, Any]]:
        if embedding.shape != (384,):
            raise ValueError(f"embedding must be 384-dim, got {embedding.shape}")

        kle_min = kle_uncertainty_range[0] if kle_uncertainty_range else 0.0
        kle_max = kle_uncertainty_range[1] if kle_uncertainty_range else 1.0

        # Supabase RPC — passes the embedding as a plain Python list
        client = self._get_client()
        response = client.rpc(
            "match_mistakes",
            {
                "query_embedding": embedding.tolist(),
                "disease_type": disease_type,
                "kle_min": float(kle_min),
                "kle_max": float(kle_max),
                "severity_min": int(severity_min),
                "top_k": int(top_k * 2),   # over-fetch; we threshold-filter below
            },
        ).execute()

        rows = response.data or []
        logger.info(
            f"[REPO:supabase_hnsw] Retrieved {len(rows)} candidates "
            f"for disease_type={disease_type!r}"
        )

        results: List[Dict[str, Any]] = []
        for row in rows:
            sim = float(row.get("similarity", 0.0))
            if sim < similarity_threshold:
                continue

            # Optionally filter by error_types (can't easily push into RPC without
            # making the function more complex; handle here for simplicity)
            if error_types and row.get("error_type") not in error_types:
                continue

            created_at = row.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

            chexbert_raw = row.get("chexbert_labels")
            chexbert = {}
            if chexbert_raw:
                try:
                    chexbert = json.loads(chexbert_raw) if isinstance(chexbert_raw, str) else chexbert_raw
                except Exception:
                    chexbert = {}

            results.append(
                {
                    "mistake_id": row.get("mistake_id"),
                    "session_id": row.get("session_id"),
                    "image_path": row.get("image_path"),
                    "original_diagnosis": row.get("original_diagnosis"),
                    "corrected_diagnosis": row.get("corrected_diagnosis"),
                    "disease_type": row.get("disease_type"),
                    "error_type": row.get("error_type"),
                    "severity_level": row.get("severity_level"),
                    "kle_uncertainty": row.get("kle_uncertainty"),
                    "safety_score": row.get("safety_score"),
                    "chexbert_labels": chexbert,
                    "clinical_summary": row.get("clinical_summary"),
                    "debate_summary": row.get("debate_summary"),
                    "created_at": created_at,
                    "similarity": sim,
                }
            )

            if len(results) >= top_k:
                break

        logger.info(
            f"[REPO:supabase_hnsw] {len(results)} results after threshold "
            f"(threshold={similarity_threshold})"
        )
        return results


# ---------------------------------------------------------------------------
# Factory — always returns Supabase backend; fails fast on missing creds
# ---------------------------------------------------------------------------

def get_past_mistakes_repository() -> PastMistakesRepository:
    """
    Return the Supabase-backed past-mistakes repository.

    Reads ``SUPABASE_URL`` and ``SUPABASE_SERVICE_KEY`` from environment
    variables and raises ``RuntimeError`` immediately if either is absent.
    DuckDB is not used at runtime; use ``scripts/migrate_duckdb_to_supabase.py``
    for one-time data migration only.
    """
    url = os.environ.get("SUPABASE_URL") or ""
    key = os.environ.get("SUPABASE_SERVICE_KEY") or ""

    missing = []
    if not url:
        missing.append("SUPABASE_URL")
    if not key:
        missing.append("SUPABASE_SERVICE_KEY")

    if missing:
        raise RuntimeError(
            "[REPO] Supabase credentials missing — cannot start past-mistakes backend. "
            f"Please set the following environment variables: {', '.join(missing)}. "
            "Refer to .env.example for guidance."
        )

    repo = SupabasePastMistakesRepository()
    logger.info("[REPO] Past-mistakes backend: supabase (pgvector HNSW)")
    return repo
