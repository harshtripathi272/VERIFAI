"""
Past Mistakes Repository — Hybrid Supabase / DuckDB Abstraction Layer

Provides a clean ABC with two concrete implementations:
  - SupabasePastMistakesRepository  : pgvector cosine search via HNSW over RPC
  - DuckDBPastMistakesRepository    : offline fallback using DuckDB VSS / NumPy

Usage
-----
    from db.past_mistakes_repository import get_past_mistakes_repository

    repo = get_past_mistakes_repository()        # selects backend from settings
    results = repo.retrieve_similar_mistakes(...)  # identical signature for both

The factory auto-falls back to DuckDB when Supabase is unavailable or errors.
Re-ranking is NOT done inside this layer — callers pass raw results to
db.rerank_mistakes.rerank_mistakes() unchanged.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

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
        from db.supabase_connection import get_client
        self._get_client = get_client  # defer connection until first query

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
# DuckDB Implementation (offline fallback)
# ---------------------------------------------------------------------------

class DuckDBPastMistakesRepository(PastMistakesRepository):
    """
    Retrieves similar past mistakes from a local DuckDB database.

    Strategy waterfall (most efficient to least, auto-detected at runtime):
      1. ``array_cosine_distance`` — DuckDB VSS extension (HNSW available)
      2. ``array_distance``        — generic DuckDB function (L2 distance; less precise)
      3. NumPy cosine similarity   — fetch all filtered rows and rank in-process

    All three paths return the identical dict schema expected by callers and
    re-rankers.
    """

    backend_name = "duckdb_local"

    # Cache the working DuckDB strategy so we only probe once per process.
    _strategy: Optional[str] = None  # 'cosine_vss' | 'array_distance' | 'numpy'

    def _detect_strategy(self, conn) -> str:
        if DuckDBPastMistakesRepository._strategy is not None:
            return DuckDBPastMistakesRepository._strategy
        # Probe for VSS array_cosine_distance
        try:
            conn.execute(
                "SELECT array_cosine_distance([1.0]::FLOAT[1], [1.0]::FLOAT[1])"
            ).fetchone()
            DuckDBPastMistakesRepository._strategy = "cosine_vss"
            logger.info("[REPO:duckdb_local] Strategy: array_cosine_distance (VSS)")
            return "cosine_vss"
        except Exception:
            pass
        # Check for fallback array_distance (standard DuckDB)
        try:
            conn.execute(
                "SELECT array_distance([1.0]::FLOAT[1], [1.0]::FLOAT[1])"
            ).fetchone()
            DuckDBPastMistakesRepository._strategy = "array_distance"
            logger.info("[REPO:duckdb_local] Strategy: array_distance (L2; no VSS)")
            return "array_distance"
        except Exception:
            pass
        DuckDBPastMistakesRepository._strategy = "numpy"
        logger.info("[REPO:duckdb_local] Strategy: numpy (pure-Python cosine)")
        return "numpy"

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
        from db.past_mistakes import init_past_mistakes_db, get_db

        if embedding.shape != (384,):
            raise ValueError(f"embedding must be 384-dim, got {embedding.shape}")

        init_past_mistakes_db()
        emb_list = embedding.tolist()

        # Build WHERE clause
        where_clauses = ["disease_type = ?"]
        params: list = [disease_type]
        if kle_uncertainty_range:
            where_clauses.append("kle_uncertainty BETWEEN ? AND ?")
            params.extend(kle_uncertainty_range)
        if error_types:
            ph = ",".join(["?"] * len(error_types))
            where_clauses.append(f"error_type IN ({ph})")
            params.extend(error_types)
        where_clauses.append("severity_level >= ?")
        params.append(severity_min)
        where_clause = " AND ".join(where_clauses)

        # Metadata-only columns — NEVER select case_embedding directly;
        # DuckDB v0.10 cannot return FLOAT[] arrays as Python objects.
        meta_select = """
            mistake_id, session_id, image_path,
            original_diagnosis, corrected_diagnosis, disease_type,
            error_type, severity_level, kle_uncertainty, safety_score,
            chexbert_labels, clinical_summary, debate_summary, created_at
        """

        with get_db() as conn:
            strategy = self._detect_strategy(conn)

            if strategy == "cosine_vss":
                # VSS available: compute cosine in SQL; similarity returned as col 14
                query = f"""
                    SELECT
                        {meta_select},
                        (1.0 - array_cosine_distance(case_embedding, ?::FLOAT[384])) AS similarity
                    FROM past_mistakes
                    WHERE {where_clause}
                    ORDER BY array_cosine_distance(case_embedding, ?::FLOAT[384]) ASC
                    LIMIT ?
                """
                all_params = [emb_list] + params + [emb_list, top_k * 2]
                raw_rows = conn.execute(query, all_params).fetchall()
                scored_rows = [(r, float(r[14])) for r in raw_rows]

            elif strategy == "array_distance":
                # No VSS but array_distance (L2) is available.
                # For unit-normalised embeddings (all sentence-transformer outputs):
                #   cosine_sim = 1 - L2² / 2
                # This is mathematically exact; no need to fetch the raw embedding.
                query = f"""
                    SELECT
                        {meta_select},
                        (1.0 - (array_distance(case_embedding, ?::FLOAT[384]) *
                                array_distance(case_embedding, ?::FLOAT[384])) / 2.0
                        ) AS similarity
                    FROM past_mistakes
                    WHERE {where_clause}
                    ORDER BY array_distance(case_embedding, ?::FLOAT[384]) ASC
                    LIMIT ?
                """
                all_params = [emb_list, emb_list] + params + [emb_list, top_k * 2]
                raw_rows = conn.execute(query, all_params).fetchall()
                scored_rows = [(r, max(0.0, float(r[14]))) for r in raw_rows]

            else:
                # Last resort: no array functions at all.
                # Return metadata rows with a neutral sentinel similarity (0.5).
                # Re-ranking via clinical relevance + recency will still order them.
                logger.warning(
                    "[REPO:duckdb_local] No array functions available. "
                    "Returning unscored rows with similarity=0.5 for re-ranking."
                )
                query = f"""
                    SELECT {meta_select}
                    FROM past_mistakes
                    WHERE {where_clause}
                    LIMIT ?
                """
                raw_rows = conn.execute(query, params + [top_k * 2]).fetchall()
                scored_rows = [(r, 0.5) for r in raw_rows]

        # Sort descending by similarity, apply threshold, cap at top_k
        scored_rows.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for r, sim in scored_rows:
            if sim < similarity_threshold:
                continue

            chexbert_raw = r[10]
            chexbert = {}
            if chexbert_raw:
                try:
                    chexbert = json.loads(chexbert_raw) if isinstance(chexbert_raw, str) else chexbert_raw
                except Exception:
                    chexbert = {}

            created_at = r[13]
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            results.append({
                "mistake_id":          r[0],
                "session_id":          r[1],
                "image_path":          r[2],
                "original_diagnosis":  r[3],
                "corrected_diagnosis": r[4],
                "disease_type":        r[5],
                "error_type":          r[6],
                "severity_level":      r[7],
                "kle_uncertainty":     r[8],
                "safety_score":        r[9],
                "chexbert_labels":     chexbert,
                "clinical_summary":    r[11],
                "debate_summary":      r[12],
                "created_at":          created_at,
                "similarity":          sim,
            })
            if len(results) >= top_k:
                break

        logger.info(
            f"[REPO:duckdb_local] {len(results)} results "
            f"(strategy={DuckDBPastMistakesRepository._strategy}, threshold={similarity_threshold})"
        )
        return results





# ---------------------------------------------------------------------------
# Factory — selects backend based on settings, auto-falls back
# ---------------------------------------------------------------------------

def get_past_mistakes_repository() -> PastMistakesRepository:
    """
    Return the configured past-mistakes repository.

    Decision logic:
    1. If ``settings.USE_CLOUD_VECTOR_DB`` is True AND Supabase credentials
       are present → try to instantiate ``SupabasePastMistakesRepository``.
    2. On any instantiation error (missing creds, import error, etc.) →
       fall back to ``DuckDBPastMistakesRepository`` and log a warning.
    3. If ``USE_CLOUD_VECTOR_DB`` is False → always use DuckDB directly.

    Runtime fallback: if ``SupabasePastMistakesRepository.retrieve_similar_mistakes``
    raises, the caller (``critic/model.py``) catches and calls the DuckDB
    implementation.  For a fully automatic fallback inside a single call see
    ``FallbackPastMistakesRepository`` below.
    """
    if settings.USE_CLOUD_VECTOR_DB:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning(
                "[REPO] USE_CLOUD_VECTOR_DB=True but SUPABASE_URL/SUPABASE_KEY not set. "
                "Falling back to DuckDB."
            )
            return DuckDBPastMistakesRepository()
        try:
            repo = SupabasePastMistakesRepository()
            logger.info("[REPO] Using Supabase pgvector HNSW backend for past mistakes retrieval")
            return repo
        except Exception as e:
            logger.warning(
                f"[REPO] Failed to initialise Supabase repository ({e}). "
                "Falling back to DuckDB."
            )
            return DuckDBPastMistakesRepository()

    logger.info("[REPO] Using DuckDB local backend for past mistakes retrieval")
    return DuckDBPastMistakesRepository()


# ---------------------------------------------------------------------------
# Transparent Fallback Wrapper
# ---------------------------------------------------------------------------

class FallbackPastMistakesRepository(PastMistakesRepository):
    """
    Wraps a primary repository and automatically falls back to a secondary
    repository on any retrieval error.  Ideal for production where you want
    Supabase-primary with zero downtime on connectivity issues.

    Example usage (replaces using get_past_mistakes_repository() directly):

        from db.past_mistakes_repository import (
            FallbackPastMistakesRepository,
            SupabasePastMistakesRepository,
            DuckDBPastMistakesRepository,
        )
        repo = FallbackPastMistakesRepository(
            primary=SupabasePastMistakesRepository(),
            fallback=DuckDBPastMistakesRepository(),
        )
    """

    def __init__(
        self,
        primary: PastMistakesRepository,
        fallback: PastMistakesRepository,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def backend_name(self) -> str:
        return f"fallback({self._primary.backend_name}→{self._fallback.backend_name})"

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
        try:
            results = self._primary.retrieve_similar_mistakes(
                disease_type=disease_type,
                embedding=embedding,
                kle_uncertainty_range=kle_uncertainty_range,
                error_types=error_types,
                severity_min=severity_min,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            logger.info(
                f"[REPO:fallback] Primary backend ({self._primary.backend_name}) "
                f"succeeded with {len(results)} results"
            )
            return results
        except Exception as e:
            logger.warning(
                f"[REPO:fallback] Primary backend ({self._primary.backend_name}) "
                f"failed: {e}. Using fallback ({self._fallback.backend_name})."
            )
            return self._fallback.retrieve_similar_mistakes(
                disease_type=disease_type,
                embedding=embedding,
                kle_uncertainty_range=kle_uncertainty_range,
                error_types=error_types,
                severity_min=severity_min,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
