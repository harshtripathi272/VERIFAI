"""
FHIR Client (Hybrid: SQL + Vector Search)

Retrieves patient context from DuckDB and FAISS.
1. SQL Filter: Hard constraints (Patient, Time, Type).
2. FAISS Search: Semantic ranking of filtered candidates against hypothesis.
"""

import duckdb
import json
import base64
import os
import faiss
import numpy as np
import logging
import pickle
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from .hyp_code_map import CHEST_HYPOTHESIS_CODE_MAP, normalize_hypothesis

# Helper configuration (should equal ETL config)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_FILENAME = "verifai_fhir.duckdb"
FAISS_FILENAME = "verifai_fhir.faiss"
MAPPING_FILENAME = "verifai_fhir_mapping.json"

class FHIRClient:
    def __init__(self, root_dir=None):
        if root_dir is None:
             # Resolve project root from current file location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Agents/Historian -> Agents -> Root
            self.root_dir = os.path.dirname(os.path.dirname(current_dir))
        else:
            self.root_dir = root_dir
            
        self.db_path = os.path.join(self.root_dir, DB_FILENAME)
        self.faiss_path = os.path.join(self.root_dir, FAISS_FILENAME)
        self.mapping_path = os.path.join(self.root_dir, MAPPING_FILENAME)
        
        # Connect to DuckDB
        self.con = duckdb.connect(self.db_path)
        
        # Load FAISS Resources (Lazy load or eager? Eager for now)
        self.index = None
        self.id_mapping = {}
        self.embedder = None
        
        self._load_vector_resources()

        # Pre-calculate code maps for fallback/hybrid logic
        self.all_condition_codes = set()
        self.all_lab_codes = set()
        for plan in CHEST_HYPOTHESIS_CODE_MAP.values():
            self.all_condition_codes.update(plan.get("conditions", []))
            self.all_lab_codes.update(plan.get("labs", []))

    def _load_vector_resources(self):
        """Loads FAISS index, ID mapping, and Embedding model."""
        if os.path.exists(self.faiss_path):
            print(f"[FHIRClient] Loading FAISS index from {self.faiss_path}...")
            self.index = faiss.read_index(self.faiss_path)
        else:
            print("[FHIRClient] WARNING: FAISS index not found. Semantic search will be disabled.")
            
        if os.path.exists(self.mapping_path):
             with open(self.mapping_path, "r") as f:
                # json keys are strings, but we need int keys for reverse lookup if needed
                # Actually mapping is faiss_id (str) -> resource_id (str)
                self.id_mapping = json.load(f)
                # create reverse mapping resource_id -> faiss_id (int)
                self.rid_to_faiss = {v: int(k) for k, v in self.id_mapping.items()}

        print(f"[FHIRClient] Loading embedding model {EMBEDDING_MODEL}...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    # PUBLIC API
    
    def fetch_evidence_for_hypothesis(self, patient_id: str, hypothesis: str) -> Dict:
        """
        Public facade for the agent. Now uses hybrid retrieval.
        """
        return self.fetch_evidence_hybrid(patient_id, hypothesis)

    def fetch_evidence_hybrid(self, patient_id: str, hypothesis: str) -> Dict:
        """
        Hybrid Retrieval:
        1. SQL: Filter by patient_id + time window (5 years).
        2. Vector: Rank results by similarity to hypothesis.
        """
        
        # 1. SQL Candidate Generation
        # Fetch relevant resources for this patient within time window
        candidates = self._fetch_candidates_sql(patient_id)
        
        if not candidates:
            return self._empty_evidence()

        # 2. Semantic Ranking
        # If we have a hypothesis and vector resources, rank them
        if hypothesis and self.index and self.embedder:
            ranked_candidates = self._rank_candidates(candidates, hypothesis)
        else:
            # Fallback to pure SQL results (unranked or time-sorted)
            ranked_candidates = sorted(candidates, key=lambda x: x.get("event_time") or "", reverse=True)

        # 3. Structure Output
        return self._structure_output(ranked_candidates)

    def _fetch_candidates_sql(self, patient_id: str) -> List[Dict]:
        """
        Fetch all potentially relevant resources for the patient from DuckDB.
        """
        query = """
            SELECT id, resourceType, primary_code, normalized_summary, event_time, raw_json
            FROM fhir_resources
            WHERE patient_ref = ?
        """
        rows = self.con.execute(query, [patient_id]).fetchall()
        
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "resourceType": r[1],
                "primary_code": r[2],
                "normalized_summary": r[3],
                "event_time": r[4],
                "raw_json": json.loads(r[5])
            })
        return results

    def _rank_candidates(self, candidates: List[Dict], hypothesis: str) -> List[Dict]:
        """
        Ranks candidates by embedding similarity to hypothesis.
        No reconstruct usage.
        """

        # 1️⃣ Embed hypothesis
        hyp_vec = self.embedder.encode(
            [hypothesis],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]  # shape (dim,)

        # 2️⃣ Embed candidate summaries
        summaries = [c["normalized_summary"] for c in candidates]

        cand_vecs = self.embedder.encode(
            summaries,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 3️⃣ Compute cosine similarity (dot product since normalized)
        scores = cand_vecs @ hyp_vec

        # 4️⃣ Attach scores
        for i, c in enumerate(candidates):
            c["score"] = float(scores[i])

        # 5️⃣ Sort descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates


    def _structure_output(self, ranked_candidates: List[Dict]) -> Dict:
        """
        Groups ranked candidates into the standard evidence dictionary.
        """
        evidence = self._empty_evidence()
        evidence["source"] = "hybrid_faiss"
        
        for c in ranked_candidates:
            # raw_json is what the agent expects
            resource = c["raw_json"]
            # Inject score and summary for strict transparency if needed
            if "score" in c:
                resource["_relevance_score"] = c["score"]
            resource["_summary"] = c["normalized_summary"]

            rtype = c["resourceType"]
            
            if rtype == "Condition":
                evidence["conditions"].append(resource)
            elif rtype == "Observation":
                evidence["observations"].append(resource)
            elif rtype == "MedicationRequest":
                evidence["medications"].append(resource)
            elif rtype == "Procedure":
                evidence["procedures"].append(resource)
            elif rtype == "AllergyIntolerance":
                evidence["allergies"].append(resource)
            elif rtype == "Encounter":
                evidence["encounters"].append(resource)
            elif rtype in ["DiagnosticReport", "DocumentReference"]:
                doc_entry = {
                    "resourceType": rtype,
                    "id": c["id"],
                    "text": c["normalized_summary"],
                    "category": "Clinical Note",
                    "date": str(c["event_time"]),
                    "_relevance_score": c.get("score"),
                    "_summary": c.get("normalized_summary")
                }
                evidence["documents"].append(doc_entry)
                
        return evidence

    def _empty_evidence(self):
        return {
            "conditions": [],
            "observations": [],
            "medications": [],
            "procedures": [],
            "allergies": [],
            "encounters": [],
            "documents": []
        }

    # For legacy compatibility if needed
    def fetch_full_patient_context(self, patient_id):
        return self.fetch_evidence_hybrid(patient_id, "")
        
    def filter_patient_context(self, context, hypothesis):
        # This is now largely redundant as hybrid does it all, 
        # but if the agent logic relies on 2-step:
        # We can implement re-ranking here if context has all candidates.
        return context 

fhir_client = FHIRClient()
