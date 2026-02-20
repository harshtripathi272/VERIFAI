"""
Clean Supabase Seeder for past_mistakes
Ensures NO NULL values in filtering fields.
"""

import os
import uuid
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

# -----------------------------
# Setup
# -----------------------------

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Seed Data (ALL NON-NULL)
# -----------------------------

cases = [
    {
        "summary": "Early pneumonia misinterpreted as atelectasis due to overlapping radiographic appearance.",
        "disease": "pneumonia",
        "error_type": "misdiagnosis",
        "severity": 2,
        "kle": 0.58,
    },
    {
        "summary": "Small apical pneumothorax missed on supine AP chest radiograph.",
        "disease": "pneumothorax",
        "error_type": "missed_differential",
        "severity": 3,
        "kle": 0.62,
    },
    {
        "summary": "Subtle pleural effusion overlooked on portable AP film.",
        "disease": "pleural_effusion",
        "error_type": "missed_differential",
        "severity": 2,
        "kle": 0.49,
    },
    {
        "summary": "AP projection exaggerates cardiac silhouette leading to false cardiomegaly diagnosis.",
        "disease": "cardiomegaly",
        "error_type": "misdiagnosis",
        "severity": 1,
        "kle": 0.41,
    },
    {
        "summary": "Second subtle lesion missed due to satisfaction of search phenomenon.",
        "disease": "multiple_findings",
        "error_type": "missed_differential",
        "severity": 3,
        "kle": 0.67,
    },
]

# -----------------------------
# Batch Insert
# -----------------------------

def seed():
    print("\nSeeding clean data...\n")

    records = []

    for case in cases:
        embedding = sbert.encode(case["summary"])
        embedding = np.array(embedding, dtype=np.float32).tolist()

        records.append({
            "mistake_id": str(uuid.uuid4()),
            "session_id": "literature_seed",
            "image_path": "",  # not null, but irrelevant
            "created_at": datetime.utcnow().isoformat(),

            "original_diagnosis": "incorrect_initial_assessment",
            "corrected_diagnosis": "validated_true_finding",

            "disease_type": case["disease"],
            "error_type": case["error_type"],
            "severity_level": case["severity"],

            "kle_uncertainty": case["kle"],     # ✅ NOT NULL
            "safety_score": 0.5,                # ✅ NOT NULL (optional but clean)

            "chexbert_labels": "{}",
            "clinical_summary": case["summary"],
            "debate_summary": "Literature seed",

            "case_embedding": embedding
        })

    # Single request insert (stable)
    response = supabase.table("past_mistakes").insert(records).execute()

    print("Inserted rows:", len(response.data))
    print("\nDone.\n")


if __name__ == "__main__":
    seed()
