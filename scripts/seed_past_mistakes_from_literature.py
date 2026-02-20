#!/usr/bin/env python3

"""
Seed Past Mistakes DB with peer-reviewed radiology error archetypes.

Sources:
- Pinto et al., Radiographics 2013
- Bruno et al., Radiographics 2015
- Waite et al., JACR 2017
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from db.past_mistakes import insert_validated_mistake

sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

literature_cases = [

    {
        "summary": (
            "Small apical pneumothorax missed on supine AP chest radiograph. "
            "Subtle pleural line overlooked."
        ),
        "disease": "pneumothorax",
        "error_type": "missed_differential",
        "severity": 3,
        "source": "Pinto et al., Radiographics 2013"
    },

    {
        "summary": (
            "Early pneumonia misinterpreted as atelectasis due to overlapping "
            "radiographic appearance in lower lobes."
        ),
        "disease": "pneumonia",
        "error_type": "misdiagnosis",
        "severity": 2,
        "source": "Bruno et al., Radiographics 2015"
    },

    {
        "summary": (
            "Small pleural effusion overlooked on portable AP film with minimal "
            "costophrenic angle blunting."
        ),
        "disease": "pleural_effusion",
        "error_type": "missed_differential",
        "severity": 2,
        "source": "Waite et al., JACR 2017"
    },

    {
        "summary": (
            "AP projection exaggerates cardiac silhouette leading to false "
            "positive cardiomegaly diagnosis."
        ),
        "disease": "cardiomegaly",
        "error_type": "misdiagnosis",
        "severity": 1,
        "source": "Bruno et al., Radiographics 2015"
    },

    {
        "summary": (
            "Second subtle lesion missed after identifying major abnormality "
            "due to satisfaction of search phenomenon."
        ),
        "disease": "multiple_findings",
        "error_type": "missed_differential",
        "severity": 3,
        "source": "Pinto et al., Radiographics 2013"
    },
]

print("Seeding literature-based mistakes...\n")

for case in literature_cases:

    embedding = sbert.encode(case["summary"])
    embedding = np.array(embedding, dtype=np.float32)

    insert_validated_mistake(
        session_id="literature_seed",
        image_path=None,
        original_diagnosis="incorrect_initial_assessment",
        corrected_diagnosis="validated_true_finding",
        disease_type=case["disease"],
        error_type=case["error_type"],
        severity_level=case["severity"],
        case_embedding=embedding,
        clinical_summary=case["summary"],
        debate_summary=f"Literature Source: {case['source']}"
    )

    print(f"Inserted: {case['disease']} ({case['error_type']})")

print("\nDone.")
