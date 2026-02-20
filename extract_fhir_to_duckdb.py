import json
import duckdb
import base64
import os
import faiss
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ==========================
# CONFIG
# ==========================

FHIR_DIR = Path("output/fhir")
DB_PATH = "verifai_fhir.duckdb"
FAISS_INDEX_PATH = "verifai_fhir.faiss"
ID_MAPPING_PATH = "verifai_fhir_mapping.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RELEVANT_RESOURCES = {
    "Patient",
    "Condition",
    "Observation",
    "MedicationRequest",
    "DiagnosticReport",
    "DocumentReference",
    "Procedure",
    "AllergyIntolerance",
    "Encounter"
}

# ==========================
# HELPERS
# ==========================

def safe_timestamp(ts: str | None):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def normalize_reference(ref: str | None):
    if not ref:
        return None
    if ref.startswith("urn:uuid:"):
        return ref.replace("urn:uuid:", "Patient/")
    return ref

def extract_patient_id(resource: dict):
    if resource["resourceType"] == "Patient":
        return resource.get("id")

    for key in ["subject", "patient"]:
        participant = resource.get(key)
        if participant and "reference" in participant:
            ref = normalize_reference(participant["reference"])
            if ref and ref.startswith("Patient/"):
                return ref.split("/")[-1]
    return None

def extract_encounter_id(resource: dict):
    enc = resource.get("encounter")
    if enc and "reference" in enc:
        ref = normalize_reference(enc["reference"])
        if ref and ref.startswith("Encounter/"):
            return ref.split("/")[-1]
    return None

def extract_event_time(resource: dict):
    for key in [
        "effectiveDateTime",
        "recordedDate",
        "onsetDateTime",
        "authoredOn",
        "issued",
        "date",
        "occurrenceDateTime"
    ]:
        if key in resource:
            return resource.get(key)

    if "period" in resource and "start" in resource["period"]:
        return resource["period"]["start"]

    return None

def get_primary_code(resource: dict):
    code_obj = resource.get("code")
    if not code_obj and "medicationCodeableConcept" in resource:
        code_obj = resource.get("medicationCodeableConcept")

    if code_obj and "coding" in code_obj:
        coding = code_obj["coding"][0]
        return coding.get("code"), coding.get("display")

    return None, None

# ==========================
# NORMALIZATION
# ==========================

def normalize_resource(resource: dict):
    rtype = resource.get("resourceType")
    rid = resource.get("id")
    patient_id = extract_patient_id(resource)
    encounter_id = extract_encounter_id(resource)
    event_time_raw = extract_event_time(resource)
    event_time = safe_timestamp(event_time_raw)

    primary_code, display = get_primary_code(resource)

    summary_parts = []

    if rtype == "Condition":
        summary_parts.append(f"Condition: {display}")

    elif rtype == "DiagnosticReport":
        summary_parts.append(f"Report: {display}")
        if resource.get("conclusion"):
            summary_parts.append(f"Conclusion: {resource['conclusion']}")

    elif rtype == "Observation":
        summary_parts.append(f"Observation: {display}")

    elif rtype == "Encounter":
        summary_parts.append("Encounter")

    if not summary_parts:
        summary_parts.append(f"{rtype}: {display}")

    return {
        "id": rid,
        "resourceType": rtype,
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "event_time": event_time,
        "primary_code": primary_code,
        "normalized_summary": " ".join(summary_parts),
        "raw_json": json.dumps(resource)
    }

# ==========================
# MAIN
# ==========================

def main():
    logger.info("Starting FHIR extraction...")

    con = duckdb.connect(DB_PATH)

    con.execute("""
        CREATE TABLE IF NOT EXISTS fhir_resources (
            id TEXT PRIMARY KEY,
            patient_ref TEXT,
            resourceType TEXT,
            primary_code TEXT,
            event_time TIMESTAMP,
            encounter_ref TEXT,
            normalized_summary TEXT,
            raw_json JSON
        )
    """)

    con.execute("DELETE FROM fhir_resources")

    id_mapping = {}
    all_embeddings = []
    current_faiss_id = 0

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(embedding_dim)

    files = sorted(FHIR_DIR.glob("*.json"))

    total_resources = 0

    for bundle_path in files:
        logger.info(f"Processing {bundle_path.name}")

        with open(bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        for entry in bundle.get("entry", []):
            resource = entry.get("resource")
            if not resource:
                continue

            if resource.get("resourceType") not in RELEVANT_RESOURCES:
                continue

            norm = normalize_resource(resource)

            if resource["resourceType"] != "Patient" and not norm["patient_id"]:
                continue

            embedding = embedder.encode(
                norm["normalized_summary"],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            all_embeddings.append(embedding.astype("float32"))
            id_mapping[str(current_faiss_id)] = norm["id"]

            con.execute("""
                INSERT INTO fhir_resources VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                norm["id"],
                norm["patient_id"],
                norm["resourceType"],
                norm["primary_code"],
                norm["event_time"],
                norm["encounter_id"],
                norm["normalized_summary"],
                norm["raw_json"]
            ))

            current_faiss_id += 1
            total_resources += 1

    if all_embeddings:
        index.add(np.array(all_embeddings))
        faiss.write_index(index, FAISS_INDEX_PATH)

        with open(ID_MAPPING_PATH, "w") as f:
            json.dump(id_mapping, f)

        logger.info(f"Indexed {index.ntotal} vectors.")

    con.execute("CREATE INDEX IF NOT EXISTS idx_patient ON fhir_resources(patient_ref)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_type ON fhir_resources(resourceType)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_code ON fhir_resources(primary_code)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_time ON fhir_resources(event_time)")

    con.close()

    logger.info(f"Finished. Total resources indexed: {total_resources}")

if __name__ == "__main__":
    main()
