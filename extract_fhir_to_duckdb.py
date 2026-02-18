import json
from pathlib import Path
import duckdb


# ==========================
# CONFIG
# ==========================

FHIR_DIR = Path("output/fhir")
DB_PATH = "verifai_fhir.duckdb"

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

def normalize_reference(ref: str | None) -> str | None:
    if not ref:
        return None
    if ref.startswith("urn:uuid:"):
        return ref.replace("urn:uuid:", "Patient/")
    return ref


def extract_patient_id(resource: dict) -> str | None:
    if resource["resourceType"] == "Patient":
        return resource.get("id")

    for key in ["subject", "patient"]:
        participant = resource.get(key)
        if participant and "reference" in participant:
            ref = normalize_reference(participant["reference"])
            if ref and ref.startswith("Patient/"):
                return ref.split("/")[-1]

    return None


def extract_encounter_id(resource: dict) -> str | None:
    enc = resource.get("encounter")
    if enc and "reference" in enc:
        ref = normalize_reference(enc["reference"])
        if ref and ref.startswith("Encounter/"):
            return ref.split("/")[-1]
    return None


def extract_event_time(resource: dict) -> str | None:
    """
    Extract best available timestamp for timeline sorting.
    Priority order is important.
    """

    # Most clinical events
    if "effectiveDateTime" in resource:
        return resource.get("effectiveDateTime")

    if "recordedDate" in resource:
        return resource.get("recordedDate")

    if "onsetDateTime" in resource:
        return resource.get("onsetDateTime")

    if "authoredOn" in resource:
        return resource.get("authoredOn")

    if "issued" in resource:
        return resource.get("issued")

    if "period" in resource and "start" in resource["period"]:
        return resource["period"].get("start")

    return None


# ==========================
# MAIN EXTRACTION
# ==========================

def main():
    con = duckdb.connect(DB_PATH)

    # Create historian-ready schema
    con.execute("""
        CREATE TABLE IF NOT EXISTS fhir (
            resourceType TEXT,
            id TEXT,
            patient_id TEXT,
            encounter_id TEXT,
            event_time TIMESTAMP,
            json TEXT
        )
    """)

    # Clean previous run
    con.execute("DELETE FROM fhir")

    total = 0

    for bundle_path in sorted(FHIR_DIR.glob("*.json")):
        print(f"📦 Processing {bundle_path.name}")

        with open(bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        for entry in bundle.get("entry", []):
            resource = entry.get("resource")
            if not resource:
                continue

            rtype = resource.get("resourceType")

            if rtype not in RELEVANT_RESOURCES:
                continue

            rid = resource.get("id")
            patient_id = extract_patient_id(resource)
            encounter_id = extract_encounter_id(resource)
            event_time = extract_event_time(resource)

            # Normalize references
            for ref_key in ["subject", "patient", "encounter"]:
                if ref_key in resource and "reference" in resource[ref_key]:
                    resource[ref_key]["reference"] = normalize_reference(
                        resource[ref_key]["reference"]
                    )

            con.execute(
                """
                INSERT INTO fhir
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rtype,
                    rid,
                    patient_id,
                    encounter_id,
                    event_time,
                    json.dumps(resource)
                )
            )

            total += 1

    # Create indexes for fast historian queries
    con.execute("CREATE INDEX IF NOT EXISTS idx_patient ON fhir(patient_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_time ON fhir(event_time)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_type ON fhir(resourceType)")

    con.close()

    print(f"\n✅ Done. Stored {total} FHIR resources in {DB_PATH}")
    print("Historian-ready database created.")


if __name__ == "__main__":
    main()
