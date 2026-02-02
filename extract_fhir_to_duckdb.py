import json
from pathlib import Path
import duckdb

# -----------------------
# CONFIG
# -----------------------

FHIR_DIR = Path("output/fhir")
DB_PATH = "verifai_fhir.duckdb"

RELEVANT_RESOURCES = {
    "Patient",
    "Condition",
    "Observation",
    "MedicationRequest"
}

# -----------------------
# HELPERS
# -----------------------

def normalize_reference(ref: str) -> str | None:
    if not ref:
        return None
    if ref.startswith("urn:uuid:"):
        return ref.replace("urn:uuid:", "Patient/")
    return ref


def extract_patient_id(resource: dict) -> str | None:
    if resource["resourceType"] == "Patient":
        return resource.get("id")

    subject = resource.get("subject") or resource.get("patient")
    if subject and "reference" in subject:
        ref = normalize_reference(subject["reference"])
        if ref and ref.startswith("Patient/"):
            return ref.split("/")[-1]

    return None


# -----------------------
# MAIN EXTRACTION
# -----------------------

def main():
    con = duckdb.connect(DB_PATH)

    con.execute("""
        CREATE TABLE IF NOT EXISTS fhir (
            resourceType TEXT,
            id TEXT,
            patient_id TEXT,
            json TEXT
        )
    """)

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

            # Normalize references in-place
            if "subject" in resource and "reference" in resource["subject"]:
                resource["subject"]["reference"] = normalize_reference(
                    resource["subject"]["reference"]
                )

            con.execute(
                "INSERT INTO fhir VALUES (?, ?, ?, ?)",
                (
                    rtype,
                    rid,
                    patient_id,
                    json.dumps(resource)
                )
            )

            total += 1

    print(f"\n✅ Done. Stored {total} FHIR resources in {DB_PATH}")


if __name__ == "__main__":
    main()
