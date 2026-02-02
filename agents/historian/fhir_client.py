"""
FHIR Client (PhysioNet MIMIC-IV FHIR Demo)

REST client for retrieving patient context from FHIR R4 servers.
Chest-focused, hypothesis-gated FHIR retrieval for Historian agent.
Supports ImagingStudy → Patient → Condition/Observation workflow.
"""

import duckdb
import json
import base64
from typing import Dict
from .hyp_code_map import CHEST_HYPOTHESIS_CODE_MAP, normalize_hypothesis

class FHIRClient:
    def __init__(self, db_path="../../verifai_fhir.duckdb"):
        self.con = duckdb.connect(db_path)

    
    # PUBLIC API (MODIFIED)
    def fetch_evidence_for_hypothesis(
        self, patient_id: str, hypothesis: str
    ) -> Dict:

        hypothesis = normalize_hypothesis(hypothesis)
        plan = CHEST_HYPOTHESIS_CODE_MAP.get(hypothesis)

        if not plan:
            return self._empty_evidence()

        # Try structured evidence first
        structured = self._fetch_structured(patient_id, plan)

        if self._has_structured_signal(structured):
            structured["source"] = "structured"
            return structured

        # allback to document-based evidence
        documents = self._fetch_documents(patient_id)

        return {
            **self._empty_evidence(),
            "documents": documents,
            "source": "documents"
        }
    # STRUCTURED PATH

    def _fetch_structured(self, patient_id: str, plan: Dict) -> Dict:
        return {
            "conditions": self._query_conditions(patient_id, plan["conditions"]),
            "observations": self._query_observations(patient_id, plan["labs"]),
            "medications": self._query_medications(patient_id),
        }

    def _has_structured_signal(self, evidence: Dict) -> bool:
        return (
            len(evidence["conditions"]) > 0
            or len(evidence["observations"]) > 0
            or len(evidence["medications"]) > 0
        )
    # DOCUMENT FALLBACK
    def _fetch_documents(self, patient_id: str):
        rows = self.con.execute("""
            SELECT json
            FROM fhir
            WHERE resourceType IN ('DiagnosticReport', 'DocumentReference')
              AND patient_id = ?
        """, [patient_id]).fetchall()

        docs = []
        for (raw,) in rows:
            r = json.loads(raw)
            text = self._extract_document_text(r)
            if text:
                docs.append({
                    "resourceType": r["resourceType"],
                    "id": r["id"],
                    "text": text
                })
        return docs

    def _extract_document_text(self, resource: dict) -> str | None:
        # DiagnosticReport.presentedForm[].data (base64)
        if resource["resourceType"] == "DiagnosticReport":
            for form in resource.get("presentedForm", []):
                if "data" in form:
                    return self._decode_base64(form["data"])

        # DocumentReference.content[].attachment.data
        if resource["resourceType"] == "DocumentReference":
            for c in resource.get("content", []):
                att = c.get("attachment", {})
                if "data" in att:
                    return self._decode_base64(att["data"])

        return None

    def _decode_base64(self, data: str) -> str:
        try:
            return base64.b64decode(data).decode("utf-8", errors="ignore")
        except Exception:
            return None

    # STRUCTURED QUERIES

    def _query_conditions(self, patient_id, codes):
        if not codes:
            return []

        return self._query_by_codes(
            "Condition", patient_id, codes, "$.code.coding"
        )

    def _query_observations(self, patient_id, codes):
        if not codes:
            return []

        return self._query_by_codes(
            "Observation", patient_id, codes, "$.code.coding"
        )

    def _query_medications(self, patient_id):
        rows = self.con.execute("""
            SELECT json
            FROM fhir
            WHERE resourceType = 'MedicationRequest'
              AND patient_id = ?
        """, [patient_id]).fetchall()

        return [json.loads(r[0]) for r in rows]

    def _query_by_codes(self, rtype, patient_id, codes, coding_path):
        rows = self.con.execute(f"""
            SELECT json
            FROM fhir
            WHERE resourceType = '{rtype}'
              AND patient_id = ?
        """, [patient_id]).fetchall()

        matches = []
        for (raw,) in rows:
            r = json.loads(raw)
            for coding in r.get("code", {}).get("coding", []):
                if coding.get("code") in codes:
                    matches.append(r)
                    break

        return matches
    # UTILS

    def _empty_evidence(self):
        return {
            "conditions": [],
            "observations": [],
            "medications": [],
            "documents": []
        }


