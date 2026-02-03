"""
FHIR Client (PhysioNet MIMIC-IV FHIR Demo)

REST client for retrieving patient context from FHIR R4 servers.
Chest-focused, hypothesis-gated FHIR retrieval for Historian agent.
Supports ImagingStudy → Patient → Condition/Observation workflow.
"""

import duckdb
import json
import base64
import os
from typing import Dict
from .hyp_code_map import CHEST_HYPOTHESIS_CODE_MAP, normalize_hypothesis

class FHIRClient:
    def __init__(self, db_path=None):
        if db_path is None:
            # Resolve verifai_fhir.duckdb in the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Agents/Historian -> Agents -> Root
            root_dir = os.path.dirname(os.path.dirname(current_dir))
            db_path = os.path.join(root_dir, "verifai_fhir.duckdb")
            
        self.con = duckdb.connect(db_path)

    
    # PUBLIC API (MODIFIED)
    def fetch_evidence_for_hypothesis(
        self, patient_id: str, hypothesis: str
    ) -> Dict:
        hypothesis_key = normalize_hypothesis(hypothesis)
        plan = CHEST_HYPOTHESIS_CODE_MAP.get(hypothesis_key)

        # We fetch everything relevant to provide full context
        evidence = self._empty_evidence()
        
        # 1. Fetch Structured Data (Conditions, Labs, Meds)
        if plan:
            evidence["conditions"] = self._query_conditions(patient_id, plan["conditions"])
            evidence["observations"] = self._query_observations(patient_id, plan["labs"])
        
        evidence["medications"] = self._query_medications(patient_id)
        
        # 2. Fetch Additional Context (Procedures, Allergies, Encounters)
        evidence["procedures"] = self._query_procedures(patient_id)
        evidence["allergies"] = self._query_allergies(patient_id)
        evidence["encounters"] = self._query_encounters(patient_id)

        # 3. Fetch Document-based evidence (Radiology reports, etc.)
        evidence["documents"] = self._fetch_documents(patient_id)

        evidence["source"] = "hybrid"
        return evidence
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
                    "text": text,
                    "category": r.get("category", [{}])[0].get("coding", [{}])[0].get("display", "Clinical Note")
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
                # Sometimes it might be in content.attachment.url or title if not data
                if "title" in att:
                    return att["title"]

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
        return self._query_by_codes("Condition", patient_id, codes)

    def _query_observations(self, patient_id, codes):
        if not codes:
            return []
        return self._query_by_codes("Observation", patient_id, codes)

    def _query_medications(self, patient_id):
        return self._query_all_by_rtype("MedicationRequest", patient_id)

    def _query_procedures(self, patient_id):
        return self._query_all_by_rtype("Procedure", patient_id)

    def _query_allergies(self, patient_id):
        return self._query_all_by_rtype("AllergyIntolerance", patient_id)

    def _query_encounters(self, patient_id):
        return self._query_all_by_rtype("Encounter", patient_id)

    def _query_all_by_rtype(self, rtype, patient_id):
        rows = self.con.execute(f"""
            SELECT json
            FROM fhir
            WHERE resourceType = '{rtype}'
              AND patient_id = ?
        """, [patient_id]).fetchall()
        return [json.loads(r[0]) for r in rows]

    def _query_by_codes(self, rtype, patient_id, codes):
        rows = self.con.execute(f"""
            SELECT json
            FROM fhir
            WHERE resourceType = '{rtype}'
              AND patient_id = ?
        """, [patient_id]).fetchall()

        matches = []
        for (raw,) in rows:
            r = json.loads(raw)
            # Handle list of codings
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
            "procedures": [],
            "allergies": [],
            "encounters": [],
            "documents": []
        }

fhir_client = FHIRClient()
