"""
FHIR Client (PhysioNet MIMIC-IV FHIR Demo)

REST client for retrieving patient context from FHIR R4 servers.
Chest-focused, hypothesis-gated FHIR retrieval for Historian agent.
Supports ImagingStudy -> Patient -> Condition/Observation workflow.
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
        self._patient_cache = {}

        # Pre-calculate all relevant codes for SQL filtering
        self.all_condition_codes = set()
        self.all_lab_codes = set()
        for plan in CHEST_HYPOTHESIS_CODE_MAP.values():
            self.all_condition_codes.update(plan.get("conditions", []))
            self.all_lab_codes.update(plan.get("labs", []))

    
    # PUBLIC API (MODIFIED)
    def fetch_evidence_for_hypothesis(
        self, patient_id: str, hypothesis: str
    ) -> Dict:
        """
        Fetches evidence for a specific hypothesis using cached full patient context.
        """
        # 1. Fetch full context (cached if available)
        full_context = self.fetch_full_patient_context(patient_id)
        
        # 2. Filter in-memory
        return self.filter_patient_context(full_context, hypothesis)

    def fetch_full_patient_context(self, patient_id: str) -> Dict:
        """
        Fetches all relevant FHIR resources for a patient in one go.
        Uses caching to avoid redundant DB hits.
        Applies temporal filtering (last 5 years) and code filtering via SQL.
        """
        if patient_id in self._patient_cache:
            return self._patient_cache[patient_id]

        context = {
            "conditions": self._query_conditions_sql(patient_id),
            "observations": self._query_observations_sql(patient_id),
            "medications": self._query_all_by_rtype_temporal("MedicationRequest", patient_id),
            "procedures": self._query_all_by_rtype_temporal("Procedure", patient_id),
            "allergies": self._query_all_by_rtype_temporal("AllergyIntolerance", patient_id),
            "encounters": self._query_all_by_rtype_temporal("Encounter", patient_id),
            "documents": self._fetch_documents_temporal(patient_id),
            "source": "hybrid_cached"
        }

        self._patient_cache[patient_id] = context
        return context

    def filter_patient_context(self, context: Dict, hypothesis: str) -> Dict:
        """
        Filters the full patient context for a specific hypothesis.
        """
        hypothesis_key = normalize_hypothesis(hypothesis)
        plan = CHEST_HYPOTHESIS_CODE_MAP.get(hypothesis_key, {})
        
        # Start with a shallow copy of context lists to avoid mutating cache
        evidence = {
            "conditions": [],
            "observations": [],
            "medications": context["medications"], # Pass through
            "procedures": context["procedures"],   # Pass through
            "allergies": context["allergies"],     # Pass through
            "encounters": context["encounters"],   # Pass through
            "documents": context["documents"],     # Pass through
            "source": context.get("source", "hybrid")
        }

        # Filter Conditions
        target_conditions = set(plan.get("conditions", []))
        if target_conditions:
            for cond in context["conditions"]:
                # Check all codings in the resource
                for coding in cond.get("code", {}).get("coding", []):
                    if coding.get("code") in target_conditions:
                        evidence["conditions"].append(cond)
                        break

        # Filter Observations
        target_labs = set(plan.get("labs", []))
        if target_labs:
            for obs in context["observations"]:
                 for coding in obs.get("code", {}).get("coding", []):
                    if coding.get("code") in target_labs:
                        evidence["observations"].append(obs)
                        break
        
        return evidence

    # STRUCTURED PATH
    # Legacy helper _fetch_structured is removed as it's superseded by fetch_full_patient_context logic
    
    def _has_structured_signal(self, evidence: Dict) -> bool:
        return (
            len(evidence["conditions"]) > 0
            or len(evidence["observations"]) > 0
            or len(evidence["medications"]) > 0
        )

    # DOCUMENT FALLBACK
    def _fetch_documents_temporal(self, patient_id: str):
        # Adding temporal filter
        rows = self.con.execute("""
            SELECT json
            FROM fhir
            WHERE resourceType IN ('DiagnosticReport', 'DocumentReference')
              AND patient_id = ?
              AND (event_time IS NULL OR event_time >= NOW() - INTERVAL '5 years')
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

    # STRUCTURED QUERIES (OPTIMIZED)

    def _query_conditions_sql(self, patient_id):
        if not self.all_condition_codes:
            return []
        
        # Convert set to tuple for SQL IN clause
        codes_tuple = tuple(self.all_condition_codes)
        if len(codes_tuple) == 1:
            code_str = f"('{codes_tuple[0]}')"
        else:
            code_str = str(codes_tuple)

        # SQL Filter: event_time >= 5 years ago OR NULL
        # Code Filter: json_extract code
        query = f"""
            SELECT json
            FROM fhir
            WHERE resourceType = 'Condition'
              AND patient_id = ?
              AND (event_time IS NULL OR event_time >= NOW() - INTERVAL '5 years')
              AND json_extract(json, '$.code.coding[0].code') IN {code_str}
        """
        rows = self.con.execute(query, [patient_id]).fetchall()
        return [json.loads(r[0]) for r in rows]

    def _query_observations_sql(self, patient_id):
        if not self.all_lab_codes:
            return []
            
        codes_tuple = tuple(self.all_lab_codes)
        if len(codes_tuple) == 1:
            code_str = f"('{codes_tuple[0]}')"
        else:
            code_str = str(codes_tuple)

        query = f"""
            SELECT json
            FROM fhir
            WHERE resourceType = 'Observation'
              AND patient_id = ?
              AND (event_time IS NULL OR event_time >= NOW() - INTERVAL '5 years')
              AND json_extract(json, '$.code.coding[0].code') IN {code_str}
        """
        rows = self.con.execute(query, [patient_id]).fetchall()
        return [json.loads(r[0]) for r in rows]

    def _query_all_by_rtype_temporal(self, rtype, patient_id):
        rows = self.con.execute(f"""
            SELECT json
            FROM fhir
            WHERE resourceType = '{rtype}'
              AND patient_id = ?
              AND (event_time IS NULL OR event_time >= NOW() - INTERVAL '5 years')
        """, [patient_id]).fetchall()
        return [json.loads(r[0]) for r in rows]

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
