"""
FHIR Client (PhysioNet MIMIC-IV FHIR Demo)

REST client for retrieving patient context from FHIR R4 servers.
Chest-focused, hypothesis-gated FHIR retrieval for Historian agent.
Supports ImagingStudy → Patient → Condition/Observation workflow.
"""

import requests
from typing import Dict, List
from app.config import settings
from .hyp_code_map import CHEST_HYPOTHESIS_CODE_MAP, normalize_hypothesis


class FHIRClient:
    def __init__(self):
        self.base_url = settings.FHIR_BASE_URL.rstrip("/")
        self.headers = {"Accept": "application/fhir+json"}

    def _get(self, resource: str, params: Dict) -> List[Dict]:
        if settings.MOCK_MODELS:
            return []

        try:
            r = requests.get(
                f"{self.base_url}/{resource}",
                headers=self.headers,
                params=params,
                timeout=10
            )
            r.raise_for_status()
            data = r.json()
            return [e["resource"] for e in data.get("entry", [])]
        except Exception as e:
            print(f"[FHIR] {resource} failed: {e}")
            return []

    def fetch_evidence_for_hypothesis(
        self, patient_id: str, hypothesis: str
    ) -> Dict:

        h = normalize_hypothesis(hypothesis)
        plan = CHEST_HYPOTHESIS_CODE_MAP.get(h)

        if not plan:
            return {"conditions": [], "observations": [], "medications": []}

        conditions = self._get(
            "Condition",
            {
                "patient": patient_id,
                "clinical-status": "active",
                **({"code": ",".join(plan["conditions"])} if plan["conditions"] else {})
            }
        )

        observations = self._get(
            "Observation",
            {
                "patient": patient_id,
                "category": "laboratory",
                "_sort": "-date",
                "_count": "10",
                **({"code": ",".join(plan["labs"])} if plan["labs"] else {})
            }
        )

        medications = self._get(
            "MedicationRequest",
            {"patient": patient_id, "status": "active"}
        )

        return {
            "conditions": conditions,
            "observations": observations,
            "medications": medications
        }


fhir_client = FHIRClient()

