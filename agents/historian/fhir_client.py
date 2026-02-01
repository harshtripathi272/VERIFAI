"""
FHIR Client

REST client for retrieving patient context from FHIR R4 servers.
Supports ImagingStudy → Patient → Condition/Observation workflow.
"""

import requests
from typing import Any
from app.config import settings
from graph.state import HistorianFact


class FHIRClient:
    """
    FHIR REST client for patient context retrieval.
    
    Implements MCP-style tool interface for consistency with
    wso2/fhir-mcp-server patterns.
    """
    
    def __init__(self):
        self.base_url = settings.FHIR_BASE_URL.rstrip("/")
        self.headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json"
        }
        if settings.FHIR_AUTH_TOKEN:
            self.headers["Authorization"] = f"Bearer {settings.FHIR_AUTH_TOKEN}"
    
    def _request(self, endpoint: str, params: dict | None = None) -> dict | None:
        """Make FHIR REST request."""
        if settings.MOCK_MODELS:
            return None
            
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[FHIR] Request failed: {e}")
            return None
    
    def get_patient(self, patient_id: str) -> dict | None:
        """Retrieve Patient resource."""
        return self._request(f"Patient/{patient_id}")
    
    def get_conditions(self, patient_id: str) -> list[dict]:
        """Retrieve active Conditions for patient."""
        result = self._request("Condition", {"patient": patient_id, "clinical-status": "active"})
        if result and "entry" in result:
            return [e["resource"] for e in result["entry"]]
        return []
    
    def get_observations(self, patient_id: str, category: str = "laboratory") -> list[dict]:
        """Retrieve Observations (labs) for patient."""
        result = self._request("Observation", {"patient": patient_id, "category": category, "_count": "20"})
        if result and "entry" in result:
            return [e["resource"] for e in result["entry"]]
        return []
    
    def get_imaging_studies(self, patient_id: str) -> list[dict]:
        """Retrieve ImagingStudy resources for comparison."""
        result = self._request("ImagingStudy", {"patient": patient_id, "_count": "5"})
        if result and "entry" in result:
            return [e["resource"] for e in result["entry"]]
        return []
    
    def get_medications(self, patient_id: str) -> list[dict]:
        """Retrieve active MedicationRequests."""
        result = self._request("MedicationRequest", {"patient": patient_id, "status": "active"})
        if result and "entry" in result:
            return [e["resource"] for e in result["entry"]]
        return []


def get_patient_context(patient_id: str, hypotheses: list[str]) -> dict:
    """
    Retrieve and synthesize patient context relevant to hypotheses.
    
    Implements the Historian agent's FHIR workflow:
    ImagingStudy → Patient → Condition → Observation → Medication
    
    Returns:
        Dict with supporting_facts, contradicting_facts, confidence_adjustment
    """
    if settings.MOCK_MODELS or not patient_id:
        return _mock_fhir_context(patient_id)
    
    client = FHIRClient()
    
    # Retrieve resources
    conditions = client.get_conditions(patient_id)
    observations = client.get_observations(patient_id)
    medications = client.get_medications(patient_id)
    
    supporting = []
    contradicting = []
    adjustment = 0.0
    
    # Analyze conditions for relevance
    for cond in conditions:
        code = cond.get("code", {}).get("coding", [{}])[0]
        display = code.get("display", "Unknown condition")
        resource_id = cond.get("id", "")
        
        # Check if condition supports pneumonia hypothesis
        if any(term in display.lower() for term in ["diabetes", "copd", "immunocompromised"]):
            supporting.append(HistorianFact(
                fact_type="supporting",
                description=f"Risk factor: {display}",
                fhir_resource_id=resource_id,
                fhir_resource_type="Condition"
            ))
            adjustment += 0.05
    
    # Analyze labs
    for obs in observations:
        code = obs.get("code", {}).get("coding", [{}])[0]
        display = code.get("display", "")
        value = obs.get("valueQuantity", {}).get("value")
        
        if "wbc" in display.lower() and value and value > 12000:
            supporting.append(HistorianFact(
                fact_type="supporting",
                description=f"Elevated WBC: {value}",
                fhir_resource_id=obs.get("id", ""),
                fhir_resource_type="Observation"
            ))
            adjustment += 0.03
    
    return {
        "supporting_facts": supporting,
        "contradicting_facts": contradicting,
        "confidence_adjustment": adjustment,
        "clinical_summary": f"Retrieved {len(conditions)} conditions, {len(observations)} labs for {patient_id}"
    }


def _mock_fhir_context(patient_id: str) -> dict:
    """Generate mock FHIR context."""
    if not patient_id:
        return {
            "supporting_facts": [],
            "contradicting_facts": [],
            "confidence_adjustment": 0.0,
            "clinical_summary": "No patient ID provided"
        }
    
    return {
        "supporting_facts": [
            HistorianFact(
                fact_type="supporting",
                description="Type 2 Diabetes Mellitus - increased infection risk",
                fhir_resource_id="Condition/dm2-12345",
                fhir_resource_type="Condition"
            ),
            HistorianFact(
                fact_type="supporting",
                description="Elevated WBC: 14,500/µL (suggests infection)",
                fhir_resource_id="Observation/wbc-67890",
                fhir_resource_type="Observation"
            ),
            HistorianFact(
                fact_type="supporting",
                description="Elevated CRP: 85 mg/L (acute inflammation)",
                fhir_resource_id="Observation/crp-11111",
                fhir_resource_type="Observation"
            )
        ],
        "contradicting_facts": [],
        "confidence_adjustment": 0.08,
        "clinical_summary": f"Patient {patient_id}: Diabetic with elevated inflammatory markers supporting infectious etiology"
    }


# Singleton client
fhir_client = FHIRClient()
