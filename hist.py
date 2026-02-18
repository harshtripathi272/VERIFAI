import json
import re
from typing import List
from pydantic import BaseModel
from transformers import pipeline
import torch


# ==========================================================
# 1️⃣ Create MedGemma Chat Pipeline (Official Interface)
# ==========================================================

print("Loading MedGemma chat pipeline...")

pipe = pipeline(
    task="image-text-to-text",   # 🔥 official chat interface
    model="google/medgemma-1.5-4b-it",
    dtype=torch.bfloat16,        # more stable than fp16
    device="cuda"
)

print("Pipeline ready!")


# ==========================================================
# 2️⃣ Structured Output Schema
# ==========================================================

class HistorianFact(BaseModel):
    description: str
    resource_type: str
    resource_id: str


class HistorianOutput(BaseModel):
    supporting_facts: List[HistorianFact]
    contradicting_facts: List[HistorianFact]
    confidence_adjustment: float
    clinical_summary: str


# ==========================================================
# 3️⃣ Mock FHIR Evidence
# ==========================================================

mock_evidence = {
    "conditions": [
        {"id": "123", "code": {"coding": [{"display": "Pneumonia"}]}}
    ],
    "observations": [
        {
            "id": "456",
            "code": {"coding": [{"display": "White Blood Cell Count"}]},
            "valueQuantity": {"value": 18000, "unit": "cells/uL"}
        }
    ],
    "medications": [
        {
            "id": "789",
            "medicationCodeableConcept": {
                "coding": [{"display": "Azithromycin"}]
            }
        }
    ],
    "encounters": [
        {
            "id": "111",
            "reasonCode": [{"coding": [{"display": "Respiratory Infection"}]}]
        }
    ],
    "documents": [
        {
            "resourceType": "DiagnosticReport",
            "id": "999",
            "text": "Chest X-ray shows right lower lobe consolidation consistent with pneumonia."
        }
    ]
}


# ==========================================================
# 4️⃣ Evidence Summarizer
# ==========================================================

def summarize_fhir_evidence(evidence: dict) -> str:
    lines = []

    for c in evidence.get("conditions", []):
        display = c["code"]["coding"][0]["display"]
        lines.append(f"- Condition: {display} (Condition/{c['id']})")

    for o in evidence.get("observations", []):
        display = o["code"]["coding"][0]["display"]
        value = o["valueQuantity"]["value"]
        unit = o["valueQuantity"]["unit"]
        lines.append(f"- Lab: {display} {value} {unit} (Observation/{o['id']})")

    for m in evidence.get("medications", []):
        display = m["medicationCodeableConcept"]["coding"][0]["display"]
        lines.append(f"- Medication: {display} (MedicationRequest/{m['id']})")

    for e in evidence.get("encounters", []):
        reason = e["reasonCode"][0]["coding"][0]["display"]
        lines.append(f"- Encounter: {reason} (Encounter/{e['id']})")

    for d in evidence.get("documents", []):
        lines.append(
            f"- Document {d['resourceType']}/{d['id']}: {d['text']}"
        )

    return "\n".join(lines)


# ==========================================================
# 5️⃣ Robust JSON Extraction
# ==========================================================

def extract_json(text: str) -> str:
    if not text.strip():
        raise ValueError("Model returned empty output.")

    # Remove markdown fences
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    # Remove internal thought tokens if any slip through
    text = re.sub(r"<unused\d+>.*?\n", "", text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in output.")

    return text[start:end + 1]


# ==========================================================
# 6️⃣ Historian Reasoning Function
# ==========================================================

def reason_over_fhir(hypothesis: str, evidence: dict) -> HistorianOutput:

    summary = summarize_fhir_evidence(evidence)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"""
You are a senior clinical historian assisting a radiologist.

STRICT RULES:
- Output ONLY valid JSON.
- No explanations.
- No markdown.
- Must start with {{ and end with }}.
- confidence_adjustment must be between -0.3 and 0.3.

Hypothesis:
{hypothesis}

FHIR Evidence:
{summary}

Required JSON structure:

{{
  "supporting_facts": [
    {{
      "description": "string",
      "resource_type": "string",
      "resource_id": "string"
    }}
  ],
  "contradicting_facts": [
    {{
      "description": "string",
      "resource_type": "string",
      "resource_id": "string"
    }}
  ],
  "confidence_adjustment": float,
  "clinical_summary": "string"
}}
"""}]
        }
    ]

    output = pipe(
        text=messages,
        max_new_tokens=4096,   # 🔥 effectively "unlimited"
        do_sample=False
    )

    raw = output[0]["generated_text"][-1]["content"].strip()

    print("\n=== RAW MODEL OUTPUT ===\n")
    print(raw)
    print("\n========================\n")

    json_str = extract_json(raw)
    data = json.loads(json_str)

    parsed = HistorianOutput(**data)

    # Clamp confidence
    parsed.confidence_adjustment = max(
        -0.3,
        min(0.3, parsed.confidence_adjustment)
    )

    return parsed


# ==========================================================
# 7️⃣ Run Test
# ==========================================================

if __name__ == "__main__":

    hypothesis = "Right lower lobe pneumonia"

    result = reason_over_fhir(hypothesis, mock_evidence)

    print("\n===== STRUCTURED OUTPUT =====\n")
    print(result.model_dump_json(indent=2))
