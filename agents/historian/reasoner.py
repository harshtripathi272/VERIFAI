import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import settings


tokenizer = None
model = None


def load_medgemma():
    global tokenizer, model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            settings.MEDGEMMA_4B_MODEL,
            token=settings.HUGGINGFACE_TOKEN
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.MEDGEMMA_4B_MODEL,
            device_map="auto",
            torch_dtype="auto",
            token=settings.HUGGINGFACE_TOKEN
        )


def summarize_fhir_evidence(evidence: dict) -> str:
    lines = []

    # 1. Conditions
    for c in evidence.get("conditions", []):
        coding = c.get("code", {}).get("coding", [{}])[0]
        lines.append(f"- Condition: {coding.get('display')} (Condition/{c.get('id')})")

    # 2. Labs/Observations
    for o in evidence.get("observations", []):
        coding = o.get("code", {}).get("coding", [{}])[0]
        value = o.get("valueQuantity", {}).get("value")
        unit = o.get("valueQuantity", {}).get("unit", "")
        lines.append(f"- Lab {coding.get('display')}: {value} {unit} (Observation/{o.get('id')})")

    # 3. Medications
    for m in evidence.get("medications", []):
        med = m.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
        lines.append(f"- Medication: {med.get('display')} (MedicationRequest/{m.get('id')})")

    # 4. Procedures
    for p in evidence.get("procedures", []):
        coding = p.get("code", {}).get("coding", [{}])[0]
        lines.append(f"- Procedure: {coding.get('display')} (Procedure/{p.get('id')})")

    # 5. Allergies
    for a in evidence.get("allergies", []):
        coding = a.get("code", {}).get("coding", [{}])[0]
        lines.append(f"- Allergy: {coding.get('display')} (AllergyIntolerance/{a.get('id')})")

    # 6. Encounters
    for e in evidence.get("encounters", []):
        reason = e.get("reasonCode", [{}])[0].get("coding", [{}])[0].get("display", "Clinical Visit")
        lines.append(f"- Encounter Reason: {reason} (Encounter/{e.get('id')})")

    # 7. Documents (Clinically weighted)
    for d in evidence.get("documents", []):
        lines.append(f"\n--- Clinical Document ({d['resourceType']}/{d['id']}) ---")
        lines.append(f"Category: {d.get('category')}")
        # Take first 1500 chars for context
        text_snippet = d['text'].strip()[:1500] 
        lines.append(f"Content: {text_snippet}...")

    return "\n".join(lines) if lines else "No relevant historical records found."


def reason_over_fhir(hypothesis: str, evidence: dict) -> dict:
    if settings.MOCK_MODELS:
        return {
            "supporting_facts": [],
            "contradicting_facts": [],
            "confidence_adjustment": 0.0
        }

    load_medgemma()
    summary = summarize_fhir_evidence(evidence)

    prompt = f"""
You are a senior clinical historian assisting a radiologist.

Hypothesis to evaluate:
{hypothesis}

Full FHIR-based Clinical History:
{summary}

Analysis Objective:
Determine if the patient's history supports or contradicts the current radiology hypothesis.
Consider prior diagnoses, lab trends, recent procedures, and the text of previous reports.

Rules:
1. Use ONLY the provided evidence.
2. For document-based evidence, identify specific snippets that confirm or rule out the diagnosis.
3. Reference resource IDs exactly (e.g., Condition/123).
4. Output VALID JSON ONLY.

JSON schema:
{{
  "supporting_facts": [
    {{"description": "...", "resource_type": "...", "resource_id": "..."}}
  ],
  "contradicting_facts": [
    {{"description": "...", "resource_type": "...", "resource_id": "..."}}
  ],
  "confidence_adjustment": number
}}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.1
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {
            "supporting_facts": [],
            "contradicting_facts": [],
            "confidence_adjustment": 0.0
        }

    try:
        result = json.loads(match.group())
    except Exception:
         return {
            "supporting_facts": [],
            "contradicting_facts": [],
            "confidence_adjustment": 0.0
        }

    # Clamp confidence safely (-0.3 to 0.3 as history can be quite telling)
    try:
        adj = float(result.get("confidence_adjustment", 0.0))
        result["confidence_adjustment"] = max(-0.3, min(0.3, adj))
    except (ValueError, TypeError):
        result["confidence_adjustment"] = 0.0

    return result
