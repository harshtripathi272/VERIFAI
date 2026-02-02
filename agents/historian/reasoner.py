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

    for c in evidence.get("conditions", []):
        coding = c["code"]["coding"][0]
        lines.append(
            f"- Condition {coding.get('display')} (Condition/{c.get('id')})"
        )

    for o in evidence.get("observations", []):
        coding = o["code"]["coding"][0]
        value = o.get("valueQuantity", {}).get("value")
        unit = o.get("valueQuantity", {}).get("unit", "")
        lines.append(
            f"- Lab {coding.get('display')}: {value} {unit} (Observation/{o.get('id')})"
        )

    for m in evidence.get("medications", []):
        med = m.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
        lines.append(
            f"- Medication {med.get('display')} (MedicationRequest/{m.get('id')})"
        )

    return "\n".join(lines) if lines else "No relevant historical data found."


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
You are a clinical historian.

Hypothesis:
{hypothesis}

FHIR Evidence:
{summary}

Rules:
- Use ONLY the evidence above
- Do NOT invent facts
- Reference resource IDs exactly
- Output VALID JSON ONLY

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
        max_new_tokens=400,
        temperature=0.2
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {
            "supporting_facts": [],
            "contradicting_facts": [],
            "confidence_adjustment": 0.0
        }

    result = json.loads(match.group())

    # Clamp confidence safely
    result["confidence_adjustment"] = max(
        -0.2,
        min(0.2, float(result.get("confidence_adjustment", 0.0)))
    )

    return result
