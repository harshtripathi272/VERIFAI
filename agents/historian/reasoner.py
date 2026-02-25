import json
import re
import threading
import torch
from app.config import settings
from app.shared_model_loader import load_shared_medgemma, get_inference_lock
from utils.inference import extract_json
from langchain_core.output_parsers import PydanticOutputParser
from graph.state import HistorianOutput, HistorianFact
from .fhir_client import fhir_client

# Thread-safe singleton pattern - now using shared loader
processor = None
model = None
_LOAD_LOCK = threading.Lock()
# Inference lock is now managed by shared_model_loader


def load_medgemma():
    """Load MedGemma model using shared loader (singleton across agents)."""
    global processor, model

    # Quick check without lock
    if processor is not None and model is not None:
        return

    # Acquire lock for loading
    with _LOAD_LOCK:
        # Double-check after acquiring lock
        if processor is not None and model is not None:
            return
        
        print("[Historian] Loading shared MedGemma model...")
        model, processor = load_shared_medgemma()
        print("[Historian] Using shared model instance")


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


def reason_over_fhir(hypothesis: str, evidence: dict, current_fhir: dict = None, uncertainty_history: list = None) -> HistorianOutput:

    if settings.MOCK_MODELS:
        return HistorianOutput(
            supporting_facts=[],
            contradicting_facts=[],
            confidence_adjustment=0.0,
            clinical_summary="Mocked output."
        )

    load_medgemma()
    summary = summarize_fhir_evidence(evidence)
    
    current_fhir_summary = "No current FHIR report provided."
    if current_fhir:
        try:
            current_fhir_summary = fhir_client.filter_current_fhir(current_fhir, hypothesis, top_k=5)
        except Exception as e:
            print(f"[Historian] Filtering failed, falling back to JSON dump: {e}")
            current_fhir_summary = json.dumps(current_fhir, indent=2)[:2000]

    prompt = f"""
You are a senior clinical historian assisting a radiologist.

Hypothesis:
{hypothesis}

Historical FHIR Context (Patient's past patterns for this hypothesis):
{summary}

Current Patient FHIR Report (Latest data):
{current_fhir_summary}
"""    
    # === Uncertainty spike: compact with reasoning guidance ===
    if uncertainty_history and len(uncertainty_history) >= 2:
        prev_u = uncertainty_history[-2]["system_uncertainty"]
        latest_u = uncertainty_history[-1]["system_uncertainty"]
        if latest_u > prev_u:
            spike_agent = uncertainty_history[-1]["agent"]
            prompt += (
                f"\n⚠️ SPIKE: System entropy rose after '{spike_agent}' ({prev_u:.3f}→{latest_u:.3f}). "
                "Reasoning: that agent's output contradicted the hypothesis, increasing doubt. "
                "Action: (1) prioritize searching for contradicting FHIR facts that explain the spike, "
                "(2) do NOT leave contradicting_facts empty if spike is present, "
                "(3) lean confidence_adjustment negative.\n"
            )

    prompt += f"""
Task:
Determine whether the combination of historical patterns and the current report SUPPORTS or CONTRADICTS the hypothesis. Use the historical context to interpret the current report.

CRITICAL OUTPUT REQUIREMENTS:

You MUST return STRICT JSON with EXACT field names.

Schema:

{{
  "supporting_facts": [
    {{
      "fact_type": "supporting",
      "description": "string",
      "fhir_resource_id": "UUID only (no prefix)",
      "fhir_resource_type": "Condition | DiagnosticReport | Observation | DocumentReference | MedicationRequest | Procedure | Encounter"
    }}
  ],
  "contradicting_facts": [
    {{
      "fact_type": "contradicting",
      "description": "string",
      "fhir_resource_id": "UUID only",
      "fhir_resource_type": "ResourceType"
    }}
  ],
  "confidence_adjustment": -0.3 to 0.3,
  "clinical_summary": "string"
}}
CRITICAL REASONING RULES:

1. You MUST extract AT LEAST ONE supporting or contradicting fact if ANY relevant information is present in the historical or current report, even if it is indirect or weak evidence.
2. If the evidence is nonspecific, you can assign it as weak support (+0.1) but YOU MUST STILL EXTRACT IT AS A FACT in the "supporting_facts" or "contradicting_facts" arrays.
3. Do NOT leave both arrays empty unless the reports are completely blank. The Critic Agent relies on these facts.
4. If a finding could be caused by many conditions, use the global historical context patterns to justify extracting it as a supporting fact.
5. If evidence is an alternative explanation for the hypothesis, classify as a contradiction.

Rules:
- DO NOT create a field called "resource_id"
- DO NOT combine type and ID
- fhir_resource_id must contain ONLY the UUID
- fact_type must be either "supporting" or "contradicting"
- Return ONLY valid JSON
"""


    _inference_lock = get_inference_lock()

    with _inference_lock:

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False
            )

        generated_tokens = outputs[0][input_len:]

        raw = processor.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()
        print("[Historian] Raw output:", raw)
    if not raw:
        print("[Historian] WARNING: Model returned empty string")
        return HistorianOutput(
            supporting_facts=[],
            contradicting_facts=[],
            confidence_adjustment=0.0,
            clinical_summary="Model returned empty output."
        )

    try:
        parsed = extract_json(raw)
        output = HistorianOutput(**parsed)
        output.confidence_adjustment = max(-0.3, min(0.3, output.confidence_adjustment))
        return output

    except Exception as e:
        print("[Historian] JSON parse error:", e)
        print("RAW OUTPUT:", raw)
        return HistorianOutput(
            supporting_facts=[],
            contradicting_facts=[],
            confidence_adjustment=0.0,
            clinical_summary="Failed to parse model output."
        )
