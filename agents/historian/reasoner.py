import json
import re
import threading
import torch
from app.config import settings
from app.shared_model_loader import load_shared_medgemma, get_inference_lock

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

    # CRITICAL: Acquire lock before using model (shared across agents)
    _inference_lock = get_inference_lock()
    with _inference_lock:
        print(f"[Thread-{threading.current_thread().name}] Historian acquired model lock")
        print(f"[Historian] Preparing text-only message with prompt length: {len(prompt)} chars")
        
        # Use chat template format for MedGemma 1.5 (text-only, no image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template using processor
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.float16)
        
        input_len = inputs["input_ids"].shape[-1]
        print(f"[Historian] Input tokens: {inputs['input_ids'].shape}")
        print(f"[Historian] Starting generation (max_new_tokens=500)...")
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False  # Greedy decoding for deterministic output
            )
        
        print(f"[Historian] Generation complete! Output tokens: {outputs.shape}")
        
        # Extract only the newly generated tokens (skip the input prompt)
        generated_tokens = outputs[0][input_len:]
        raw = processor.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"[Historian] Generated output length: {len(raw)} chars")
        print(f"[Historian] Output preview: {raw[:300]}...")
        print(f"[Thread-{threading.current_thread().name}] Historian released model lock")



    # Clean up output - remove any special tokens that weren't caught
    raw = re.sub(r'<unused\d+>', '', raw)  # Remove <unusedNN> tokens
    raw = re.sub(r'<[^>]+>', '', raw)  # Remove any other special tags
    
    # Try to extract JSON - look for complete JSON object with our expected schema
    match = re.search(r'\{[^{}]*"supporting_facts"[^{}]*\}', raw, re.DOTALL)
    if not match:
        # Fallback: look for any JSON-like structure
        match = re.search(r"\{.*\}", raw, re.DOTALL)
    
    if not match:
        print("[Historian] WARNING: No JSON found in output")
        print(f"[Historian] Raw text: {raw[:500]}...")
        return {
            "supporting_facts": [],
            "contradicting_facts": [],
            "confidence_adjustment": 0.0
        }

    try:
        json_str = match.group()
        result = json.loads(json_str)
        print(f"[Historian] Successfully parsed JSON")
    except Exception as e:
        print(f"[Historian] JSON parse error: {e}")
        print(f"[Historian] Attempted to parse: {match.group()[:200]}...")
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
    print(result)

    return result
