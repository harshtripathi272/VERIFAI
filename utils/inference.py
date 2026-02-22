import re
import json
from typing import Any, Dict, List, Union

def extract_json(text: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Robustly extract JSON from model output text.
    Handles markdown code blocks, <unused> tokens, and surrounding text.
    """
    if not text:
        raise ValueError("Model returned empty output.")

    # 1. Remove markdown code fences
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    # 2. Remove internal thought tokens or special tokens (e.g. <unused123>)
    text = re.sub(r"<unused\d+>.*?\n", "", text)
    text = re.sub(r"<unused\d+>", "", text)
    
    # 3. Find the JSON object or array
    # Look for the first '{' or '[' and the last '}' or ']'
    start_brace = text.find("{")
    start_bracket = text.find("[")
    
    if start_brace == -1 and start_bracket == -1:
         raise ValueError("No JSON object or array found in output.")
    
    # Determine if we are looking for an object or array based on which comes first
    if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
        start = start_brace
        end = text.rfind("}")
    else:
        start = start_bracket
        end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError("Incomplete JSON object or array found in output.")

    json_str = text[start:end + 1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}\nExtracted string: {json_str}")
