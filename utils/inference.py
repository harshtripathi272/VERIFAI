import re
import json
import ast
from typing import Any, Dict, List, Union

def extract_json(text: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Robustly extract JSON from model output text.
    Handles markdown code blocks, <unused> tokens, and surrounding text.
    """
    if not text:
        raise ValueError("Model returned empty output.")

    # 1. Remove markdown code fences strictly
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    # 2. Remove internal thought tokens or special tokens
    text = re.sub(r"<unused\d+>.*?(?:</unused\d+>|\n|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"</?unused\d+>", "", text)
    
    # 3. Find candidates for JSON object or array
    candidates = []
    
    # Find block starting with { and ending with }
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        candidates.append(text[obj_start:obj_end + 1])
        
    # Find block starting with [ and ending with ]
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        candidates.append(text[arr_start:arr_end + 1])
        
    if not candidates:
         raise ValueError(f"No JSON object or array found in output. Raw: {text[:100]}...")
         
    # Sort candidates by length to prefer the larger encompassing block
    candidates.sort(key=len, reverse=True)
    
    last_error = None
    for json_str in candidates:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            last_error = e
            
            # Attempt 1: Fix missing trailing commas or dangling commas before brace
            try:
                fixed_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                return json.loads(fixed_str)
            except json.JSONDecodeError:
                pass
                
            # Attempt 2: Use ast.literal_eval for Python-style dicts (single quotes, True/False)
            try:
                fixed_str = json_str
                # Handle true/false/null -> True/False/None
                fixed_str = re.sub(r'\btrue\b', 'True', fixed_str)
                fixed_str = re.sub(r'\bfalse\b', 'False', fixed_str)
                fixed_str = re.sub(r'\bnull\b', 'None', fixed_str)
                parsed = ast.literal_eval(fixed_str)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                pass
                
    raise ValueError(f"Failed to decode JSON: {last_error}\nRaw block: {candidates[0][:200]}...")
