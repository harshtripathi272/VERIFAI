"""
Literature Agent Node

RAG-style retrieval from PubMed, PMC, and Semantic Scholar.
"""
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config import settings
from agents.literature.tools import LITERATURE_TOOLS
from agents.literature.prompt import SYSTEM_PROMPT


import json
from typing import Dict, Any

from agents.literature.tools import LITERATURE_TOOLS
from agents.literature.prompt import SYSTEM_PROMPT



def load_medgemma():
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
    return model, tokenizer

class ReActStepError(Exception):
    pass


class MedGemmaAgent:
    def __init__(self, model, tokenizer, max_steps: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ReActStepError("No JSON object found in model output")
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise ReActStepError(f"Invalid JSON: {e}")

    def _run_tool(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if "tool" not in action or "input" not in action:
            raise ReActStepError("Action must contain 'tool' and 'input'")

        tool_name = action["tool"]
        tool_input = action["input"]

        if not isinstance(tool_input, str):
            raise ReActStepError("Tool input must be a string")

        if tool_name not in LITERATURE_TOOLS:
            raise ReActStepError(f"Unknown tool: {tool_name}")

        result = LITERATURE_TOOLS[tool_name](tool_input)

        # Convert observations to pure JSON
        return {
            "tool": tool_name,
            "input": tool_input,
            "results": [r.model_dump() for r in result]
        }

    def run(self, user_query: str) -> str:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"USER QUERY:\n{user_query}\n"
        )

        for step in range(self.max_steps):
            raw_output = self._generate(prompt)
            parsed = self._extract_json(raw_output)

            if "final" in parsed:
                return parsed["final"]

            if "action" not in parsed:
                raise ReActStepError("Expected 'action' or 'final'")

            observation = self._run_tool(parsed["action"])

            prompt += (
                "\nASSISTANT_ACTION:\n"
                f"{json.dumps(parsed['action'], indent=2)}\n"
                "\nOBSERVATION:\n"
                f"{json.dumps(observation, indent=2)}\n"
            )

        raise ReActStepError("Exceeded maximum reasoning steps")



def literature_agent_node(state):
    model, tokenizer = load_medgemma()

    agent = MedGemmaAgent(
        model=model,
        tokenizer=tokenizer,
        max_steps=5
    )

    query = f"""
Primary diagnosis hypothesis:
{state.radiologist_output.hypotheses[0].diagnosis}

Clinical history summary:
{state.historian_output.clinical_summary}

Retrieve supporting or contradicting biomedical literature.
"""

    answer = agent.run(query)

    return {
        "literature_output": answer,
        "trace": [
            "LITERATURE_AGENT: MedGemma ReAct loop executed",
            "TOOLS: PubMed / EuropePMC / SemanticScholar"
        ]
    }

