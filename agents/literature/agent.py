"""
Literature Agent Node

RAG-style retrieval from PubMed, PMC, and Semantic Scholar.
OPTIMIZED: Singleton model loading, caching, parallel execution
"""
import json
import re
import asyncio
import concurrent.futures
from functools import lru_cache
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config import settings
from agents.literature.tools import LITERATURE_TOOLS
from agents.literature.prompt import SYSTEM_PROMPT

# === OPTIMIZATION 1: Singleton Model Loader ===
_MODEL_CACHE: Optional[tuple] = None

def load_medgemma():
    """Load MedGemma model once and cache it."""
    global _MODEL_CACHE
    
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    print("[LiteratureAgent] Loading MedGemma model (one-time initialization)...")
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
    _MODEL_CACHE = (model, tokenizer)
    print("[LiteratureAgent] Model loaded and cached")
    return _MODEL_CACHE

class ReActStepError(Exception):
    pass


class MedGemmaAgent:
    def __init__(self, model, tokenizer, max_steps: int = 3):  # Reduced from 5 to 3
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=400,  # Reduced from 600
            temperature=0.1,
            do_sample=False,  # Deterministic for speed
            pad_token_id=self.tokenizer.eos_token_id
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

    def _run_tool_parallel(self, tool_name: str, tool_input: str) -> Dict[str, Any]:
        """Run tool and return results."""
        if tool_name not in LITERATURE_TOOLS:
            raise ReActStepError(f"Unknown tool: {tool_name}")
        
        result = LITERATURE_TOOLS[tool_name](tool_input)
        return {
            "tool": tool_name,
            "input": tool_input,
            "results": [r.model_dump() for r in result]
        }

    # === OPTIMIZATION 2: Smart Tool Selection (Skip ReAct for simple queries) ===
    def _smart_tool_selection(self, query: str) -> Optional[str]:
        """Heuristically select the best tool without LLM overhead."""
        query_lower = query.lower()
        
        # High-specificity terms -> PubMed first
        if any(term in query_lower for term in ['pneumonia', 'diabetes', 'consolidation', 'radiograph']):
            return 'pubmed_search'
        
        # Research/citations -> Semantic Scholar
        if any(term in query_lower for term in ['meta-analysis', 'systematic review', 'citation']):
            return 'semantic_scholar_search'
        
        # Default to PubMed (most reliable)
        return 'pubmed_search'

    # === OPTIMIZATION 3: Parallel Multi-Tool Execution ===
    def run_parallel_search(self, query: str) -> str:
        """Execute all tools in parallel and aggregate results."""
        print(f"[LiteratureAgent] Running parallel search across all sources...")
        
        # Execute all tools concurrently
        futures = {}
        for tool_name in LITERATURE_TOOLS.keys():
            future = self.executor.submit(self._run_tool_parallel, tool_name, query)
            futures[tool_name] = future
        
        # Collect results as they complete
        all_results = []
        for tool_name, future in futures.items():
            try:
                result = future.result(timeout=10)  # 10s timeout per tool
                all_results.extend(result.get('results', []))
                print(f"[LiteratureAgent] {tool_name}: {len(result.get('results', []))} results")
            except Exception as e:
                print(f"[LiteratureAgent] {tool_name} failed: {e}")
        
        # Sort by evidence strength and deduplicate
        all_results.sort(key=lambda x: {'high': 3, 'medium': 2, 'low': 1}.get(x.get('evidence_strength', 'low'), 0), reverse=True)
        
        # Take top 10 unique results
        seen_titles = set()
        unique_results = []
        for result in all_results:
            title = result.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_results.append(result)
            if len(unique_results) >= 10:
                break
        
        # Format response
        summary = self._format_literature_summary(unique_results)
        return summary

    def _format_literature_summary(self, results: list) -> str:
        """Format literature results into concise summary."""
        if not results:
            return "No relevant literature found."
        
        summary_parts = [f"Found {len(results)} relevant studies:"]
        
        for i, r in enumerate(results[:5], 1):  # Top 5
            title = r.get('title', 'Unknown')
            authors = r.get('authors', [])
            year = r.get('year', 'N/A')
            evidence = r.get('evidence_strength', 'medium')
            relevance = r.get('relevance_summary', '')[:150]
            
            author_str = authors[0] if authors else "Unknown"
            summary_parts.append(
                f"\n{i}. [{evidence.upper()}] {author_str} et al. ({year}): {title}\n   {relevance}"
            )
        
        return "\n".join(summary_parts)

    def run(self, user_query: str) -> str:
        """
        Run literature search with optimizations.
        
        Strategy:
        1. Try smart tool selection first (no LLM overhead)
        2. If that fails or for complex queries, use parallel multi-search
        3. Only fall back to full ReAct if needed
        """
        # === OPTIMIZATION 4: Fast path for simple queries ===
        if settings.USE_FAST_LITERATURE_MODE:
            print("[LiteratureAgent] Using fast parallel mode")
            return self.run_parallel_search(user_query)
        
        # === Original ReAct loop (fallback) ===
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"USER QUERY:\n{user_query}\n"
            f"IMPORTANT: Choose ONE tool and provide final answer in 1-2 steps.\n"
        )

        for step in range(self.max_steps):
            raw_output = self._generate(prompt)
            parsed = self._extract_json(raw_output)

            if "final" in parsed:
                return parsed["final"]

            if "action" not in parsed:
                # Fallback to parallel search
                return self.run_parallel_search(user_query)

            observation = self._run_tool_parallel(parsed["action"]["tool"], parsed["action"]["input"])

            # === OPTIMIZATION 5: Early stopping if good results ===
            if len(observation.get('results', [])) >= 3:
                return self._format_literature_summary(observation['results'])

            prompt += (
                "\nASSISTANT_ACTION:\n"
                f"{json.dumps(parsed['action'], indent=2)}\n"
                "\nOBSERVATION:\n"
                f"{json.dumps(observation, indent=2)}\n"
            )

        # If exceeded steps, return what we have
        return self.run_parallel_search(user_query)


# === OPTIMIZATION 6: Query result caching ===
@lru_cache(maxsize=100)
def _cached_literature_search(query_hash: str, query: str) -> str:
    """Cache literature search results."""
    model, tokenizer = load_medgemma()
    agent = MedGemmaAgent(model=model, tokenizer=tokenizer, max_steps=3)
    return agent.run(query)


def literature_agent_node(state):
    # Load model once (singleton pattern)
    model, tokenizer = load_medgemma()

    # Create query
    query = f"""
Primary diagnosis hypothesis:
{state.radiologist_output.hypotheses[0].diagnosis}

Clinical history summary:
{state.historian_output.clinical_summary if state.get('historian_output') else 'Not available'}

Retrieve supporting or contradicting biomedical literature.
"""

    # Use cached search if possible
    query_hash = str(hash(query))
    
    try:
        if settings.USE_LITERATURE_CACHE:
            answer = _cached_literature_search(query_hash, query)
        else:
            agent = MedGemmaAgent(model=model, tokenizer=tokenizer, max_steps=3)
            answer = agent.run(query)
    except Exception as e:
        print(f"[LiteratureAgent] Error: {e}")
        answer = "Literature search temporarily unavailable."

    return {
        "literature_output": answer,
        "trace": [
            "LITERATURE_AGENT: Optimized execution with caching and parallel search"
        ]
    }

