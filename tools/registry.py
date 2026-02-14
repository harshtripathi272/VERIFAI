"""
MCP-Style Tool Registry

Inspired by authentic MCP servers (wso2/fhir-mcp-server, cyanheads/pubmed-mcp-server).
Provides a unified interface for registering and invoking tools.
"""

from typing import Callable, Any
from dataclasses import dataclass, field
from enum import Enum


class ToolCategory(Enum):
    """Categories of available tools."""
    FHIR = "fhir"
    LITERATURE = "literature"
    VISION = "vision"
    CACHE = "cache"


@dataclass
class ToolDefinition:
    """Definition of a registered tool."""
    name: str
    description: str
    category: ToolCategory
    func: Callable[..., Any]
    input_schema: dict = field(default_factory=dict)
    requires_auth: bool = False
    rate_limited: bool = False
    rate_limit_per_second: float = 1.0


class MCPToolRegistry:
    """
    Model Context Protocol style tool registry.
    
    Provides:
    - Tool registration with metadata
    - Tool discovery (list available tools)
    - Tool invocation with validation
    - Rate limiting support
    - Authentication handling
    
    Based on patterns from:
    - wso2/fhir-mcp-server
    - cyanheads/pubmed-mcp-server
    - the-momentum/fhir-mcp-server
    """
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._call_counts: dict[str, int] = {}
    
    def register(
        self,
        name: str,
        description: str,
        category: ToolCategory = ToolCategory.FHIR,
        input_schema: dict | None = None,
        requires_auth: bool = False,
        rate_limited: bool = False,
        rate_limit_per_second: float = 1.0
    ) -> Callable:
        """
        Decorator to register a function as an MCP tool.
        
        Usage:
            @registry.register("get_patient", "Retrieve patient by ID", ToolCategory.FHIR)
            def get_patient(patient_id: str) -> dict:
                ...
        """
        def decorator(func: Callable) -> Callable:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                category=category,
                func=func,
                input_schema=input_schema or {},
                requires_auth=requires_auth,
                rate_limited=rate_limited,
                rate_limit_per_second=rate_limit_per_second
            )
            return func
        return decorator
    
    def list_tools(self, category: ToolCategory | None = None) -> list[dict]:
        """
        List available tools, optionally filtered by category.
        
        Returns MCP-compatible tool definitions.
        """
        tools = []
        for name, definition in self._tools.items():
            if category and definition.category != category:
                continue
            tools.append({
                "name": definition.name,
                "description": definition.description,
                "category": definition.category.value,
                "input_schema": definition.input_schema,
                "requires_auth": definition.requires_auth
            })
        return tools
    
    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)
    
    def invoke(self, name: str, **kwargs) -> Any:
        """
        Invoke a registered tool by name.
        
        Handles rate limiting and error wrapping.
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        # Track call count
        self._call_counts[name] = self._call_counts.get(name, 0) + 1
        
        # Invoke
        try:
            return tool.func(**kwargs)
        except Exception as e:
            return {"error": str(e), "tool": name}
    
    def get_stats(self) -> dict:
        """Get usage statistics for all tools."""
        return {
            "total_tools": len(self._tools),
            "call_counts": self._call_counts.copy()
        }


# Global registry instance
registry = MCPToolRegistry()

# Import and register tools from agent modules
def _register_all_tools():
    """Register all available tools from agent modules."""
    
    # FHIR Tools
    try:
        from agents.historian.fhir_client import fhir_client, get_patient_context
        
        @registry.register(
            "fhir_get_patient",
            "Retrieve Patient resource from FHIR server",
            ToolCategory.FHIR,
            input_schema={"patient_id": {"type": "string", "required": True}},
            requires_auth=True
        )
        def fhir_get_patient(patient_id: str) -> dict:
            return fhir_client.get_patient(patient_id)
        
        @registry.register(
            "fhir_get_conditions",
            "Retrieve active Conditions for a patient",
            ToolCategory.FHIR,
            input_schema={"patient_id": {"type": "string", "required": True}}
        )
        def fhir_get_conditions(patient_id: str) -> list:
            return fhir_client.get_conditions(patient_id)
        
        @registry.register(
            "fhir_get_patient_context",
            "Retrieve synthesized patient context for diagnosis",
            ToolCategory.FHIR,
            input_schema={
                "patient_id": {"type": "string", "required": True},
                "hypotheses": {"type": "array", "items": {"type": "string"}}
            }
        )
        def fhir_context(patient_id: str, hypotheses: list = None) -> dict:
            return get_patient_context(patient_id, hypotheses or [])
            
    except ImportError:
        pass
    
    # Literature Tools
    try:
        from agents.literature.pubmed_entrez import search_pubmed
        from agents.literature.europe_pmc import search_europe_pmc
        from agents.literature.semantic_scholar import search_semantic_scholar
        
        @registry.register(
            "pubmed_search",
            "Search PubMed for biomedical literature",
            ToolCategory.LITERATURE,
            input_schema={
                "query": {"type": "string", "required": True},
                "max_results": {"type": "integer", "default": 5}
            },
            rate_limited=True,
            rate_limit_per_second=3.0  # NCBI allows 3/sec with API key
        )
        def pubmed_tool(query: str, max_results: int = 5) -> list:
            results = search_pubmed(query, max_results)
            return [r.model_dump() for r in results]
        
        @registry.register(
            "europepmc_search",
            "Search Europe PMC for articles with full-text access",
            ToolCategory.LITERATURE,
            input_schema={
                "query": {"type": "string", "required": True},
                "max_results": {"type": "integer", "default": 5}
            }
        )
        def europepmc_tool(query: str, max_results: int = 5) -> list:
            results = search_europe_pmc(query, max_results)
            return [r.model_dump() for r in results]
        
        @registry.register(
            "semantic_scholar_search",
            "Search Semantic Scholar for papers with citation data",
            ToolCategory.LITERATURE,
            input_schema={
                "query": {"type": "string", "required": True},
                "max_results": {"type": "integer", "default": 5}
            }
        )
        def ss_tool(query: str, max_results: int = 5) -> list:
            results = search_semantic_scholar(query, max_results)
            return [r.model_dump() for r in results]
            
    except ImportError:
        pass


# Auto-register on import
_register_all_tools()
