"""
Semantic Scholar Client

Graph API client for citation networks and paper metadata.
"""

import requests
from app.config import settings
from graph.state import LiteratureCitation


SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"


def search_semantic_scholar(query: str, max_results: int = 5) -> list[LiteratureCitation]:
    """
    Search Semantic Scholar for papers.
    
    Semantic Scholar provides:
    - Citation counts and networks
    - Influential citations
    - Abstract embeddings
    """
    if settings.MOCK_MODELS:
        return []
    
    headers = {}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
    
    try:
        params = {
            "query": query,
            "limit": max_results,
            "fields": "paperId,title,authors,year,abstract,citationCount,journal"
        }
        
        response = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/search",
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        papers = data.get("data", [])
        
        citations = []
        for paper in papers:
            citation = LiteratureCitation(
                pmid=paper.get("paperId", ""),
                title=paper.get("title", ""),
                authors=[a.get("name", "") for a in paper.get("authors", [])[:3]],
                journal=paper.get("journal", {}).get("name", "") if paper.get("journal") else "",
                year=paper.get("year"),
                relevance_summary=paper.get("abstract", "")[:300] if paper.get("abstract") else "",
                evidence_strength="medium",
                source="semanticscholar"
            )
            citations.append(citation)
        
        return citations
        
    except Exception as e:
        print(f"[SemanticScholar] Search failed: {e}")
        return []


def get_citing_papers(paper_id: str, max_results: int = 5) -> list[dict]:
    """Get papers that cite the given paper."""
    if settings.MOCK_MODELS:
        return []
    
    headers = {}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
    
    try:
        response = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations",
            params={"fields": "title,year,citationCount", "limit": max_results},
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception:
        return []
