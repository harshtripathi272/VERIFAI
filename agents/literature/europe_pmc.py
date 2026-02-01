"""
Europe PMC Client

REST client for Europe PMC API (alternative to PubMed with additional content).
Provides access to PMC Open Access subset and full-text links.
"""

import requests
from app.config import settings
from graph.state import LiteratureCitation


EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def search_europe_pmc(query: str, max_results: int = 5) -> list[LiteratureCitation]:
    """
    Search Europe PMC for relevant articles.
    
    Europe PMC includes:
    - PubMed content
    - PMC full-text articles
    - Patents, guidelines, and other sources
    """
    if settings.MOCK_MODELS:
        return []  # Defer to PubMed mock
    
    try:
        params = {
            "query": query,
            "format": "json",
            "pageSize": max_results,
            "resultType": "core"
        }
        
        response = requests.get(
            f"{EUROPE_PMC_API}/search",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        
        data = response.json()
        results = data.get("resultList", {}).get("result", [])
        
        citations = []
        for result in results:
            citation = LiteratureCitation(
                pmid=result.get("pmid", result.get("id", "")),
                title=result.get("title", ""),
                authors=[a.get("fullName", "") for a in result.get("authorList", {}).get("author", [])[:3]],
                journal=result.get("journalTitle", ""),
                year=int(result.get("pubYear", 0)) or None,
                relevance_summary=result.get("abstractText", "")[:300],
                evidence_strength="medium",
                source="europepmc"
            )
            citations.append(citation)
        
        return citations
        
    except Exception as e:
        print(f"[EuropePMC] Search failed: {e}")
        return []


def get_full_text_links(pmcid: str) -> list[str]:
    """
    Get full-text download links for a PMC article.
    
    Returns list of available format URLs (PDF, XML, etc.)
    """
    if settings.MOCK_MODELS:
        return []
    
    try:
        response = requests.get(
            f"{EUROPE_PMC_API}/{pmcid}/fullTextXML",
            timeout=10
        )
        if response.status_code == 200:
            return [f"{EUROPE_PMC_API}/{pmcid}/fullTextXML"]
        return []
    except Exception:
        return []
