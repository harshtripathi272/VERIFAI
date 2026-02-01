"""
Literature Agent Node

RAG-style retrieval from PubMed, PMC, and Semantic Scholar.
"""

from graph.state import VerifaiState, LiteratureOutput, LiteratureCitation
from .pubmed_entrez import search_pubmed
from .europe_pmc import search_europe_pmc
from .semantic_scholar import search_semantic_scholar


def literature_node(state: VerifaiState) -> dict:
    """
    Literature Agent: Retrieve and rank evidence from biomedical literature.
    
    Given hypotheses and clinical context, performs:
    1. Query formulation from hypotheses
    2. Multi-source search (PubMed, Europe PMC, Semantic Scholar)
    3. Deduplication and relevance ranking
    4. Evidence strength scoring
    
    Returns top-5 articles with relevance summaries.
    """
    rad_output = state.get("radiologist_output")
    hist_output = state.get("historian_output")
    
    # Formulate search query from hypotheses
    query_parts = []
    if rad_output and rad_output.hypotheses:
        top_dx = rad_output.hypotheses[0].diagnosis
        query_parts.append(f'"{top_dx}"')
    
    # Add clinical context if available
    if hist_output and hist_output.supporting_facts:
        # Extract key conditions for query
        for fact in hist_output.supporting_facts[:2]:
            if "diabetes" in fact.description.lower():
                query_parts.append("diabetes")
    
    # Default query components
    query_parts.extend(["chest radiograph", "diagnosis"])
    query = " ".join(query_parts)
    
    # Multi-source search
    all_citations = []
    
    # Primary: PubMed
    pubmed_results = search_pubmed(query, max_results=5)
    all_citations.extend(pubmed_results)
    
    # Secondary: Europe PMC (may have additional content)
    europepmc_results = search_europe_pmc(query, max_results=3)
    all_citations.extend(europepmc_results)
    
    # Tertiary: Semantic Scholar
    ss_results = search_semantic_scholar(query, max_results=3)
    all_citations.extend(ss_results)
    
    # Deduplicate by PMID/title
    seen = set()
    unique_citations = []
    for cit in all_citations:
        key = cit.pmid or cit.title[:50]
        if key not in seen:
            seen.add(key)
            unique_citations.append(cit)
    
    # Take top 5
    top_citations = unique_citations[:5]
    
    # Determine overall evidence strength
    high_count = sum(1 for c in top_citations if c.evidence_strength == "high")
    if high_count >= 2:
        overall_strength = "high"
    elif len(top_citations) >= 3:
        overall_strength = "medium"
    else:
        overall_strength = "low"
    
    output = LiteratureOutput(
        citations=top_citations,
        overall_evidence_strength=overall_strength
    )
    
    trace_entry = (
        f"LITERATURE: Retrieved {len(top_citations)} citations. "
        f"Sources: PubMed={len(pubmed_results)}, EuropePMC={len(europepmc_results)}, "
        f"S2={len(ss_results)}. Strength={overall_strength}"
    )
    
    return {
        "literature_output": output,
        "trace": [trace_entry]
    }
