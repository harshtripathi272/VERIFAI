"""
Citation Formatting Tools

Format literature citations for display and export.
"""

from graph.state import LiteratureCitation


def format_ama(citation: LiteratureCitation) -> str:
    """Format citation in AMA style."""
    authors = ", ".join(citation.authors[:3])
    if len(citation.authors) > 3:
        authors += ", et al"
    
    year = citation.year or "N.d."
    
    return f"{authors}. {citation.title}. {citation.journal}. {year}. PMID: {citation.pmid}"


def format_vancouver(citation: LiteratureCitation) -> str:
    """Format citation in Vancouver style."""
    authors = ", ".join(citation.authors[:6])
    if len(citation.authors) > 6:
        authors += ", et al."
    
    return f"{authors}. {citation.title}. {citation.journal} {citation.year or ''}."


def format_for_display(citations: list[LiteratureCitation]) -> list[dict]:
    """Format citations for UI display."""
    formatted = []
    for i, cit in enumerate(citations, 1):
        formatted.append({
            "number": i,
            "pmid": cit.pmid,
            "title": cit.title,
            "authors_short": ", ".join(cit.authors[:2]) + (" et al" if len(cit.authors) > 2 else ""),
            "journal_year": f"{cit.journal} ({cit.year})" if cit.journal else str(cit.year or "N/A"),
            "strength": cit.evidence_strength,
            "source": cit.source,
            "summary": cit.relevance_summary
        })
    return formatted
