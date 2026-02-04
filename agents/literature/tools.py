from .pubmed_entrez import search_pubmed
from .europe_pmc import search_europe_pmc
from .semantic_scholar import search_semantic_scholar

LITERATURE_TOOLS = {
    "pubmed_search": search_pubmed,
    "europe_pmc_search": search_europe_pmc,
    "semantic_scholar_search": search_semantic_scholar,
}



