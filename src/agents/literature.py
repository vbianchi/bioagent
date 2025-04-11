import logging
from typing import Dict, Any

# Import tools
from src.tools.literature_search import search_pubmed, search_arxiv
# Import central AgentState definition
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# --- Node Function ---

def call_literature_agent(state: AgentState, max_pubmed: int, max_arxiv: int) -> Dict[str, Any]:
    """
    Literature Agent: Searches PubMed & ArXiv using the refined query.
    Query refinement is now done in a separate preceding node.
    """
    logger.info("--- Calling Literature Agent ---")
    # Use the refined query set by the previous node
    refined_query_for_search = state.get("refined_query") or state.get("query") # Fallback to original query
    logger.info(f"Using refined query: {refined_query_for_search}")

    error_message = state.get("error") # Preserve existing errors
    combined_results = []
    arxiv_found = False

    try:
        # --- Perform Searches using Tools ---
        pubmed_results = search_pubmed(refined_query_for_search, max_pubmed)
        arxiv_results = search_arxiv(refined_query_for_search, max_arxiv)

        combined_results.extend(pubmed_results)
        combined_results.extend(arxiv_results)
        logger.info(f"Total combined results: {len(combined_results)}")

        # Check if any ArXiv results were found
        if any(r.get('source') == 'ArXiv' for r in combined_results):
            arxiv_found = True
            logger.info("ArXiv results found.")

    except Exception as e:
        search_error = f"Literature search failed: {str(e)}"
        logger.error(search_error, exc_info=True)
        error_message = (error_message + "; " + search_error) if error_message else search_error
        combined_results = combined_results or []

    # Return only updated fields
    return {
        "search_results": combined_results,
        "error": error_message,
        "arxiv_results_found": arxiv_found
    }
