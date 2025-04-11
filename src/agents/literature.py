import logging
from typing import Dict, Any

# Import tools
from src.tools.literature_search import search_pubmed, search_arxiv

# Try importing the central AgentState definition
try:
    from src.core.state import AgentState
except ImportError:
    # Fallback definition if import fails
    from typing import TypedDict, List, Dict, Any, Optional, Tuple # Keep import here
    logger.warning("Could not import AgentState from src.core.state, using fallback definition in literature.py.")
    class AgentState(TypedDict): # <<< Moved class definition to new line
        query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
        search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
        chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]
        run_dir: Optional[str]; arxiv_results_found: bool; download_preference: Optional[str]
        code_request: Optional[str]; generated_code: Optional[str]
        generated_code_language: Optional[str]; google_results: Optional[List[Dict[str, Any]]]
        synthesized_report: Optional[str]; route_intent: Optional[str] # Ensure all fields

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
