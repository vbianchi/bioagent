import logging
from typing import TypedDict, List, Dict, Any, Optional, Tuple

# Import tools
from src.tools.literature_search import search_pubmed, search_arxiv

# Assuming AgentState is defined centrally or passed appropriately
# TODO: Define AgentState in a shared location later
class AgentState(TypedDict):
    query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
    chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]
    run_dir: str; arxiv_results_found: bool; download_preference: Optional[str]
    code_request: Optional[str]; generated_code: Optional[str]
    generated_code_language: Optional[str]

logger = logging.getLogger(__name__)

# --- Node Function ---

def call_literature_agent(state: AgentState, llm, refinement_prompt_template: str,
                          max_pubmed: int, max_arxiv: int) -> AgentState:
    """Literature Agent: Refines query, searches PubMed & ArXiv."""
    logger.info("--- Calling Literature Agent ---")
    original_query = state['query']
    run_dir = state['run_dir'] # run_dir is needed by download node later, pass it through state
    logger.info(f"Received original query: {original_query}")
    error_message = state.get("error") # Preserve existing errors
    refined_query_for_search = original_query
    combined_results = []
    arxiv_found = False

    try:
        # --- LLM Query Refinement Step ---
        logger.info("Refining query for literature search...")
        refinement_prompt = refinement_prompt_template.format(query=original_query)
        try:
            refinement_response = llm.invoke(refinement_prompt)
            refined_query_for_search = refinement_response.content.strip()
            logger.info(f"Refined query: {refined_query_for_search}")
        except Exception as refine_e:
            logger.warning(f"LLM query refinement failed: {refine_e}. Using original query.")
            refine_error = f"Query refinement failed: {refine_e}. "
            error_message = (error_message + "; " + refine_error) if error_message else refine_error

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
        # Ensure combined_results is a list even on error
        combined_results = combined_results or []

    # Return the *entire state* merged with updates
    return {
        **state,
        "refined_query": refined_query_for_search,
        "search_results": combined_results,
        "error": error_message,
        "arxiv_results_found": arxiv_found
    }

