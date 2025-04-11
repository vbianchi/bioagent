import logging
from typing import Dict, Any

# Import the search tool
from src.tools.web_search import search_google

# Try importing the central AgentState definition
try:
    from src.core.state import AgentState
except ImportError:
    # Fallback definition if import fails
    from typing import TypedDict, List, Dict, Any, Optional, Tuple # Keep import here
    logger.warning("Could not import AgentState from src.core.state, using fallback definition in google_search.py.")
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

def call_google_search_agent(state: AgentState, num_results: int = 5) -> Dict[str, Any]:
    """
    Agent node that performs a Google search using the refined query.
    """
    logger.info("--- Calling Google Search Agent ---")
    refined_query = state.get("refined_query") or state.get("query") # Use refined if available
    error_message = state.get("error")
    google_search_results = []

    logger.info(f"Using query for Google Search: '{refined_query}'")
    try:
        google_search_results = search_google(query=refined_query, num_results=num_results)
    except Exception as e:
        search_error = f"Google Search failed: {str(e)}"
        logger.error(search_error, exc_info=True)
        error_message = (error_message + "; " + search_error) if error_message else search_error

    # Return only updated fields
    return {"google_results": google_search_results, "error": error_message}

