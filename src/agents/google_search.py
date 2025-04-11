import logging
from typing import Dict, Any

# Import the search tool
from src.tools.web_search import search_google

# Import central AgentState definition
from src.core.state import AgentState

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

