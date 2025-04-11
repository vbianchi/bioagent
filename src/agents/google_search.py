import logging
from typing import Dict, Any

# Import the search tool wrapper function
from src.tools.web_search import search_google

# Import central AgentState definition
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# --- Node Function ---

# <<< Added google_search_tool argument >>>
def call_google_search_agent(state: AgentState, google_search_tool: Any, num_results: int = 5) -> Dict[str, Any]:
    """
    Agent node that performs a Google search using the refined query
    and the provided google_search_tool object.
    """
    logger.info("--- Calling Google Search Agent ---")
    refined_query = state.get("refined_query") or state.get("query") # Use refined if available
    error_message = state.get("error")
    google_search_results = []

    logger.info(f"Using query for Google Search: '{refined_query}'")
    try:
        # <<< Pass the tool object to the wrapper function >>>
        google_search_results = search_google(
            query=refined_query,
            search_tool=google_search_tool,
            num_results=num_results
        )
        # Check if the result itself indicates an error from the tool wrapper
        if google_search_results and isinstance(google_search_results[0], dict) and "error" in google_search_results[0]:
             search_error = f"Google Search tool error: {google_search_results[0]['error']}"
             error_message = (error_message + "; " + search_error) if error_message else search_error
             google_search_results = [] # Clear results if tool reported error

    except Exception as e:
        # Catch unexpected errors during the call
        search_error = f"Unexpected error calling search_google: {str(e)}"
        logger.error(search_error, exc_info=True)
        error_message = (error_message + "; " + search_error) if error_message else search_error
        google_search_results = []

    # Return only updated fields
    return {"google_results": google_search_results, "error": error_message}

