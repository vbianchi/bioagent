import logging
from typing import List, Dict, Any

# This module now relies on the actual google_search tool object being passed in.

logger = logging.getLogger(__name__)

def search_google(query: str, search_tool: Any, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a Google search for the given query using the provided search tool object
    and returns formatted results.

    Args:
        query: The search query string.
        search_tool: The actual google_search tool object provided by the environment.
        num_results: The desired number of results.

    Returns:
        A list of dictionaries, each containing 'title', 'link', 'snippet', 'source',
        or a list containing an error dictionary if failed.
    """
    logger.info(f"Preparing to call Google Search tool for '{query}' (num_results={num_results})...")
    results = []
    if search_tool is None:
        logger.error("Google Search tool object was not provided to search_google function.")
        return [{"error": "Google Search tool not configured."}]

    try:
        # Call the search method on the provided tool object
        # Assuming the tool object has a 'search' method matching the API definition
        search_response = search_tool.search(query=query, num_results=num_results)
        logger.info("Google Search tool called successfully.")

        # Process the response - structure depends on the actual tool's output
        # Assuming response has a 'results' attribute which is a list of objects/dicts
        # with 'title', 'url' (or 'link'), and 'snippet' attributes/keys.
        if search_response and hasattr(search_response, 'results') and search_response.results:
             raw_results = search_response.results
             logger.info(f"Google Search tool returned {len(raw_results)} raw results.")
             for res in raw_results:
                 # Adapt attribute names if needed (e.g., res.url vs res.link)
                 title = getattr(res, 'title', 'N/A')
                 link = getattr(res, 'url', getattr(res, 'link', '#')) # Check for url or link
                 snippet = getattr(res, 'snippet', 'N/A')
                 results.append({
                     "title": title,
                     "link": link,
                     "snippet": snippet,
                     "source": "Google Search" # Add source identifier
                 })
        else:
             logger.info("Google Search tool returned no results or results attribute missing.")

    except AttributeError as ae:
        logger.error(f"Error calling Google Search tool: Method 'search' not found or attribute error. {ae}", exc_info=True)
        results = [{"error": f"Google Search tool method error: {ae}"}]
    except Exception as e:
        logger.error(f"Error during Google Search execution: {e}", exc_info=True)
        results = [{"error": f"Google Search failed: {e}"}]

    logger.info(f"Formatted {len(results)} Google Search results.")
    return results

