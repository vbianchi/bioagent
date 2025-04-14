import logging
from typing import List, Dict, Any, Optional
# <<< Import DDGS from the underlying library >>>
from duckduckgo_search import DDGS
# json and re are no longer needed for parsing the response here
# import json
# import re

logger = logging.getLogger(__name__)

def search_duckduckgo(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a DuckDuckGo search using the duckduckgo-search library directly
    and returns structured results.

    Args:
        query: The search query string.
        num_results: The desired maximum number of results.

    Returns:
        A list of dictionaries, each containing 'title', 'link', 'snippet', 'source'.
        Returns an empty list or list with error dict on failure.
    """
    logger.info(f"Performing DuckDuckGo search for: '{query}' (max_results={num_results})...")
    results: List[Dict[str, Any]] = []
    try:
        # <<< Use the DDGS object directly >>>
        # Instantiate the search object
        with DDGS() as ddgs:
            # Use ddgs.text() for standard search results
            # It returns a generator, convert it to a list
            # Note: The library's max_results might behave slightly differently than the wrapper's
            ddg_results_list = list(ddgs.text(query, max_results=num_results))

        logger.info(f"duckduckgo-search library returned {len(ddg_results_list)} items.")

        # --- Process the structured response ---
        # The library returns a list of dicts with keys: 'title', 'href', 'body'
        for i, res in enumerate(ddg_results_list):
            if isinstance(res, dict):
                # Map the library's keys to our standard format
                title = res.get('title', f'DDG Result {i+1}')
                link = res.get('href', '#') # Map 'href' to 'link'
                snippet = res.get('body', 'No snippet available.') # Map 'body' to 'snippet'

                results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "source": f"DuckDuckGo:{i+1}" # Add source identifier
                })
            else:
                logger.warning(f"Skipping non-dictionary item in DDG results: {res}")

    except ImportError:
         # This error shouldn't happen if requirements are met, but keep for safety
         logger.error("duckduckgo_search package not found. Please run 'uv pip install -U duckduckgo-search'")
         results = [{"error": "Missing dependency: duckduckgo-search"}]
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search execution: {e}", exc_info=True)
        results = [{"error": f"DuckDuckGo search failed: {e}"}]

    logger.info(f"Formatted {len(results)} DuckDuckGo results.")
    return results
