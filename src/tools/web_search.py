import logging
from typing import List, Dict, Any
# <<< Import DuckDuckGo Tool >>>
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger(__name__)

# Initialize the DDG search tool once
# It doesn't require API keys for basic snippet/link retrieval
ddg_search = DuckDuckGoSearchRun()

def search_duckduckgo(query: str) -> List[Dict[str, Any]]:
    """
    Performs a DuckDuckGo search for the given query using DuckDuckGoSearchRun
    and returns formatted results (primarily snippets).

    Args:
        query: The search query string.

    Returns:
        A list of dictionaries, each containing 'title', 'link', 'snippet', 'source'.
        Returns an empty list or list with error dict on failure.
        NOTE: DuckDuckGoSearchRun primarily returns a single string blob of results.
              We need to parse it or use a different DDG tool if we need structured output reliably.
              Let's try basic parsing first. If it's unreliable, we might need
              DuckDuckGoSearchResults which requires installing 'duckduckgo_search'.
    """
    logger.info(f"Performing DuckDuckGo search for: '{query}'...")
    results: List[Dict[str, Any]] = []
    try:
        # Run the search - this typically returns a formatted string
        search_response_str = ddg_search.run(query)
        logger.debug(f"DuckDuckGoSearchRun response string:\n{search_response_str}")

        # --- Attempt to parse the string response ---
        # This parsing is basic and might break if the output format changes.
        # A more robust approach might involve DuckDuckGoSearchResults tool
        # or regex, but let's try simple splitting.
        # Often results look like: "Snippet 1... [Source: Title 1](link1)"
        import re
        # Simple pattern: Look for markdown-style links often used for sources
        # Extract snippet before link, title within link text, and URL
        pattern = r"\[Source:\s*(.*?)\s*\]\((.*?)\)" # Pattern for [Source: Title](link)
        found_items = re.findall(pattern, search_response_str)

        # Split the main text by the source markers to try and get snippets
        snippets_parts = re.split(pattern, search_response_str)

        if found_items:
            logger.info(f"Attempting to parse {len(found_items)} items from DDG response.")
            for i, (title, link) in enumerate(found_items):
                # Try to associate snippet based on split parts
                snippet = snippets_parts[i].strip() if i < len(snippets_parts) else "Snippet not parsed."
                # Clean up snippet (remove potential leading/trailing noise)
                snippet = snippet.split("\n")[-1].strip() # Take last part before source marker

                results.append({
                    "title": title.strip(),
                    "link": link.strip(),
                    "snippet": snippet if snippet else "Snippet not parsed.",
                    "source": f"DuckDuckGo:{i+1}" # Add source identifier
                })
        elif search_response_str: # If no structured sources found, return the whole blob as one result
             logger.warning("Could not parse structured results from DuckDuckGoSearchRun output. Returning full text as snippet.")
             results.append({
                 "title": f"DuckDuckGo Results for '{query}'",
                 "link": "#",
                 "snippet": search_response_str,
                 "source": "DuckDuckGo:Raw"
             })
        else:
            logger.info("DuckDuckGo search returned no results.")

    except Exception as e:
        logger.error(f"Error during DuckDuckGo search execution: {e}", exc_info=True)
        results = [{"error": f"DuckDuckGo search failed: {e}"}]

    logger.info(f"Formatted {len(results)} DuckDuckGo results.")
    return results

# Remove the old search_google function
# def search_google(...):
#     pass
