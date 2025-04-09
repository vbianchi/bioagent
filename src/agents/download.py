import os
import logging
from typing import TypedDict, List, Dict, Any, Optional, Tuple

# Import the download tool
from src.tools.pdf_downloader import download_arxiv_pdf_tool

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

# --- Node Functions ---

def ask_download_preference(state: AgentState) -> AgentState:
    """Asks the user if they want to download found ArXiv PDFs."""
    logger.info("--- Asking Download Preference ---")
    arxiv_found = state.get("arxiv_results_found", False)
    preference = "no" # Default to no
    error_message = state.get("error") # Preserve existing errors

    if arxiv_found:
        try:
            # Use colorama for the prompt (imported in main.py)
            # Need to access colors - better to just print plain here
            # Or pass colors/use a dedicated UI class later
            user_input = input("ArXiv papers found. Download PDFs? (yes/no): ").strip().lower()
            if user_input == "yes" or user_input == "y":
                preference = "yes"
                logger.info("User chose to download ArXiv PDFs.")
            else:
                logger.info("User chose not to download ArXiv PDFs.")
        except EOFError:
            logger.warning("EOF received while asking download preference. Defaulting to 'no'.")
            preference = "no"
        except Exception as e:
             logger.error(f"Error reading download preference: {e}", exc_info=True)
             error_message = (error_message + f"; Error reading download preference: {e}") if error_message else f"Error reading download preference: {e}"
             preference = "no" # Default to no on error
    else:
        logger.info("No ArXiv papers found, skipping download prompt.")
        preference = "no"

    return {**state, "download_preference": preference, "error": error_message}


def download_arxiv_pdfs(state: AgentState) -> AgentState:
    """Downloads ArXiv PDFs based on user preference and search results."""
    logger.info("--- Downloading ArXiv PDFs ---")
    run_dir = state['run_dir']
    search_results = state.get("search_results", [])
    results_save_dir = os.path.join(run_dir, "results")
    download_count = 0
    error_message = state.get("error") # Preserve existing errors

    # Create a new list for potentially updated results
    updated_search_results = []

    for result in search_results:
        # Process only ArXiv results
        if result.get("source") == "ArXiv":
            arxiv_id = result.get("id")
            pdf_url = result.get("pdf_url") # Get URL from original search result

            if not arxiv_id or not pdf_url:
                 logger.warning(f"Skipping ArXiv result due to missing ID or PDF URL: {result.get('title')}")
                 updated_search_results.append(result) # Keep original result
                 continue

            # Call the download tool
            local_pdf_path = download_arxiv_pdf_tool(
                arxiv_id=arxiv_id,
                pdf_url=pdf_url,
                save_dir=results_save_dir
            )

            if local_pdf_path:
                download_count += 1

            # Update the result dictionary with the local path (or None)
            result_copy = result.copy()
            result_copy["local_pdf_path"] = local_pdf_path
            updated_search_results.append(result_copy)
        else:
             # Add non-ArXiv results to the updated list unchanged
             updated_search_results.append(result)

    logger.info(f"Attempted to download PDFs. Successful downloads: {download_count}")

    # Return state with potentially updated search results list
    return {**state, "search_results": updated_search_results, "error": error_message}


# --- Conditional Edge Logic ---

def should_download(state: AgentState) -> str:
    """Determines the next node after asking download preference."""
    preference = state.get("download_preference", "no")
    if preference == "yes":
        logger.info("Proceeding to download ArXiv PDFs.")
        return "download_arxiv_pdfs"
    else:
        logger.info("Skipping ArXiv PDF download.")
        return "summarizer" # Go directly to summarizer if no download

