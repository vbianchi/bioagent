import os
import logging
from typing import Dict, Any

# Import the download tool
from src.tools.pdf_downloader import download_arxiv_pdf_tool
# Import END from langgraph
from langgraph.graph import END
# Import central AgentState definition
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# --- Node Functions ---

def ask_download_preference(state: AgentState) -> Dict[str, Any]:
    """
    Asks the user if they want to download found ArXiv PDFs (CLI only).
    Defaults to 'no' if running in UI mode (run_dir is None).
    """
    logger.info("--- Asking Download Preference ---")
    arxiv_found = state.get("arxiv_results_found", False)
    run_dir = state.get("run_dir") # Check if run_dir is set
    preference = "no" # Default to no
    error_message = state.get("error") # Preserve existing errors

    # Only prompt if ArXiv papers were found AND we are in CLI mode (run_dir is set)
    if arxiv_found and run_dir is not None:
        try:
            user_input = input("ArXiv papers found. Download PDFs? (yes/no): ").strip().lower()
            if user_input == "yes" or user_input == "y":
                preference = "yes"; logger.info("User chose to download ArXiv PDFs.")
            else: logger.info("User chose not to download ArXiv PDFs.")
        except EOFError: logger.warning("EOF received asking download preference. Defaulting to 'no'."); preference = "no"
        except Exception as e:
             logger.error(f"Error reading download preference: {e}", exc_info=True)
             error_message = (error_message + f"; Error reading download preference: {e}") if error_message else f"Error reading download preference: {e}"
             preference = "no"
    elif arxiv_found and run_dir is None:
         logger.info("Running in UI mode (run_dir is None). Skipping PDF download prompt, defaulting to 'no'.")
         preference = "no"
    else: logger.info("No ArXiv papers found, skipping download prompt."); preference = "no"

    return {"download_preference": preference, "error": error_message}


def download_arxiv_pdfs(state: AgentState) -> Dict[str, Any]:
    """Downloads ArXiv PDFs based on user preference and search results."""
    logger.info("--- Downloading ArXiv PDFs ---")
    run_dir = state.get("run_dir"); search_results = state.get("search_results", [])
    error_message = state.get("error"); download_count = 0; updated_search_results = []

    if not run_dir:
         logger.error("Download node reached but run_dir is not set. Cannot save PDFs.")
         err = "; Cannot download PDFs without run_dir."
         return {"error": (error_message or "") + err}

    results_save_dir = os.path.join(run_dir, "results")

    for result in search_results:
        result_copy = result.copy()
        if result.get("source") == "ArXiv":
            arxiv_id = result.get("id"); pdf_url = result.get("pdf_url")
            if not arxiv_id or not pdf_url: logger.warning(f"Skipping ArXiv result: {result.get('title')}")
            else:
                local_pdf_path = download_arxiv_pdf_tool(arxiv_id=arxiv_id, pdf_url=pdf_url, save_dir=results_save_dir)
                if local_pdf_path: download_count += 1; result_copy["local_pdf_path"] = local_pdf_path
        updated_search_results.append(result_copy)

    logger.info(f"Attempted to download PDFs. Successful downloads: {download_count}")
    return {"search_results": updated_search_results, "error": error_message}


# --- Conditional Edge Logic ---
def should_download(state: AgentState) -> str:
    """Determines the next node after asking download preference."""
    preference = state.get("download_preference", "no")
    intent = state.get("route_intent", "literature_search") # Check original intent

    # Only download if preference is yes AND we are in CLI mode (run_dir is set)
    if preference == "yes" and state.get("run_dir"):
        logger.info("Proceeding to download ArXiv PDFs.")
        return "download_arxiv_pdfs"
    else:
        if preference == "yes": logger.warning("Download preference is 'yes' but run_dir not set (UI mode?). Skipping download.")
        else: logger.info("Skipping ArXiv PDF download.")

        # Decide next step based on original intent
        if intent == "deep_research":
            return "google_search" # Go to google search after skipping download
        else:
            return "summarizer" # Go to summarizer for normal literature search

