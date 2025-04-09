import os
import logging
from typing import List, Dict, Any, Optional
import arxiv # Import arxiv library

logger = logging.getLogger(__name__)

def download_arxiv_pdf_tool(arxiv_id: str, pdf_url: str, save_dir: str) -> Optional[str]:
    """
    Downloads a single ArXiv PDF using its ID and saves it.

    Args:
        arxiv_id: The ArXiv ID (e.g., '2409.07376v1').
        pdf_url: The direct PDF URL (used as fallback or verification, not strictly needed for download).
        save_dir: The directory to save the PDF in (e.g., 'workplace/RUN_DIR/results').

    Returns:
        The full path to the saved PDF if successful, otherwise None.
    """
    if not arxiv_id:
        logger.warning("Cannot download ArXiv PDF without an ID.")
        return None

    pdf_filename = f"arxiv_{arxiv_id}.pdf"
    pdf_saveloc = os.path.join(save_dir, pdf_filename)
    local_pdf_path = None

    try:
        logger.info(f"Attempting to download ArXiv PDF: {arxiv_id}")
        # Use arxiv.Client to fetch the specific paper by ID
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))

        if results:
            paper = results[0]
            # Ensure the save directory exists
            os.makedirs(save_dir, exist_ok=True)
            # Download using the result object
            paper.download_pdf(dirpath=save_dir, filename=pdf_filename)
            logger.info(f"Successfully downloaded PDF to: {pdf_saveloc}")
            local_pdf_path = pdf_saveloc
        else:
            logger.error(f"Could not fetch ArXiv result for ID {arxiv_id} to download.")

    except Exception as pdf_e:
        logger.error(f"Failed to download ArXiv PDF {arxiv_id}: {pdf_e}", exc_info=True)

    return local_pdf_path
