import logging
from io import StringIO
from typing import List, Dict, Any

# Import BioPython and ArXiv
from Bio import Entrez, Medline
import arxiv

logger = logging.getLogger(__name__)

# --- Helper Functions for Literature Search ---

def search_pubmed(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Helper function to search PubMed and return formatted results."""
    logger.info(f"Searching PubMed for '{query}' (max_results={max_results})...")
    results = []
    try:
        # Use Entrez email set globally in main.py
        handle_search = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        search_results_entrez = Entrez.read(handle_search)
        handle_search.close()
        id_list = search_results_entrez["IdList"]
        if not id_list:
            logger.info("No results found on PubMed.")
            return []

        logger.info(f"Found {len(id_list)} PMIDs on PubMed. Fetching details...")
        handle_fetch = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records_text = handle_fetch.read()
        handle_fetch.close()
        records = Medline.parse(StringIO(records_text))
        count = 0
        for record in records:
            pmid = record.get("PMID", "N/A")
            # Skip if PMID is somehow missing after fetch
            if pmid == "N/A": continue
            results.append({
                "id": pmid, "source": "PubMed",
                "title": record.get("TI", "No title found"),
                "abstract": record.get("AB", "No abstract found"),
                "journal": record.get("JT", "N/A"),
                "authors": record.get("AU", []),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
            count += 1
        logger.info(f"Successfully fetched/parsed {count} PubMed records.")
    except Exception as e:
        logger.error(f"Error during PubMed search: {e}", exc_info=True)
    return results

def search_arxiv(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Helper function to search ArXiv and return metadata."""
    logger.info(f"Searching ArXiv for '{query}' (max_results={max_results})...")
    results = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        arxiv_results = list(client.results(search))
        if not arxiv_results:
            logger.info("No results found on ArXiv.")
            return []

        logger.info(f"Found {len(arxiv_results)} results on ArXiv.")
        for result in arxiv_results:
            arxiv_id = result.entry_id.split('/')[-1]
            results.append({
                "id": arxiv_id, "source": "ArXiv",
                "title": result.title,
                "abstract": result.summary.replace('\n', ' '),
                "authors": [str(author) for author in result.authors],
                "published": str(result.published),
                "url": result.pdf_url, # Use pdf_url as the primary URL
                "pdf_url": result.pdf_url # Keep pdf_url explicitly for download logic
            })
    except Exception as e:
        logger.error(f"Error during ArXiv search: {e}", exc_info=True)
    return results

