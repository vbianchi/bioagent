import logging
from typing import Dict, Any, List, Optional, Tuple # Added List, Optional, Tuple for AgentState definition

# Import central AgentState definition
try:
    from src.core.state import AgentState
except ImportError:
    # Fallback definition if import fails
    # from typing import TypedDict, List, Dict, Any, Optional, Tuple # Keep import here
    logger = logging.getLogger(__name__) # Need logger defined for warning
    logger.warning("Could not import AgentState from src.core.state, using fallback definition in summarize.py.")
    # Define AgentState locally ONLY IF import fails (ensure structure matches core/state.py)
    class AgentState(TypedDict):
        query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
        search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
        chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]
        run_dir: Optional[str]; arxiv_results_found: bool; download_preference: Optional[str]
        code_request: Optional[str]; generated_code: Optional[str]
        generated_code_language: Optional[str]; google_results: Optional[List[Dict[str, Any]]]
        synthesized_report: Optional[str]; route_intent: Optional[str]

# Remove the faulty import of get_config_value and config
# from src.core.config_loader import get_config_value, config # REMOVED THIS LINE

logger = logging.getLogger(__name__)

# --- Node Function ---

def summarize_results(state: AgentState, llm, summarization_prompt_template: str, max_abstracts: int) -> Dict[str, Any]:
    """Summarizer node: Summarizes abstracts from literature search results."""
    logger.info("--- Calling Summarizer ---")
    search_results = state.get("search_results")
    original_query = state.get("query")
    error_message = state.get("error") # Preserve existing errors
    summary_text = None

    # Filter for literature results only
    literature_results = [res for res in search_results if res.get("source") in ["PubMed", "ArXiv"]] if search_results else []

    if not literature_results:
        logger.info("No literature search results to summarize.")
        summary_text = "No literature results found to summarize."
        # Return only updated fields
        return {"summary": summary_text, "error": error_message}

    abstracts_to_summarize = []
    logger.info(f"Preparing abstracts for summarization (max {max_abstracts})...")
    count = 0
    for result in literature_results:
        if count >= max_abstracts: break
        abstract = result.get("abstract")
        if abstract and abstract != "No abstract found":
            abstracts_to_summarize.append(f"Abstract {count+1} (Source: {result.get('source', 'N/A')}, ID: {result.get('id', 'N/A')}):\n{abstract}\n")
            count += 1

    if not abstracts_to_summarize:
        logger.info("No valid abstracts found in literature results to summarize.")
        return {"summary": "No abstracts available to summarize.", "error": error_message}

    abstracts_text = "\n---\n".join(abstracts_to_summarize)
    # Use prompt template passed via partial
    if "Error:" in summarization_prompt_template:
         logger.error("Summarization prompt template not loaded correctly from config.")
         return {"summary": "Configuration error: Summarization prompt missing.", "error": (error_message or "") + "; Config error: Summarization prompt missing."}

    summarization_prompt = summarization_prompt_template.format(query=original_query, abstracts_text=abstracts_text)

    logger.info(f"Sending {len(abstracts_to_summarize)} abstracts to LLM for summarization...")
    try:
        response = llm.invoke(summarization_prompt)
        summary_text = response.content.strip()
        logger.info("LLM Summary generated.")
    except Exception as e:
        summary_error = f"Summarization failed: {str(e)}"
        logger.error(summary_error, exc_info=True)
        error_message = (error_message + "; " + summary_error) if error_message else summary_error
        summary_text = "Sorry, I couldn't generate a summary."

    # Return only updated fields
    return {"summary": summary_text, "error": error_message}
