import logging
from typing import TypedDict, List, Dict, Any, Optional, Tuple

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

# --- Node Function ---

def summarize_results(state: AgentState, llm, summarization_prompt_template: str, max_abstracts: int) -> AgentState:
    """Summarizer node: Summarizes abstracts from search results."""
    logger.info("--- Calling Summarizer ---")
    search_results = state.get("search_results")
    original_query = state.get("query")
    error_message = state.get("error") # Preserve existing errors
    summary_text = None

    if not search_results:
        logger.info("No search results to summarize.")
        return {**state, "summary": None} # Pass through state

    abstracts_to_summarize = []
    logger.info(f"Preparing abstracts for summarization (max {max_abstracts})...")
    count = 0
    for result in search_results:
        if count >= max_abstracts: break
        abstract = result.get("abstract")
        if abstract and abstract != "No abstract found":
            abstracts_to_summarize.append(f"Abstract {count+1} (Source: {result.get('source', 'N/A')}, ID: {result.get('id', 'N/A')}):\n{abstract}\n")
            count += 1

    if not abstracts_to_summarize:
        logger.info("No valid abstracts found in results to summarize.")
        return {**state, "summary": "No abstracts available to summarize."}

    abstracts_text = "\n---\n".join(abstracts_to_summarize)
    # Use prompt template from config
    summarization_prompt = summarization_prompt_template.format(query=original_query, abstracts_text=abstracts_text)

    logger.info(f"Sending {len(abstracts_to_summarize)} abstracts to LLM for summarization...")
    try:
        response = llm.invoke(summarization_prompt) # Use main llm passed as argument
        summary_text = response.content.strip()
        logger.info("LLM Summary generated.")
    except Exception as e:
        summary_error = f"Summarization failed: {str(e)}"
        logger.error(summary_error, exc_info=True)
        error_message = (error_message + "; " + summary_error) if error_message else summary_error
        summary_text = "Sorry, I couldn't generate a summary."

    # Pass through state and add summary/error
    return {**state, "summary": summary_text, "error": error_message}

