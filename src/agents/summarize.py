import logging
from typing import Dict, Any

# Import central AgentState definition
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# --- Node Function ---

def summarize_results(state: AgentState, llm, summarization_prompt_template: str, max_abstracts: int) -> AgentState: # Return full state
    """Summarizer node: Summarizes abstracts from literature search results."""
    logger.info("--- Calling Summarizer ---")
    search_results = state.get("search_results"); original_query = state.get("query")
    error_message = state.get("error"); summary_text = None

    literature_results = [res for res in search_results if res.get("source") in ["PubMed", "ArXiv"]] if search_results else []

    if not literature_results:
        logger.info("No literature search results to summarize.")
        summary_text = "No literature results found to summarize."
        return {**state, "summary": summary_text, "error": error_message} # Return full state

    abstracts_to_summarize = []; count = 0
    logger.info(f"Preparing abstracts for summarization (max {max_abstracts})...")
    for result in literature_results:
        if count >= max_abstracts: break
        abstract = result.get("abstract");
        if abstract and abstract != "No abstract found": abstracts_to_summarize.append(f"Abstract {count+1} (Source: {result.get('source', 'N/A')}, ID: {result.get('id', 'N/A')}):\n{abstract}\n"); count += 1

    if not abstracts_to_summarize:
        logger.info("No valid abstracts found in literature results to summarize.")
        return {**state, "summary": "No abstracts available to summarize.", "error": error_message} # Return full state

    abstracts_text = "\n---\n".join(abstracts_to_summarize)
    if "Error:" in summarization_prompt_template:
         logger.error("Summarization prompt template not loaded correctly.")
         return {**state, "summary": "Config error: Summarization prompt missing.", "error": (error_message or "") + "; Config error"}

    summarization_prompt = summarization_prompt_template.format(query=original_query, abstracts_text=abstracts_text)
    logger.info(f"Sending {len(abstracts_to_summarize)} abstracts to LLM for summarization...")
    try:
        response = llm.invoke(summarization_prompt); summary_text = response.content.strip(); logger.info("LLM Summary generated.")
    except Exception as e:
        summary_error = f"Summarization failed: {str(e)}"; logger.error(summary_error, exc_info=True)
        error_message = (error_message + "; " + summary_error) if error_message else summary_error
        summary_text = "Sorry, I couldn't generate a summary."

    # Return the entire state merged with updates
    return {**state, "summary": summary_text, "error": error_message}

