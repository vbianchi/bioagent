import logging
from typing import Dict, Any, List, Optional

# Import central AgentState definition
from src.core.state import AgentState
# Import config loading function to get settings within the node
from src.core.config_loader import get_config_value

logger = logging.getLogger(__name__)

# --- Helper Function ---
def format_results_for_synthesis(
        literature_results: Optional[List[Dict[str, Any]]],
        google_results: Optional[List[Dict[str, Any]]],
        max_lit: int = 5,
        max_web: int = 5
    ) -> str:
    # (Unchanged)
    content = []; content.append("=== Literature Search Results (PubMed/ArXiv) ==="); lit_count = 0
    if literature_results:
        for res in literature_results:
            if lit_count >= max_lit: break
            if res.get("abstract") and res.get("abstract") != "No abstract found": content.append(f"\n--- Literature Result {lit_count+1} ---"); content.append(f"Source: {res.get('source', 'N/A')} - ID: {res.get('id', 'N/A')}"); content.append(f"Title: {res.get('title', 'N/A')}"); content.append(f"Abstract: {res.get('abstract', 'N/A')[:500]}..."); lit_count += 1
    if lit_count == 0: content.append("No relevant literature results with abstracts found or provided.")
    content.append("\n=== Web Search Results (Google) ==="); web_count = 0
    if google_results:
        for res in google_results:
            if web_count >= max_web: break
            content.append(f"\n--- Web Result {web_count+1} ---"); content.append(f"Title: {res.get('title', 'N/A')}"); content.append(f"Link: {res.get('link', '#')}"); content.append(f"Snippet: {res.get('snippet', 'N/A')}"); web_count += 1
    if web_count == 0: content.append("No web search results found or provided.")
    return "\n".join(content)

# --- Node Function ---
def synthesize_results_agent(state: AgentState, llm, synthesis_prompt_template: str, app_config: dict) -> AgentState: # Return full state
    """
    Synthesizes information from literature and web searches into a report.
    """
    logger.info("--- Calling Synthesis Agent ---")
    # (Logging to check config removed for brevity, assuming fix worked)
    original_query = state['query']; literature_results = state.get("search_results"); google_results = state.get("google_results")
    error_message = state.get("error"); report = "Could not generate synthesis report."

    if not literature_results and not google_results:
        logger.warning("No results from any source available for synthesis.")
        return {**state, "synthesized_report": "No information found to synthesize.", "error": error_message}

    max_lit_ctx = get_config_value(app_config, "search_settings.max_abstracts_to_summarize", 3)
    max_web_ctx = get_config_value(app_config, "search_settings.num_google_results", 5)
    formatted_data = format_results_for_synthesis(literature_results, google_results, max_lit=max_lit_ctx, max_web=max_web_ctx)

    if "Error:" in synthesis_prompt_template:
         logger.error("Synthesis prompt template not loaded correctly.")
         return {**state, "synthesized_report": report, "error": (error_message or "") + "; Config error"}

    synthesis_prompt = synthesis_prompt_template.format(query=original_query, search_results_text=formatted_data)
    logger.info("Sending combined results to LLM for synthesis...")
    try:
        response = llm.invoke(synthesis_prompt); report = response.content.strip(); logger.info("LLM Synthesis complete.")
    except Exception as e:
        synth_error = f"Synthesis failed: {str(e)}"; logger.error(synth_error, exc_info=True)
        error_message = (error_message + "; " + synth_error) if error_message else synth_error
        report = "Sorry, I failed to synthesize the results."

    # Return the entire state merged with updates
    return {**state, "synthesized_report": report, "error": error_message}
