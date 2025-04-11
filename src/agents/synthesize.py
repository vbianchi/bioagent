import logging
from typing import Dict, Any, List, Optional

# Try importing the central AgentState definition
try:
    from src.core.state import AgentState
except ImportError:
    # Fallback definition if import fails
    from typing import TypedDict, List, Dict, Any, Optional, Tuple # Keep import here
    logger.warning("Could not import AgentState from src.core.state, using fallback definition in synthesize.py.")
    class AgentState(TypedDict): # <<< Moved class definition to new line
        query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
        search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
        chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]
        run_dir: Optional[str]; arxiv_results_found: bool; download_preference: Optional[str]
        code_request: Optional[str]; generated_code: Optional[str]
        generated_code_language: Optional[str]; google_results: Optional[List[Dict[str, Any]]]
        synthesized_report: Optional[str]; route_intent: Optional[str] # Ensure all fields

logger = logging.getLogger(__name__)

# --- Helper Function ---
def format_results_for_synthesis(
        literature_results: Optional[List[Dict[str, Any]]],
        google_results: Optional[List[Dict[str, Any]]],
        max_lit: int = 5, # Max literature results to include
        max_web: int = 5 # Max web results to include
    ) -> str:
    """Formats combined results into a string for the synthesis prompt."""
    content = []
    content.append("=== Literature Search Results (PubMed/ArXiv) ===")
    lit_count = 0
    if literature_results:
        for res in literature_results:
            if lit_count >= max_lit: break
            # Include only results with abstracts for synthesis context
            if res.get("abstract") and res.get("abstract") != "No abstract found":
                content.append(f"Source: {res.get('source', 'N/A')} - ID: {res.get('id', 'N/A')}")
                content.append(f"Title: {res.get('title', 'N/A')}")
                content.append(f"Abstract: {res.get('abstract', 'N/A')[:500]}...") # Limit abstract length
                content.append("---")
                lit_count += 1
    if lit_count == 0:
        content.append("No relevant literature results with abstracts found or provided.")

    content.append("\n=== Web Search Results (Google) ===")
    web_count = 0
    if google_results:
        for res in google_results:
            if web_count >= max_web: break
            content.append(f"Title: {res.get('title', 'N/A')}")
            content.append(f"Link: {res.get('link', '#')}")
            content.append(f"Snippet: {res.get('snippet', 'N/A')}")
            content.append("---")
            web_count += 1
    if web_count == 0:
        content.append("No web search results found or provided.")

    return "\n".join(content)

# --- Node Function ---
def synthesize_results_agent(state: AgentState, llm, synthesis_prompt_template: str) -> Dict[str, Any]:
    """
    Synthesizes information from literature and web searches into a report.
    """
    logger.info("--- Calling Synthesis Agent ---")
    original_query = state['query']
    literature_results = state.get("search_results")
    google_results = state.get("google_results")
    error_message = state.get("error") # Preserve previous errors
    report = "Could not generate synthesis report." # Default

    if not literature_results and not google_results:
        logger.warning("No results from any source available for synthesis.")
        return {"synthesized_report": "No information found from any source to synthesize.", "error": error_message}

    # Format the combined results for the LLM prompt
    # Use settings for max items to include in context
    max_lit_ctx = get_config_value(config, "search_settings.max_abstracts_to_summarize", 3) # Reuse summary limit
    max_web_ctx = get_config_value(config, "search_settings.num_google_results", 5) # Use google result limit
    formatted_data = format_results_for_synthesis(literature_results, google_results, max_lit=max_lit_ctx, max_web=max_web_ctx)

    # Create the synthesis prompt
    if "Error:" in synthesis_prompt_template:
         logger.error("Synthesis prompt template not loaded correctly from config.")
         return {"synthesized_report": report, "error": (error_message or "") + "; Config error: Synthesis prompt missing."}

    synthesis_prompt = synthesis_prompt_template.format(
        query=original_query,
        search_results_text=formatted_data
    )

    logger.info("Sending combined results to LLM for synthesis...")
    try:
        response = llm.invoke(synthesis_prompt) # Use main llm
        report = response.content.strip()
        logger.info("LLM Synthesis complete.")
    except Exception as e:
        synth_error = f"Synthesis failed: {str(e)}"
        logger.error(synth_error, exc_info=True)
        error_message = (error_message + "; " + synth_error) if error_message else synth_error
        report = "Sorry, I failed to synthesize the results."

    # Return only updated fields
    return {"synthesized_report": report, "error": error_message}

