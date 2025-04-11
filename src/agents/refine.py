import logging
from typing import Dict, Any

# Try importing the central AgentState definition
try:
    from src.core.state import AgentState
except ImportError:
    # Fallback definition if import fails
    from typing import TypedDict, List, Dict, Any, Optional, Tuple # Keep import here
    logger.warning("Could not import AgentState from src.core.state, using fallback definition in refine.py.")
    class AgentState(TypedDict): # <<< Moved class definition to new line
        query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
        search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
        chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]
        run_dir: Optional[str]; arxiv_results_found: bool; download_preference: Optional[str]
        code_request: Optional[str]; generated_code: Optional[str]
        generated_code_language: Optional[str]; google_results: Optional[List[Dict[str, Any]]]
        synthesized_report: Optional[str]; route_intent: Optional[str] # Ensure all fields

logger = logging.getLogger(__name__)

# --- Node Function ---

def refine_query_node(state: AgentState, llm, refinement_prompt_template: str) -> Dict[str, Any]:
    """
    Refines the user's original query into search terms using an LLM.
    """
    logger.info("--- Calling Query Refiner ---")
    original_query = state['query']
    logger.info(f"Received original query: {original_query}")
    error_message = state.get("error") # Preserve existing errors
    refined_query_for_search = original_query # Default

    # Ensure prompt template is valid
    if "Error:" in refinement_prompt_template:
         logger.error("Refinement prompt template not loaded correctly from config.")
         return {"refined_query": refined_query_for_search, "error": (error_message or "") + "; Config error: Refinement prompt missing."}

    try:
        logger.info("Refining query for search...")
        refinement_prompt = refinement_prompt_template.format(query=original_query)
        refinement_response = llm.invoke(refinement_prompt)
        refined_query_for_search = refinement_response.content.strip()
        # Handle potential empty refinement
        if not refined_query_for_search:
            logger.warning("LLM refinement resulted in empty query. Using original.")
            refined_query_for_search = original_query
        else:
            logger.info(f"Refined query: {refined_query_for_search}")
    except Exception as refine_e:
        logger.warning(f"LLM query refinement failed: {refine_e}. Using original query.")
        refine_error = f"Query refinement failed: {refine_e}. "
        error_message = (error_message + "; " + refine_error) if error_message else refine_error
        refined_query_for_search = original_query # Ensure fallback on error

    # Return only updated fields
    return {"refined_query": refined_query_for_search, "error": error_message}

