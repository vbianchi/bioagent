import logging
from typing import Dict, Any

# Import central AgentState definition
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# --- Node Function ---

def refine_query_node(state: AgentState, llm, refinement_prompt_template: str) -> AgentState: # Return full state
    """
    Refines the user's original query into search terms using an LLM.
    """
    logger.info("--- Calling Query Refiner ---")
    original_query = state['query']
    logger.info(f"Received original query: {original_query}")
    error_message = state.get("error") # Preserve existing errors
    refined_query_for_search = original_query # Default

    if "Error:" in refinement_prompt_template:
         logger.error("Refinement prompt template not loaded correctly.")
         return {**state, "refined_query": refined_query_for_search, "error": (error_message or "") + "; Config error"}

    try:
        logger.info("Refining query for search...")
        refinement_prompt = refinement_prompt_template.format(query=original_query)
        refinement_response = llm.invoke(refinement_prompt)
        refined_query_for_search = refinement_response.content.strip()
        if not refined_query_for_search:
            logger.warning("LLM refinement resulted in empty query. Using original query.")
            refined_query_for_search = original_query
        else: logger.info(f"Refined query: {refined_query_for_search}")
    except Exception as refine_e:
        logger.warning(f"LLM query refinement failed: {refine_e}. Using original query.")
        refine_error = f"Query refinement failed: {refine_e}. "
        error_message = (error_message + "; " + refine_error) if error_message else refine_error
        refined_query_for_search = original_query # Ensure fallback on error

    # Return the entire state merged with updates
    return {**state, "refined_query": refined_query_for_search, "error": error_message}

