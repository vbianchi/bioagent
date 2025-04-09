import logging
from typing import TypedDict, List, Dict, Any, Optional, Tuple

# Assuming AgentState is defined centrally or passed appropriately
# For now, let's redefine it here based on the last version in main.py
# TODO: Define AgentState in a shared location later
class AgentState(TypedDict):
    query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
    chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]
    run_dir: str; arxiv_results_found: bool; download_preference: Optional[str]
    code_request: Optional[str]; generated_code: Optional[str]
    generated_code_language: Optional[str]

logger = logging.getLogger(__name__)

# --- Node Functions ---

def route_query(state: AgentState, llm, routing_prompt_template: str) -> AgentState:
    """Router node: Classifies query and determines next node."""
    logger.info("--- Calling Router ---")
    query = state['query']
    logger.info(f"Routing query: {query}")
    prompt = routing_prompt_template.format(query=query)
    next_node_decision = "chat_agent" # Default
    error_message = state.get("error") # Preserve existing errors

    try:
        response = llm.invoke(prompt)
        classification = response.content.strip().lower().replace("'", "").replace('"', '')
        logger.info(f"LLM Classification: {classification}")

        if classification == "literature_search":
            logger.info("Routing to Literature Agent.")
            next_node_decision = "literature_agent"
        elif classification == "code_generation":
            logger.info("Routing to Coding Agent.")
            next_node_decision = "coding_agent"
        elif classification == "chat":
             logger.info("Routing to Chat Agent.")
             next_node_decision = "chat_agent"
        else:
            logger.warning(f"Unexpected classification '{classification}'. Defaulting to Chat Agent.")
            # Keep default decision 'chat_agent'

    except Exception as e:
        route_error = f"Routing error: {e}"
        logger.error(f"Error during LLM routing: {e}. Defaulting to Chat Agent.", exc_info=True)
        error_message = (error_message + "; " + route_error) if error_message else route_error
        next_node_decision = "chat_agent" # Default to chat on error

    # Return the *entire state* merged with updates
    return {**state, "next_node": next_node_decision, "error": error_message}


# --- Conditional Edge Logic ---

def decide_next_node(state: AgentState) -> str:
    """Determines the next node after the router."""
    next_node = state.get("next_node")
    valid_nodes = ["literature_agent", "chat_agent", "coding_agent"]
    if next_node not in valid_nodes:
        logger.warning(f"Invalid next_node value '{next_node}' after router. Ending.")
        return END # Use END constant from langgraph.graph
    logger.debug(f"Decided next node: {next_node}")
    return next_node

