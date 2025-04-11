import logging
from typing import TypedDict, List, Dict, Any, Optional, Tuple

# Import END from langgraph - needs to be accessible
from langgraph.graph import END

# Try importing the central AgentState definition
try:
    from src.core.state import AgentState
except ImportError:
    # Fallback definition if import fails (ensure AgentState structure matches core/state.py)
    from typing import TypedDict, List, Dict, Any, Optional, Tuple # Keep import here
    logger.warning("Could not import AgentState from src.core.state, using fallback definition in router.py.")
    class AgentState(TypedDict): # <<< Moved class definition to new line
        query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
        search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
        chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]
        run_dir: Optional[str]; arxiv_results_found: bool; download_preference: Optional[str]
        code_request: Optional[str]; generated_code: Optional[str]
        generated_code_language: Optional[str]; google_results: Optional[List[Dict[str, Any]]]
        synthesized_report: Optional[str] # Ensure all latest fields are here

logger = logging.getLogger(__name__)

# --- Node Functions ---

def route_query(state: AgentState, llm, routing_prompt_template: str) -> Dict[str, Any]:
    """Router node: Classifies query and determines next node."""
    logger.info("--- Calling Router ---")
    query = state['query']
    logger.info(f"Routing query: {query}")
    # Ensure the prompt template exists before formatting
    if "Error:" in routing_prompt_template:
         logger.error("Routing prompt template not loaded correctly from config.")
         return {"next_node": "chat_agent", "error": "Configuration error: Routing prompt missing."}

    prompt = routing_prompt_template.format(query=query)
    next_node_decision = "chat_agent" # Default
    error_message = state.get("error") # Preserve existing errors

    try:
        response = llm.invoke(prompt)
        classification = response.content.strip().lower().replace("'", "").replace('"', '')
        logger.info(f"LLM Classification: {classification}")

        # Map classification to the next node
        if classification == "literature_search":
            logger.info("Routing to Refine Query (for Literature Search).")
            next_node_decision = "refine_query"
        elif classification == "code_generation":
            logger.info("Routing to Coding Agent.")
            next_node_decision = "coding_agent"
        elif classification == "deep_research": # New route
            logger.info("Routing to Refine Query (for Deep Research).")
            next_node_decision = "refine_query"
        elif classification == "chat":
             logger.info("Routing to Chat Agent.")
             next_node_decision = "chat_agent"
        else:
            logger.warning(f"Unexpected classification '{classification}'. Defaulting to Chat Agent.")
            next_node_decision = "chat_agent"

    except Exception as e:
        route_error = f"Routing error: {e}"
        logger.error(f"Error during LLM routing: {e}. Defaulting to Chat Agent.", exc_info=True)
        error_message = (error_message + "; " + route_error) if error_message else route_error
        next_node_decision = "chat_agent" # Default to chat on error

    # Return only updated fields for LangGraph to merge
    # Store the original classification intent if routing to refine_query
    route_intent = None
    if next_node_decision == "refine_query":
        # Infer original intent based on classification that led here
        if classification == "literature_search": route_intent = "literature_search"
        elif classification == "deep_research": route_intent = "deep_research"
        # Add route_intent to AgentState definition if not already present

    # Update AgentState definition to include route_intent: Optional[str]
    # (This definition should ideally be central)
    class AgentState(TypedDict): # Redefine locally if needed for clarity
        query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
        search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
        chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]
        run_dir: Optional[str]; arxiv_results_found: bool; download_preference: Optional[str]
        code_request: Optional[str]; generated_code: Optional[str]
        generated_code_language: Optional[str]; google_results: Optional[List[Dict[str, Any]]]
        synthesized_report: Optional[str]; route_intent: Optional[str] # Added route_intent

    return {"next_node": next_node_decision, "route_intent": route_intent, "error": error_message}


# --- Conditional Edge Logic ---

def decide_next_node(state: AgentState) -> str:
    """Determines the next node after the router."""
    next_node = state.get("next_node")
    # Add refine_query as a valid starting point from router
    valid_nodes = ["refine_query", "chat_agent", "coding_agent"]
    if next_node not in valid_nodes:
        logger.warning(f"Invalid next_node value '{next_node}' after router. Ending.")
        return END
    logger.debug(f"Decided next node after router: {next_node}")
    return next_node

# Conditional edge logic after refinement
def decide_after_refine(state: AgentState) -> str:
    """Determines path after query refinement based on original route intent."""
    intent = state.get("route_intent", "literature_search") # Default if missing
    logger.debug(f"Deciding path after refine based on intent: {intent}")
    if intent == "literature_search":
        return "literature_agent"
    elif intent == "deep_research":
        # Start deep research by first doing literature search
        return "literature_agent"
    else:
        logger.warning(f"Unexpected intent '{intent}' after refine query node. Ending.")
        return END
