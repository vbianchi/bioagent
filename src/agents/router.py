import logging
from typing import Dict, Any

# Import END from langgraph
from langgraph.graph import END
# Import central AgentState definition
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# --- Node Functions ---

def route_query(state: AgentState, llm, routing_prompt_template: str) -> AgentState: # Return full state
    """Router node: Classifies query, determines next node, and stores intent."""
    logger.info("--- Calling Router ---")
    query = state['query']
    logger.info(f"Routing query: {query}")

    if "Error:" in routing_prompt_template:
         logger.error("Routing prompt template not loaded correctly.")
         # Update state with error and default next node/intent
         return {**state, "next_node": "chat_agent", "route_intent": "chat", "error": "Config error: Routing prompt missing."}

    prompt = routing_prompt_template.format(query=query)
    next_node_decision = "chat_agent"; route_intent_decision = "chat"; error_message = state.get("error")

    try:
        response = llm.invoke(prompt)
        classification = response.content.strip().lower().replace("'", "").replace('"', '')
        logger.info(f"LLM Classification: {classification}")
        if classification == "literature_search":
            logger.info("Routing to Refine Query (for Literature Search).")
            next_node_decision = "refine_query"; route_intent_decision = "literature_search"
        elif classification == "code_generation":
            logger.info("Routing to Coding Agent."); next_node_decision = "coding_agent"; route_intent_decision = "code_generation"
        elif classification == "deep_research":
            logger.info("Routing to Refine Query (for Deep Research)."); next_node_decision = "refine_query"; route_intent_decision = "deep_research"
        elif classification == "chat":
             logger.info("Routing to Chat Agent."); next_node_decision = "chat_agent"; route_intent_decision = "chat"
        else: logger.warning(f"Unexpected classification '{classification}'. Defaulting to Chat Agent.")
    except Exception as e:
        route_error = f"Routing error: {e}"; logger.error(f"Error during LLM routing: {e}. Defaulting to Chat Agent.", exc_info=True)
        error_message = (error_message + "; " + route_error) if error_message else route_error
        next_node_decision = "chat_agent"; route_intent_decision = "chat"

    # Return the entire state merged with updates
    return {**state, "next_node": next_node_decision, "route_intent": route_intent_decision, "error": error_message}

# --- Conditional Edge Logic ---
# (These don't need AgentState return type hint, they return strings)
def decide_next_node(state: AgentState) -> str:
    """Determines the next node after the router."""
    next_node = state.get("next_node")
    valid_nodes = ["refine_query", "chat_agent", "coding_agent"]
    if next_node not in valid_nodes: logger.warning(f"Invalid next_node value '{next_node}' after router. Ending."); return END
    logger.debug(f"Decided next node after router: {next_node}"); return next_node

def decide_after_refine(state: AgentState) -> str:
    """Determines path after query refinement based on original route intent."""
    intent = state.get("route_intent", "literature_search"); logger.debug(f"Deciding path after refine based on intent: {intent}")
    if intent in ["literature_search", "deep_research"]: return "literature_agent"
    else: logger.warning(f"Unexpected intent '{intent}' after refine query node. Ending."); return END

def decide_after_summary(state: AgentState) -> str:
    """Decide whether to continue to Google Search (for deep research) or end."""
    intent = state.get("route_intent")
    if intent == "deep_research": logger.info("Continuing to Google Search for Deep Research."); return "google_search"
    else: logger.info("Ending after Literature Summary."); return END
