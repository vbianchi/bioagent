import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

# Try importing the central AgentState definition
try:
    from src.core.state import AgentState
except ImportError:
    # Fallback definition if import fails
    from typing import TypedDict, List, Dict, Any, Optional, Tuple # Keep import here
    logger.warning("Could not import AgentState from src.core.state, using fallback definition in chat.py.")
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

def call_chat_agent(state: AgentState, llm) -> Dict[str, Any]:
    """Chat Agent node: Generates response using history."""
    logger.info("--- Calling Chat Agent ---")
    query = state['query']
    history = state['history']
    error_message = state.get("error") # Preserve existing errors
    chat_response_text = "Sorry, I couldn't generate a response."

    # Format message list for LLM
    formatted_history = []
    for user_msg, ai_msg in history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    formatted_history.append(HumanMessage(content=query)) # Add current query

    logger.info(f"Received query: {query}")
    logger.info(f"Using history (last {len(history)} turns)")
    try:
        response = llm.invoke(formatted_history)
        chat_response_text = response.content.strip()
        logger.info(f"LLM chat response generated.")
    except Exception as e:
        chat_error = f"Chat generation failed: {str(e)}"
        logger.error(chat_error, exc_info=True)
        error_message = (error_message + "; " + chat_error) if error_message else chat_error

    # Return only updated fields
    return {"chat_response": chat_response_text, "error": error_message}

