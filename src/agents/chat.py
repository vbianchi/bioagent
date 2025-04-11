import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

# Import central AgentState definition
from src.core.state import AgentState

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

