import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import central AgentState definition
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# --- Node Function ---

def call_chat_agent(state: AgentState, llm) -> AgentState: # Return full state
    """Chat Agent node: Generates response using history, with improved context prompting."""
    logger.info("--- Calling Chat Agent ---")
    query = state['query']; history = state['history']
    error_message = state.get("error"); chat_response_text = "Sorry, I couldn't generate a response."

    system_prompt = """You are BioAgent, a helpful AI assistant specializing in epidemiology and bioinformatics.
Answer the user's latest query based on the provided conversation history.
If the user's query refers to previous information (like literature search results, summaries, or generated code mentioned earlier in the history), use the conversation history context to find the relevant information and provide the answer.
Be concise and helpful."""

    messages = [SystemMessage(content=system_prompt)]
    for user_msg, ai_msg in history:
        messages.append(HumanMessage(content=str(user_msg)))
        messages.append(AIMessage(content=str(ai_msg)))
    messages.append(HumanMessage(content=query))

    logger.info(f"Received query: {query}"); logger.info(f"Using history (last {len(history)} turns) with system prompt.")
    try:
        response = llm.invoke(messages)
        chat_response_text = response.content.strip()
        logger.info(f"LLM chat response generated.")
    except Exception as e:
        chat_error = f"Chat generation failed: {str(e)}"; logger.error(chat_error, exc_info=True)
        error_message = (error_message + "; " + chat_error) if error_message else chat_error

    # Return the entire state merged with updates
    return {**state, "chat_response": chat_response_text, "error": error_message}

