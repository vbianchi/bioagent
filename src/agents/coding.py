import logging
import re
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import central AgentState definition
from src.core.state import AgentState

logger = logging.getLogger(__name__)

# --- Node Function ---

def call_coding_agent(state: AgentState, coding_llm, code_generation_prompt_template: str) -> Dict[str, Any]:
    """Coding Agent: Generates code, using history message list, detects language."""
    logger.info("--- Calling Coding Agent ---")
    query = state['query']; history = state['history']
    error_message = state.get("error"); generated_code_text = None; detected_language = "text"
    logger.info(f"Received code request: {query}")

    # Format message list for LLM, including System Prompt
    # Ensure prompt template is valid
    if "Error:" in code_generation_prompt_template:
        logger.error("Code generation prompt template not loaded correctly from config.")
        return {"generated_code": "# Config error: Code generation prompt missing.",
                "generated_code_language": "text",
                "error": (error_message or "") + "; Config error: Code generation prompt missing."}

    messages = [SystemMessage(content=code_generation_prompt_template)]
    for user_msg, ai_msg in history:
        messages.append(HumanMessage(content=user_msg))
        if "Generated code snippet" in ai_msg:
             try: saved_file = ai_msg.split("results/")[-1].split(")")[0]; messages.append(AIMessage(content=f"(Generated code saved to {saved_file})"))
             except: messages.append(AIMessage(content=ai_msg))
        else: messages.append(AIMessage(content=ai_msg))
    messages.append(HumanMessage(content=query))
    logger.info(f"Using history (last {len(history)} turns) for code context")

    logger.info(f"Using LLM for code generation: {type(coding_llm).__name__}")

    try:
        response = coding_llm.invoke(messages); raw_code_response = response.content.strip()
        cleaned_code = raw_code_response
        # Updated Regex to be more robust
        match_py = re.search(r"```(?:python)?\n?(.*?)```", raw_code_response, re.DOTALL | re.IGNORECASE)
        match_r = re.search(r"```r\n?(.*?)```", raw_code_response, re.DOTALL | re.IGNORECASE)

        if match_r: # Check R first as it's more specific
            detected_language = "r"; cleaned_code = match_r.group(1).strip(); logger.info("Detected language: R (from ```R)")
        elif match_py: # Then check Python
            detected_language = "python"; cleaned_code = match_py.group(1).strip(); logger.info("Detected language: Python (from ```python or ```)")
        else: # No backticks found, use heuristics
             cleaned_code = raw_code_response
             if any(kw in cleaned_code for kw in ['library(', '<-', 'ggplot', 'dplyr']): detected_language = "r"; logger.info("Detected language: R (heuristic, no backticks)")
             elif any(kw in cleaned_code for kw in ['def ', 'import ', 'class ', 'print(']): detected_language = "python"; logger.info(f"Detected language: {detected_language} (heuristic, no backticks)")
             else: detected_language = "text"; logger.info(f"Detected language: {detected_language} (default, no backticks/keywords)")
        generated_code_text = cleaned_code

        logger.info("LLM code generation complete.")
        logger.debug("Generated code (start):\n%s", "\n".join(generated_code_text.splitlines()[:5]))
    except Exception as e:
        code_error = f"Code generation failed: {str(e)}"; logger.error(code_error, exc_info=True)
        error_message = (error_message + "; " + code_error) if error_message else code_error
        generated_code_text = f"# Error during code generation: {e}"; detected_language = "text"

    # Return only updated fields
    return {"generated_code": generated_code_text, "generated_code_language": detected_language, "error": error_message}

