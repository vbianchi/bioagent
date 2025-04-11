import logging
import re
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Try importing the central AgentState definition
try:
    from src.core.state import AgentState
except ImportError:
    # Fallback definition if import fails
    from typing import TypedDict, List, Dict, Any, Optional, Tuple # Keep import here
    logger.warning("Could not import AgentState from src.core.state, using fallback definition in coding.py.")
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

def call_coding_agent(state: AgentState, coding_llm, code_generation_prompt_template: str) -> Dict[str, Any]:
    """Coding Agent: Generates code, using history message list, detects language."""
    logger.info("--- Calling Coding Agent ---")
    query = state['query']; history = state['history']
    error_message = state.get("error"); generated_code_text = None; detected_language = "text"
    logger.info(f"Received code request: {query}")

    # Format message list for LLM, including System Prompt
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
        match_py = re.search(r"```python\n?(.*?)```", raw_code_response, re.DOTALL | re.IGNORECASE)
        match_r = re.search(r"```r\n?(.*?)```", raw_code_response, re.DOTALL | re.IGNORECASE)
        match_generic = re.search(r"```\n?(.*?)```", raw_code_response, re.DOTALL)
        if match_py: detected_language = "python"; cleaned_code = match_py.group(1).strip(); logger.info("Detected language: Python (from ```python)")
        elif match_r: detected_language = "r"; cleaned_code = match_r.group(1).strip(); logger.info("Detected language: R (from ```R)")
        elif match_generic:
             cleaned_code = match_generic.group(1).strip()
             if any(kw in cleaned_code for kw in ['library(', '<-', 'ggplot', 'dplyr']): detected_language = "r"; logger.info("Detected language: R (heuristic within ```)")
             else: detected_language = "python"; logger.info(f"Detected language: {detected_language} (heuristic/default within ```)")
        else:
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

