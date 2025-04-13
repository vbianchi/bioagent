import logging
import re # For parsing LLM responses
import sys # <<< Added import sys >>>
from typing import Dict, Any, List, Optional

# Import central AgentState definition
from src.core.state import AgentState
# Import config loading function to get settings within the node
from src.core.config_loader import load_config, get_config_value
# Import search tool functions
from src.tools.literature_search import search_pubmed, search_arxiv
from src.tools.web_search import search_google
# Import LLM type hint
from langchain_core.language_models.chat_models import BaseChatModel
# Import colors for CLI output
from colorama import Fore, Style

# Define colors used in this module (or import from main)
COLOR_QUESTION = Fore.BLUE + Style.BRIGHT
COLOR_OUTPUT = Fore.GREEN
COLOR_RESET = Style.RESET_ALL
COLOR_WARN = Fore.YELLOW
COLOR_INPUT = Fore.YELLOW # For user input prompt

logger = logging.getLogger(__name__)

# --- Node Function: Start Deep Research ---
# (No changes needed in this function from v1.7)
def start_deep_research(state: AgentState, llm: BaseChatModel, hypothesis_prompt_template: str) -> AgentState:
    """
    Initializes the deep research process.
    - Takes the user query.
    - Uses an LLM to formulate an initial hypothesis or research question.
    - Generates an initial **enumerated list** search plan.
    - Initializes related state fields.
    """
    logger.info("--- Starting Deep Research Node ---")
    original_query = state['query']
    error_message = state.get("error") # Preserve existing errors

    # Initialize fields
    hypothesis: Optional[str] = None
    search_plan_list: List[str] = [] # Plan is now a list
    evidence_log: List[Dict[str, Any]] = []
    research_iterations: int = 0
    evaluation_summary: Optional[str] = None
    more_research_needed: bool = True
    plan_approved: Optional[str] = None
    plan_modifications: Optional[str] = None # Initialize new field

    if "Error:" in hypothesis_prompt_template:
        logger.error("Hypothesis prompt template not loaded correctly.")
        err_msg = "Config error: Hypothesis prompt missing."
        # Ensure all relevant fields are initialized even on error
        return {**state, "hypothesis": "Error: Config.", "search_plan": [], "evidence_log": [], "research_iterations": 0, "more_research_needed": False, "plan_approved": "no", "plan_modifications": None, "error": (error_message or "") + "; " + err_msg}

    try:
        logger.info(f"Formulating initial hypothesis and search plan list for query: {original_query}")
        prompt = hypothesis_prompt_template.format(query=original_query)
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        logger.debug(f"LLM raw response for hypothesis/plan:\n{response_text}")

        # --- Parse LLM Response for Hypothesis and Enumerated List Plan ---
        hypothesis = "Could not parse hypothesis." # Default
        parsed_search_plan_items = []

        # Extract Hypothesis
        hyp_match = re.search(r"Hypothesis:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
        if hyp_match:
            potential_hyp = hyp_match.group(1).split("Initial Search Plan:")[0].strip()
            if potential_hyp: hypothesis = potential_hyp

        # Extract Enumerated List Items for Plan
        plan_match = re.search(r"Initial Search Plan:\s*\n(.*?)(?:\n\n|\Z)", response_text, re.IGNORECASE | re.DOTALL)
        if plan_match:
            plan_block = plan_match.group(1).strip()
            list_items = re.findall(r"^\s*(?:\d+\.|[-*+])\s+(.*)", plan_block, re.MULTILINE)
            if list_items: parsed_search_plan_items = [item.strip() for item in list_items]

        logger.info(f"Parsed Hypothesis: {hypothesis}")
        if parsed_search_plan_items:
            search_plan_list = parsed_search_plan_items
            logger.info(f"Parsed Initial Search Plan List: {search_plan_list}")
        else:
             logger.warning("Could not parse enumerated 'Initial Search Plan:' list reliably.")
             search_plan_list = [state.get("refined_query", original_query)]
             logger.info(f"Using fallback search plan: {search_plan_list}")

    except Exception as e:
        formulation_error = f"LLM call failed during hypothesis/plan formulation: {e}"
        logger.error(formulation_error, exc_info=True)
        error_message = (error_message + "; " + formulation_error) if error_message else formulation_error
        hypothesis = "Error during formulation."
        search_plan_list = []
        more_research_needed = False
        plan_approved = "no"

    # Return the entire state merged with updates
    return {
        **state,
        "hypothesis": hypothesis,
        "search_plan": search_plan_list,
        "evidence_log": evidence_log,
        "research_iterations": research_iterations,
        "evaluation_summary": evaluation_summary,
        "more_research_needed": more_research_needed,
        "plan_approved": plan_approved,
        "plan_modifications": plan_modifications, # Starts as None
        "error": error_message
    }

# --- Node Function: Ask for Plan Approval ---
# (No changes needed in this function from v1.7)
def ask_plan_approval(state: AgentState) -> AgentState:
    """
    Presents the generated/refined hypothesis and search plan list to the user.
    Asks for approval, modification request, or rejection (CLI only).
    Updates the 'plan_approved' field ('yes', 'mod', 'no').
    """
    logger.info("--- Asking User for Plan Approval ---")
    hypothesis = state.get("hypothesis", "N/A")
    search_plan_list = state.get("search_plan", [])
    run_dir = state.get("run_dir")
    error_message = state.get("error")
    approval = "no" # Default

    plan_display = "\n".join([f"{i+1}. {item}" for i, item in enumerate(search_plan_list)]) if search_plan_list else "No plan generated."

    if run_dir is not None and hypothesis and search_plan_list and "Error" not in hypothesis:
        print(f"\n{COLOR_OUTPUT}----------------------------------------{COLOR_RESET}")
        print(f"{COLOR_OUTPUT}Proposed Research Plan:{COLOR_RESET}")
        print(f"{Style.BRIGHT}Hypothesis/Question:{Style.NORMAL}\n{hypothesis}")
        print(f"\n{Style.BRIGHT}Search Plan Items:{Style.NORMAL}\n{plan_display}")
        print(f"{COLOR_OUTPUT}----------------------------------------{COLOR_RESET}")
        try:
            user_input = input(f"\n{COLOR_QUESTION}Approve plan (yes), suggest modifications (mod), or reject (no)?: {COLOR_RESET}").strip().lower()
            if user_input == "yes" or user_input == "y": approval = "yes"; logger.info("User approved the research plan.")
            elif user_input == "mod" or user_input == "m": approval = "mod"; logger.info("User wants to modify the research plan.")
            else: approval = "no"; logger.info("User rejected the research plan. Stopping deep research.")
        except (EOFError, KeyboardInterrupt): logger.warning("Input interrupted during plan approval. Defaulting to 'no'."); approval = "no"
        except Exception as e: logger.error(f"Error reading plan approval input: {e}", exc_info=True); error_message = (error_message + f"; Error reading plan approval: {e}") if error_message else f"Error reading plan approval: {e}"; approval = "no"
    elif run_dir is None:
        logger.info("Running in non-CLI mode or run_dir not set. Auto-approving plan.")
        if hypothesis and search_plan_list and "Error" not in hypothesis: approval = "yes"
        else: approval = "no"; logger.warning("Plan generation failed, cannot auto-approve.")
    else:
        logger.warning("Hypothesis or search plan contains errors or is empty. Cannot ask for approval. Denying plan.")
        approval = "no"

    # Clear modification suggestions from previous rounds if any
    return {**state, "plan_approved": approval, "plan_modifications": None, "error": error_message}


# --- Node Function: Get Plan Modifications (Fix v1.8) ---

def get_plan_modifications(state: AgentState) -> AgentState:
    """
    Prompts the user (CLI only) to enter their suggested modifications to the plan.
    Stores the input in 'plan_modifications'. Requires 'sys' module to be imported.
    """
    logger.info("--- Getting Plan Modifications from User ---")
    run_dir = state.get("run_dir")
    error_message = state.get("error")
    modifications: Optional[str] = None

    if run_dir is not None: # Only prompt in CLI mode
        try:
            print(f"\n{COLOR_OUTPUT}Please describe the changes you'd like to make to the hypothesis or search plan items below.{COLOR_RESET}")
            print(f"{COLOR_INPUT}Enter your suggestions (press Ctrl+D or Ctrl+Z then Enter when done):{COLOR_RESET}")
            # <<< Use sys.stdin to read potentially multiple lines >>>
            lines = sys.stdin.readlines()
            modifications = "".join(lines).strip()
            if modifications:
                logger.info(f"Received user modifications:\n{modifications}")
            else:
                logger.warning("User provided empty modifications.")
                modifications = None # Treat empty input as no modification suggested

        except (EOFError, KeyboardInterrupt):
            logger.warning("Input interrupted while getting modifications. Proceeding without changes.")
            modifications = None
        except NameError as ne: # Catch if sys is still not imported somehow
            logger.error(f"NameError during input reading (is 'sys' imported?): {ne}", exc_info=True)
            error_message = (error_message + f"; NameError reading modifications: {ne}") if error_message else f"NameError reading modifications: {ne}"
            modifications = None
        except Exception as e:
            logger.error(f"Error reading plan modifications: {e}", exc_info=True)
            error_message = (error_message + f"; Error reading plan modifications: {e}") if error_message else f"Error reading plan modifications: {e}"
            modifications = None # Clear modifications on error
    else:
        logger.warning("Cannot get plan modifications in non-CLI mode. Skipping modification step.")
        modifications = None # Cannot get modifications in non-CLI mode

    return {**state, "plan_modifications": modifications, "error": error_message}


# --- Node Function: Refine Plan with Feedback ---
# (No changes needed in this function from v1.7)
def refine_plan_with_feedback(state: AgentState, llm: BaseChatModel, refine_plan_prompt_template: str) -> AgentState:
    """
    Uses an LLM to re-draft the hypothesis and search plan based on user feedback.
    Updates 'hypothesis' and 'search_plan' state fields.
    """
    logger.info("--- Refining Plan with User Feedback ---")
    original_hypothesis = state.get("hypothesis", "")
    original_plan_list = state.get("search_plan", [])
    user_modifications = state.get("plan_modifications")
    error_message = state.get("error")

    if not user_modifications:
        logger.warning("No user modifications provided. Skipping plan refinement.")
        return {**state, "plan_approved": None}

    original_plan_display = "\n".join([f"{i+1}. {item}" for i, item in enumerate(original_plan_list)])

    if "Error:" in refine_plan_prompt_template:
        logger.error("Refine plan prompt template not loaded correctly.")
        err_msg = "Config error: Refine plan prompt missing."
        return {**state, "error": (error_message or "") + "; " + err_msg, "plan_approved": None}

    new_hypothesis = original_hypothesis
    new_search_plan_list = original_plan_list

    try:
        logger.info("Calling LLM to refine plan based on user feedback...")
        prompt = refine_plan_prompt_template.format(
            original_hypothesis=original_hypothesis,
            original_plan=original_plan_display,
            user_feedback=user_modifications
        )
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        logger.debug(f"LLM raw response for refined plan:\n{response_text}")

        # --- Parse LLM Response for Refined Hypothesis and Plan ---
        parsed_new_plan_items = []
        hyp_match = re.search(r"Refined Hypothesis:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
        if hyp_match:
            potential_hyp = hyp_match.group(1).split("Refined Search Plan:")[0].strip()
            if potential_hyp: new_hypothesis = potential_hyp

        plan_match = re.search(r"Refined Search Plan:\s*\n?(.*)", response_text, re.IGNORECASE | re.DOTALL)
        if plan_match:
            plan_block = plan_match.group(1).strip()
            list_items = re.findall(r"^\s*(?:\d+\.|[-*+])\s+(.*)", plan_block, re.MULTILINE)
            if list_items: parsed_new_plan_items = [item.strip() for item in list_items]

        logger.info(f"Parsed Refined Hypothesis: {new_hypothesis}")
        if parsed_new_plan_items:
            new_search_plan_list = parsed_new_plan_items
            logger.info(f"Parsed Refined Search Plan List: {new_search_plan_list}")
        else:
             logger.warning("Could not parse enumerated 'Refined Search Plan:' list. Keeping original plan.")

    except Exception as e:
        refine_error = f"LLM call failed during plan refinement: {e}"
        logger.error(refine_error, exc_info=True)
        error_message = (error_message + "; " + refine_error) if error_message else refine_error
        new_hypothesis = original_hypothesis
        new_search_plan_list = original_plan_list

    return {
        **state,
        "hypothesis": new_hypothesis,
        "search_plan": new_search_plan_list,
        "plan_approved": None, # Reset approval status
        "plan_modifications": None, # Clear modifications
        "error": error_message
    }


# --- Node Function: Execute Search Plan ---
# (No changes needed in this function from v1.6)
def execute_search_plan(state: AgentState, google_search_tool_object: Optional[Any]) -> AgentState:
    """
    Executes literature (PubMed/ArXiv) and web (Google) searches for each item
    in the current search_plan list. Formats results and appends them to the evidence_log.
    Loads config internally to avoid potential issues with partial.
    """
    logger.info("--- Executing Search Plan Node ---")
    search_plan_list = state.get("search_plan", [])
    current_evidence_log = state.get("evidence_log", [])
    error_message = state.get("error")
    iteration = state.get("research_iterations", 0)

    # Load config inside the node
    config = load_config()
    if not config:
        logger.error("Failed to load config inside execute_search_plan.")
        error_message = (error_message + "; Failed to load config in search node") if error_message else "Failed to load config in search node"
        max_lit, max_web = 3, 5 # Defaults
    else:
        max_lit = get_config_value(config, "search_settings.max_results_per_source", 3)
        max_web = get_config_value(config, "search_settings.num_google_results", 5)

    if not search_plan_list:
        logger.warning(f"Search plan list is empty for iteration {iteration+1}. Skipping search execution.")
        return {**state}

    logger.info(f"Executing search iteration {iteration+1} with {len(search_plan_list)} plan items.")
    logger.debug(f"Using search limits per item: max_lit={max_lit}, max_web={max_web}")

    round_evidence: List[Dict[str, Any]] = []

    for item_index, plan_item_query in enumerate(search_plan_list):
        logger.info(f"Searching for plan item {item_index+1}/{len(search_plan_list)}: '{plan_item_query}'")
        search_query = plan_item_query

        # Literature Search
        try:
            logger.debug(f"Performing literature search for item '{search_query}'...")
            pubmed_results = search_pubmed(search_query, max_lit)
            arxiv_results = search_arxiv(search_query, max_lit)
            raw_literature_results = pubmed_results + arxiv_results
            logger.debug(f"Found {len(raw_literature_results)} raw literature results for item.")
            for res in raw_literature_results:
                source_prefix = res.get("source", "Lit"); source_id = f"{source_prefix}:{res.get('id', 'N/A')}"
                content = res.get("abstract", "No abstract found")
                if content and content != "No abstract found":
                    evidence_entry = {"plan_item": plan_item_query, "source_type": "literature", "source_id": source_id, "title": res.get("title", "N/A"), "content": content, "url": res.get("url", "#")}
                    round_evidence.append(evidence_entry)
                else: logger.debug(f"Skipping lit result {source_id} (no abstract).")
        except Exception as e:
            lit_error = f"Literature search failed for item '{search_query}': {e}"; logger.error(lit_error, exc_info=False)
            error_message = (error_message + "; " + lit_error) if error_message else lit_error

        # Web Search
        try:
            logger.debug(f"Performing web search for item '{search_query}'...")
            raw_google_results = search_google(query=search_query, search_tool=google_search_tool_object, num_results=max_web)
            logger.debug(f"Found {len(raw_google_results)} raw web results for item.")
            for i, res in enumerate(raw_google_results):
                if isinstance(res, dict) and "error" in res:
                    web_error = f"Web search tool error for item '{search_query}': {res['error']}"; logger.error(web_error)
                    error_message = (error_message + "; " + web_error) if error_message else web_error; continue
                source_id = f"Google:{i+1}"; content = res.get("snippet", "No snippet available.")
                if content and content != "No snippet available.":
                    evidence_entry = {"plan_item": plan_item_query, "source_type": "web", "source_id": source_id, "title": res.get("title", "N/A"), "content": content, "url": res.get("link", "#")}
                    round_evidence.append(evidence_entry)
                else: logger.debug(f"Skipping web result {source_id} (no snippet).")
        except Exception as e:
            web_error = f"Web search failed for item '{search_query}': {e}"; logger.error(web_error, exc_info=False)
            error_message = (error_message + "; " + web_error) if error_message else web_error

    logger.info(f"Adding {len(round_evidence)} new evidence items from this round to the log.")
    updated_evidence_log = current_evidence_log + round_evidence
    logger.info(f"Total evidence items in log: {len(updated_evidence_log)}")

    return {**state, "evidence_log": updated_evidence_log, "search_results": None, "google_results": None, "error": error_message}


# --- Node Function: Evaluate Findings (Implemented v1.6) ---
# (No changes needed in this function from v1.6)
def evaluate_findings(state: AgentState, llm: BaseChatModel, evaluation_prompt_template: str) -> AgentState:
     """
     Evaluates the evidence gathered so far against the hypothesis using an LLM.
     Decides whether to continue research and generates a refined search plan list if needed.
     Updates 'evaluation_summary', 'search_plan', 'more_research_needed', and increments 'research_iterations'.
     """
     logger.info("--- Evaluating Findings Node ---")
     hypothesis = state.get("hypothesis", "No hypothesis provided.")
     evidence_log = state.get("evidence_log", [])
     current_iterations = state.get("research_iterations", 0)
     error_message = state.get("error")

     logger.info(f"Evaluating {len(evidence_log)} evidence items after iteration {current_iterations}.")

     # Initialize defaults
     evaluation_summary = "Evaluation failed or no new evidence."
     next_search_plan_list = []
     more_research_needed = False # Default to concluding

     if not evidence_log and current_iterations == 0 : # No evidence after first search
         logger.warning("No evidence collected in the first round to evaluate. Concluding research.")
         return {**state, "evaluation_summary": "No evidence collected.", "more_research_needed": False, "research_iterations": current_iterations + 1, "search_plan": []}

     # Format evidence for the prompt (limit length)
     evidence_texts = []
     MAX_EVIDENCE_CHARS = 4000 # Adjust as needed
     current_chars = 0; items_included = 0
     relevant_evidence = evidence_log[-50:] # Limit context window
     for entry in reversed(relevant_evidence):
         entry_text = f"Source: {entry.get('source_id', 'N/A')} (Related to: {entry.get('plan_item', 'N/A')})\nContent Snippet: {entry.get('content', '')[:300]}...\n---\n"
         if current_chars + len(entry_text) > MAX_EVIDENCE_CHARS and items_included > 0:
             logger.warning(f"Truncating evidence log for evaluation prompt at {items_included} items due to length.")
             break
         evidence_texts.append(entry_text); current_chars += len(entry_text); items_included += 1
     evidence_prompt_text = "\n".join(reversed(evidence_texts)) if evidence_texts else "No evidence to display."

     if "Error:" in evaluation_prompt_template:
         logger.error("Evaluation prompt template not loaded correctly.")
         err_msg = "Config error: Evaluation prompt missing."
         return {**state, "evaluation_summary": "Error: Config.", "more_research_needed": False, "research_iterations": current_iterations + 1, "search_plan": [], "error": (error_message or "") + "; " + err_msg}

     try:
         logger.info("Calling LLM for evidence evaluation and next plan...")
         prompt = evaluation_prompt_template.format(hypothesis=hypothesis, evidence_text=evidence_prompt_text)
         response = llm.invoke(prompt)
         response_text = response.content.strip()
         logger.debug(f"LLM raw response for evaluation:\n{response_text}")

         # --- Parse LLM Response ---
         summary_match = re.search(r"Evaluation Summary:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
         decision_match = re.search(r"Decision:\s*(Continue Research|Conclude Research)", response_text, re.IGNORECASE | re.DOTALL)
         plan_match = re.search(r"Refined Search Plan:\s*\n?(.*)", response_text, re.IGNORECASE | re.DOTALL)

         if summary_match:
             evaluation_summary = summary_match.group(1).split("Gaps/Next Steps:")[0].split("Decision:")[0].strip()
             logger.info(f"Parsed Evaluation Summary: {evaluation_summary}")
         else: logger.warning("Could not parse 'Evaluation Summary:' from LLM response."); evaluation_summary = "LLM response format error."

         if decision_match:
             decision = decision_match.group(1).strip()
             if "Continue Research".lower() in decision.lower(): more_research_needed = True; logger.info("Parsed Decision: Continue Research")
             else: more_research_needed = False; logger.info("Parsed Decision: Conclude Research")
         else: logger.warning("Could not parse 'Decision:' from LLM response. Defaulting to Conclude."); more_research_needed = False

         if more_research_needed and plan_match:
             plan_block = plan_match.group(1).strip()
             if plan_block.lower() != "n/a":
                 list_items = re.findall(r"^\s*(?:\d+\.|[-*+])\s+(.*)", plan_block, re.MULTILINE)
                 if list_items: next_search_plan_list = [item.strip() for item in list_items]; logger.info(f"Parsed Refined Search Plan List: {next_search_plan_list}")
                 else: logger.warning("Could not parse enumerated list from 'Refined Search Plan:'. Stopping research."); more_research_needed = False; next_search_plan_list = []
             else: logger.info("Refined Search Plan is 'N/A'. Stopping research."); more_research_needed = False; next_search_plan_list = []
         elif more_research_needed: logger.warning("Decision 'Continue' but no 'Refined Search Plan:' found/parsed. Stopping."); more_research_needed = False; next_search_plan_list = []
         else: next_search_plan_list = [] # Concluding

     except Exception as e:
         eval_error = f"LLM call failed during evaluation: {e}"; logger.error(eval_error, exc_info=True)
         error_message = (error_message + "; " + eval_error) if error_message else eval_error
         evaluation_summary = "Error during evaluation."; more_research_needed = False; next_search_plan_list = []

     # Update state
     return {
         **state,
         "evaluation_summary": evaluation_summary,
         "search_plan": next_search_plan_list, # Update with the NEW plan list
         "more_research_needed": more_research_needed,
         "research_iterations": current_iterations + 1, # Increment counter
         "error": error_message
     }


# --- Node Function: Generate Final Report (Placeholder) ---
# (No changes needed in this function from v1.7)
def generate_final_report(state: AgentState) -> AgentState:
     """Placeholder node for generating the final synthesized report."""
     logger.info("--- (Placeholder) Generating Final Report ---")
     logger.info(f"Generating report based on {len(state.get('evidence_log', []))} evidence items.")
     # TODO: Implement LLM call using final_report_prompt_template, state['evidence_log'], state['hypothesis'], state['query']
     # This function *will* need llm and final_report_prompt_template passed via partial once implemented.
     # TODO: Update state['synthesized_report'] with the generated report
     return {**state, "synthesized_report": "This is a placeholder final report based on the refined approach."}
