import os
import sys
import json
import datetime
import logging
from io import StringIO # Not used directly here, but maybe by imported modules
import re # Used by helper function potentially
import traceback
from typing import List, Dict, Any, Optional, Tuple
from functools import partial # Used for graph node setup

# Import colorama for colored CLI output
from colorama import init as colorama_init, Fore, Style

# Import dotenv for loading environment variables from .env file
from dotenv import load_dotenv

# LLM and LangGraph Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END, START # Use START constant

# --- Import Core Components ---
from src.core.config_loader import load_config, get_config_value
try:
    # Use the updated AgentState definition (v1.3)
    from src.core.state import AgentState
except ImportError:
    print(f"{Fore.RED}FATAL ERROR: src.core.state.py not found or cannot be imported.{Style.RESET_ALL}")
    sys.exit(1)

# --- Import Tool Utilities ---
from src.tools.llm_utils import initialize_llm
# --- Assume google_search tool is available in the environment ---
google_search = None
try: from core_tools import google_search
except ImportError:
    if 'google_search' in globals(): google_search = globals()['google_search']
    else: pass

# -----------------------------------------------------------------
# --- Import Agent Nodes and Conditional Logic ---
# Standard Path Agents
from src.agents.router import route_query, decide_next_node
from src.agents.refine import refine_query_node
from src.agents.literature import call_literature_agent
from src.agents.download import ask_download_preference, download_arxiv_pdfs
from src.agents.download import should_download as should_download_lit_path
from src.agents.summarize import summarize_results
from src.agents.chat import call_chat_agent
from src.agents.coding import call_coding_agent
# New Deep Research Agents
from src.agents.deep_research import (
    start_deep_research,
    ask_plan_approval,
    get_plan_modifications,  # <<< New node >>>
    refine_plan_with_feedback, # <<< New node >>>
    execute_search_plan,
    evaluate_findings,
    generate_final_report
)

# --- Initialize Colorama ---
colorama_init(autoreset=True)

# --- Define Color Scheme Constants ---
COLOR_INFO = Fore.CYAN; COLOR_INPUT = Fore.YELLOW; COLOR_OUTPUT = Fore.GREEN
COLOR_SUMMARY = Fore.MAGENTA; COLOR_ERROR = Fore.RED; COLOR_WARN = Fore.YELLOW
COLOR_DEBUG = Fore.BLUE; COLOR_RESET = Style.RESET_ALL; COLOR_FILE = Fore.LIGHTBLUE_EX
COLOR_QUESTION = Fore.BLUE + Style.BRIGHT; COLOR_CODE = Fore.LIGHTYELLOW_EX; COLOR_SYNTHESIS = Fore.LIGHTGREEN_EX

# --- Custom Colored Logging Formatter ---
class ColoredFormatter(logging.Formatter):
    LOG_COLORS = { logging.DEBUG: COLOR_DEBUG, logging.INFO: COLOR_INFO, logging.WARNING: COLOR_WARN, logging.ERROR: COLOR_ERROR, logging.CRITICAL: Fore.RED + Style.BRIGHT, }
    def format(self, record): log_color = self.LOG_COLORS.get(record.levelno, COLOR_RESET); log_fmt = f"{log_color}{record.levelname}: {record.getMessage()}{COLOR_RESET}"; return log_fmt

# --- Basic Logging Setup (Early) ---
logger = logging.getLogger(__name__); logger.setLevel(logging.INFO); logger.propagate = False
console_handler = logging.StreamHandler(sys.stdout); console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter()); logger.addHandler(console_handler)

# --- Configuration Loading ---
logger.info("Loading configuration...")
config = load_config()
if not config: logger.critical("Config empty/not loaded. Exiting."); sys.exit(1)
logger.info("Configuration loaded successfully.")

# --- Environment Setup (.env and Entrez Email) ---
logger.info("Loading environment variables...")
load_dotenv()
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
if not ENTREZ_EMAIL: logger.critical("ENTREZ_EMAIL not set. Exiting."); sys.exit(1)
try: from Bio import Entrez; Entrez.email = ENTREZ_EMAIL; logger.info(f"Entrez Email set.")
except ImportError: logger.critical("BioPython not found. Exiting."); sys.exit(1)

# --- LLM Instantiation ---
logger.info("Initializing Language Models...")
llm = initialize_llm(config, "llm_provider", "llm_settings")
if llm is None: logger.critical("Failed to initialize main LLM. Exiting."); sys.exit(1)
coding_llm = initialize_llm(config, "coding_agent_settings.llm_provider", "coding_agent_settings")
if coding_llm is None: logger.warning("Failed coding LLM init. Falling back to main LLM."); coding_llm = llm
deep_research_llm = llm # Using main LLM for DR for now
logger.info("Language Models initialized.")

# --- Search Settings & Deep Research Settings ---
MAX_RESULTS_PER_SOURCE = get_config_value(config, "search_settings.max_results_per_source", 3)
MAX_ABSTRACTS_TO_SUMMARIZE = get_config_value(config, "search_settings.max_abstracts_to_summarize", 3)
NUM_GOOGLE_RESULTS = get_config_value(config, "search_settings.num_google_results", 5)
MAX_RESEARCH_ITERATIONS = get_config_value(config, "search_settings.max_research_iterations", 3)
logger.info(f"Search settings: MaxLit={MAX_RESULTS_PER_SOURCE}, MaxGoog={NUM_GOOGLE_RESULTS}, MaxAbs={MAX_ABSTRACTS_TO_SUMMARIZE}")
logger.info(f"Deep Research settings: MaxIter={MAX_RESEARCH_ITERATIONS}")

# --- Prompt Templates ---
ROUTING_PROMPT_TEMPLATE = get_config_value(config, "prompts.routing_prompt", "Error: Routing prompt missing.")
REFINEMENT_PROMPT_TEMPLATE = get_config_value(config, "prompts.refinement_prompt", "Error: Refinement prompt missing.")
SUMMARIZATION_PROMPT_TEMPLATE = get_config_value(config, "prompts.summarization_prompt", "Error: Summarization prompt missing.")
CODE_GENERATION_PROMPT_TEMPLATE = get_config_value(config, "prompts.code_generation_prompt", "Error: Code generation prompt missing.")
SYNTHESIS_PROMPT_TEMPLATE = get_config_value(config, "prompts.synthesis_prompt", "Error: Synthesis prompt missing.")
HYPOTHESIS_PROMPT_TEMPLATE = get_config_value(config, "prompts.hypothesis_prompt", "Error: Hypothesis prompt missing.")
EVALUATION_PROMPT_TEMPLATE = get_config_value(config, "prompts.evaluation_prompt", "Error: Evaluation prompt missing.")
# <<< Load new prompt >>>
REFINE_PLAN_PROMPT_TEMPLATE = get_config_value(config, "prompts.refine_plan_prompt", "Error: Refine plan prompt missing.")
FINAL_REPORT_PROMPT_TEMPLATE = get_config_value(config, "prompts.final_report_prompt", "Error: Final report prompt missing.")
prompt_templates = {k: v for k, v in locals().items() if k.endswith('_PROMPT_TEMPLATE')}
for name, template in prompt_templates.items():
    if isinstance(template, str) and template.startswith("Error:"): logger.error(f"Prompt '{name}' failed load!");

# --- Google Search Tool Setup ---
google_search_tool_object = None; logger.info("Checking for Google Search tool...")
try: from core_tools import google_search;
except ImportError:
    if 'google_search' in globals() and globals()['google_search'] is not None: google_search_tool_object = globals()['google_search']
if google_search_tool_object: logger.info(f"Google Search tool available.")
else: logger.warning("Google Search tool is None.")

# --- Conditional Edge Logic ---
def decide_after_refine(state: AgentState) -> str:
    intent = state.get("route_intent"); logger.debug(f"Deciding path after refine: {intent}")
    if intent == "literature_search": return "literature_agent"
    elif intent == "deep_research": return "start_deep_research"
    else: logger.warning(f"Unexpected intent '{intent}' after refine."); return END

def should_download_lit_path(state: AgentState) -> str:
    preference = state.get("download_preference", "no"); intent = state.get("route_intent"); run_dir = state.get("run_dir")
    if intent != "literature_search": logger.error(f"should_download_lit_path invalid intent: {intent}."); return END
    if preference == "yes" and run_dir: return "download_arxiv_pdfs"
    else: return "summarizer"

def decide_after_summary(state: AgentState) -> str:
    intent = state.get("route_intent")
    if intent == "literature_search": return END
    else: logger.warning(f"Unexpected intent '{intent}' after summary."); return END

# <<< Updated: Conditional edge after plan approval prompt >>>
def decide_after_approval_prompt(state: AgentState) -> str:
    """Routes based on user approval/modification choice."""
    approval_status = state.get("plan_approved", "no") # Default to 'no' if None
    if approval_status == "yes":
        logger.info("Plan approved. Proceeding to execute search plan.")
        return "execute_search_plan"
    elif approval_status == "mod":
        logger.info("Routing to get plan modifications from user.")
        return "get_plan_modifications"
    else: # 'no' or any other case
        logger.warning("Plan not approved or modified. Ending deep research workflow.")
        return END

# Conditional edge for the deep research loop
def decide_next_research_step(state: AgentState) -> str:
    more_research = state.get("more_research_needed", False); iterations = state.get("research_iterations", 0)
    if more_research and iterations < MAX_RESEARCH_ITERATIONS: logger.info(f"Decision: Continue research (Iter {iterations+1})."); return "execute_search_plan"
    elif more_research: logger.warning(f"Decision: Max iterations ({MAX_RESEARCH_ITERATIONS}) reached."); return "generate_final_report"
    else: logger.info("Decision: Conclude research."); return "generate_final_report"
# --- End Conditional Edge Logic ---


# --- Graph Definition ---
logger.info("Defining agent graph...")
graph_builder = StateGraph(AgentState)

# Add nodes (Standard Paths)
graph_builder.add_node("router", partial(route_query, llm=llm, routing_prompt_template=ROUTING_PROMPT_TEMPLATE))
graph_builder.add_node("refine_query", partial(refine_query_node, llm=llm, refinement_prompt_template=REFINEMENT_PROMPT_TEMPLATE))
graph_builder.add_node("literature_agent", partial(call_literature_agent, max_pubmed=MAX_RESULTS_PER_SOURCE, max_arxiv=MAX_RESULTS_PER_SOURCE))
graph_builder.add_node("ask_download_preference", ask_download_preference)
graph_builder.add_node("download_arxiv_pdfs", download_arxiv_pdfs)
graph_builder.add_node("summarizer", partial(summarize_results, llm=llm, summarization_prompt_template=SUMMARIZATION_PROMPT_TEMPLATE, max_abstracts=MAX_ABSTRACTS_TO_SUMMARIZE))
graph_builder.add_node("chat_agent", partial(call_chat_agent, llm=llm))
graph_builder.add_node("coding_agent", partial(call_coding_agent, coding_llm=coding_llm, code_generation_prompt_template=CODE_GENERATION_PROMPT_TEMPLATE))

# Add nodes (New Deep Research Path)
graph_builder.add_node("start_deep_research", partial(start_deep_research, llm=deep_research_llm, hypothesis_prompt_template=HYPOTHESIS_PROMPT_TEMPLATE))
graph_builder.add_node("ask_plan_approval", ask_plan_approval)
# <<< Add new nodes for modification >>>
graph_builder.add_node("get_plan_modifications", get_plan_modifications)
graph_builder.add_node("refine_plan_with_feedback", partial(refine_plan_with_feedback, llm=deep_research_llm, refine_plan_prompt_template=REFINE_PLAN_PROMPT_TEMPLATE))
# ---
graph_builder.add_node("execute_search_plan", partial(execute_search_plan, google_search_tool_object=google_search_tool_object))
# <<< Update: Pass args to evaluate_findings now that it's implemented >>>
graph_builder.add_node("evaluate_findings", partial(evaluate_findings, llm=deep_research_llm, evaluation_prompt_template=EVALUATION_PROMPT_TEMPLATE))
# Pass only function for generate_final_report placeholder
graph_builder.add_node("generate_final_report", generate_final_report)


# --- Define Edges ---
graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges("router", decide_next_node, {"refine_query": "refine_query", "chat_agent": "chat_agent", "coding_agent": "coding_agent", END: END})
graph_builder.add_conditional_edges("refine_query", decide_after_refine, {"literature_agent": "literature_agent", "start_deep_research": "start_deep_research", END: END})
# Lit Search Path
graph_builder.add_edge("literature_agent", "ask_download_preference")
graph_builder.add_conditional_edges("ask_download_preference", should_download_lit_path, {"download_arxiv_pdfs": "download_arxiv_pdfs", "summarizer": "summarizer"})
graph_builder.add_edge("download_arxiv_pdfs", "summarizer")
graph_builder.add_conditional_edges("summarizer", decide_after_summary, {END: END})

# Deep Research Path (with Modification Loop)
graph_builder.add_edge("start_deep_research", "ask_plan_approval")
graph_builder.add_conditional_edges( # <<< Updated edge >>>
    "ask_plan_approval",
    decide_after_approval_prompt, # <<< Use new conditional logic >>>
    {
        "execute_search_plan": "execute_search_plan", # If 'yes'
        "get_plan_modifications": "get_plan_modifications", # If 'mod'
        END: END # If 'no'
    }
)
# Modification loop
graph_builder.add_edge("get_plan_modifications", "refine_plan_with_feedback")
graph_builder.add_edge("refine_plan_with_feedback", "ask_plan_approval") # Loop back to ask approval for refined plan

# Main research loop
graph_builder.add_edge("execute_search_plan", "evaluate_findings")
graph_builder.add_conditional_edges("evaluate_findings", decide_next_research_step, {"execute_search_plan": "execute_search_plan", "generate_final_report": "generate_final_report"})
graph_builder.add_edge("generate_final_report", END)

# Simple Paths
graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("coding_agent", END)


# Compile the graph
logger.info("Compiling agent graph...")
try: app = graph_builder.compile(); logger.info("Agent graph compiled successfully.")
except Exception as compile_e: logger.critical(f"Failed to compile agent graph: {compile_e}", exc_info=True); sys.exit(1)
# --- End Graph Definition ---


# --- Helper Function to Save Output Files ---
def save_output(run_dir: str, relative_path: str, data: Any):
    if not run_dir: logger.warning("Cannot save output: run_dir not set."); return
    filepath = os.path.join(run_dir, relative_path)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(data, str): f.write(data)
            else: json.dump(data, f, indent=2, default=str)
        logger.info(f"Output successfully saved to: {filepath}")
    except Exception as e: logger.error(f"Failed to save output to {filepath}: {e}", exc_info=True)

# --- Helper Function for CLI Input ---
def get_input(prompt: str) -> str:
    try: line = input(prompt); return line
    except (EOFError, KeyboardInterrupt): logger.info("Input interrupted, treating as 'quit'."); return 'quit'

# --- Helper Function to Format Results for History ---
def format_results_for_history(results: Optional[List[Dict[str, Any]]], max_to_show: int = 3) -> str:
    if not results: return "No results found."
    lines = ["Found results:"]; count = 0
    for i, res in enumerate(results):
        if count >= max_to_show: lines.append(f"... plus {len(results) - max_to_show} more."); break
        title = res.get('title', 'N/A'); authors = res.get('authors', [])
        authors_str = ", ".join(map(str, authors)) if authors else "N/A"
        source = res.get('source', 'N/A'); res_id = res.get('id', 'N/A')
        lines.append(f"{i+1}. {title} by {authors_str} ({source}: {res_id})")
        count += 1
    return "\n".join(lines)

# ==================================
# Main Execution Block (CLI)
# ==================================
if __name__ == "__main__":

    # --- Setup Run Directory & File Logging ---
    WORKPLACE_DIR = "workplace"; run_dir = None
    try:
        os.makedirs(WORKPLACE_DIR, exist_ok=True); timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(WORKPLACE_DIR, f"bioagent_run_{timestamp}")
        logs_dir = os.path.join(run_dir, "logs"); results_dir = os.path.join(run_dir, "results")
        os.makedirs(logs_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)
        log_filepath = os.path.join(logs_dir, "run.log"); file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'); file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler); logger.info("--- BioAgent CLI Run Started ---"); logger.info(f"Run directory created: {run_dir}")
        logger.info(f"File logging level: {logging.getLevelName(file_handler.level)} -> {log_filepath}")
        provider_for_log = get_config_value(config, "llm_provider", "unknown").lower(); logger.info(f"Main LLM Provider Configured: {provider_for_log}")
        coding_provider_setting = get_config_value(config, "coding_agent_settings.llm_provider", "default").lower()
        actual_coding_provider = type(coding_llm).__name__; logger.info(f"Coding LLM Provider Setting: '{coding_provider_setting}', Actual Used: {actual_coding_provider}")
        actual_dr_provider = type(deep_research_llm).__name__; logger.info(f"Deep Research LLM Actual Used: {actual_dr_provider}")

    except Exception as setup_e:
        log_func = logger.critical if 'logger' in locals() else print
        log_func(f"CRITICAL ERROR during setup: Failed to set up run environment: {setup_e}", exc_info=True)
        sys.exit(1)

    # --- Initialize Conversation State ---
    conversation_state = {"history": [], "last_search_results": None, "last_summary": None}
    MAX_HISTORY_TURNS = 5; interaction_count = 0

    # --- Main Interaction Loop ---
    while True:
        interaction_count += 1
        prompt_text = f"\n{COLOR_INPUT}Enter query {interaction_count} (or type 'quit'): {COLOR_RESET} "
        initial_query = get_input(prompt_text)
        if initial_query.strip().lower() == 'quit': logger.info("Quit command received. Exiting."); break
        if not initial_query.strip(): print(f"{COLOR_WARN}Please enter a query or message.{COLOR_RESET}"); interaction_count -= 1; continue

        logger.info(f"--- Starting Interaction #{interaction_count} ---"); logger.info(f"User Query: {initial_query}")
        input_for_graph = AgentState(
            query=initial_query.strip(), history=conversation_state["history"], run_dir=run_dir,
            refined_query=None, search_results=None, summary=None, chat_response=None,
            error=None, next_node=None, arxiv_results_found=False, download_preference=None,
            code_request=None, generated_code=None, generated_code_language=None,
            google_results=None, synthesized_report=None, route_intent=None,
            hypothesis=None, search_plan=[], evidence_log=[], research_iterations=0, # Ensure search_plan is list
            evaluation_summary=None, more_research_needed=False, plan_approved=None, plan_modifications=None # Init new fields
        )

        logger.info(f"Invoking agent graph...")
        final_state: Optional[AgentState] = None
        try:
            stream_config: Dict[str, Any] = {"recursion_limit": 35} # Increased limit slightly for mod loop
            for event in app.stream(input_for_graph, config=stream_config):
                node_name = list(event.keys())[0]; node_output_state = event[node_name]
                logger.debug(f"Node '{node_name}' executed."); final_state = node_output_state

            if final_state: logger.info("Graph execution complete.")
            else: logger.error("Graph execution finished, but no final state was captured."); final_state = input_for_graph; final_state["error"] = (final_state.get("error") or "") + "; Graph execution failed."

            # --- Output Handling Logic ---
            print(f"\n{COLOR_INFO}--- Agent Output (Interaction #{interaction_count}) ---{COLOR_RESET}")
            agent_response_for_history = None; output_message_for_console = None; saved_filename = None
            try: logger.debug("Final State Dump: %s", json.dumps(final_state, indent=2, default=str))
            except Exception as dump_e: logger.warning(f"Could not serialize final state for debug logging: {dump_e}")

            conversation_state["last_search_results"] = final_state.get("search_results")
            conversation_state["last_summary"] = final_state.get("summary")

            # --- Determine Output Based on Final State ---
            if final_state.get("error"):
                error_msg = f"An error occurred: {final_state['error']}"; output_message_for_console = f"{COLOR_ERROR}{error_msg}{COLOR_RESET}"; logger.error(f"Error(s) in final state: {final_state['error']}")
                agent_response_for_history = "Sorry, an error occurred."; save_output(run_dir, os.path.join("results", f"error_{interaction_count}.txt"), final_state['error'])
            elif final_state.get("synthesized_report"):
                report = final_state["synthesized_report"]
                if "placeholder final report" in report.lower(): output_message_for_console = f"{COLOR_WARN}Deep research finished (placeholder report).{COLOR_RESET}\n{COLOR_SYNTHESIS}{report}{COLOR_RESET}"; logger.warning("Deep research path completed with placeholder report.")
                else: output_message_for_console = f"{COLOR_SYNTHESIS}--- Deep Research Report ---{COLOR_RESET}\n{report}"; logger.info("Deep research report generated.")
                saved_filename = f"deep_research_report_{interaction_count}.txt"; save_output(run_dir, os.path.join("results", saved_filename), report)
                evidence_count = len(final_state.get("evidence_log", [])); iterations = final_state.get("research_iterations", 0)
                agent_response_for_history = f"Deep Research Report generated (saved to {saved_filename}). Based on {evidence_count} evidence items over {iterations} iteration(s)."
            elif final_state.get("route_intent") == "deep_research" and final_state.get("plan_approved") == "no":
                 output_message_for_console = f"{COLOR_WARN}Deep research plan not approved. Workflow terminated.{COLOR_RESET}"; logger.info("Deep research workflow terminated by user.")
                 agent_response_for_history = "Okay, I will not proceed with that research plan."
            elif final_state.get("generated_code"):
                code = final_state["generated_code"]; language = final_state.get("generated_code_language", "text"); extension = {"python": "py", "r": "R"}.get(language, "txt"); filename = f"generated_code_{interaction_count}.{extension}"
                output_message_for_console = f"{COLOR_OUTPUT}Generated Code ({language}):{COLOR_RESET}\n{COLOR_CODE}```{language}\n{code}\n```{COLOR_RESET}"; logger.info(f"Code ({language}) generated.")
                saved_filename = filename; save_output(run_dir, os.path.join("results", saved_filename), code); agent_response_for_history = f"Generated {language} code (saved to results/{saved_filename}):\n```\n{code[:200]}...\n```"; conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None
            elif final_state.get("summary"): # End of literature_search path ONLY
                summary = final_state["summary"]; results = final_state.get("search_results", []); refined_query = final_state.get('refined_query', 'N/A'); output_lines = []
                if results:
                    save_output(run_dir, os.path.join("results", f"search_results_{interaction_count}.json"), results); output_lines.append(f"{COLOR_OUTPUT}Found {len(results)} lit results (refined: '{refined_query}'):{COLOR_RESET}")
                    for i, result in enumerate(results): output_lines.append(f"\n{Style.BRIGHT}Res {i+1} ({result.get('source')}){Style.NORMAL}"); output_lines.append(f"{COLOR_OUTPUT} Title: {result.get('title')}{COLOR_RESET}"); output_lines.append(f" ID: {result.get('id')} URL: {result.get('url')}");
                    if result.get("local_pdf_path"): output_lines.append(f" {COLOR_FILE}PDF: {result['local_pdf_path']}{COLOR_RESET}")
                    output_lines.append(f"\n{COLOR_SUMMARY}--- Summary ---{COLOR_RESET}"); output_lines.append(f"{summary}"); logger.info("Summary generated.")
                    saved_filename = f"summary_{interaction_count}.txt"; save_output(run_dir, os.path.join("results", saved_filename), summary); results_str = format_results_for_history(results, max_to_show=MAX_ABSTRACTS_TO_SUMMARIZE); agent_response_for_history = f"Summary:\n{summary}\n\nBased on:\n{results_str}"
                else: output_message_for_console = f"{COLOR_WARN}Summary generated but no results found.{COLOR_RESET}"; logger.warning("Summary node ran without results."); agent_response_for_history = summary; saved_filename = f"summary_{interaction_count}.txt"; save_output(run_dir, os.path.join("results", saved_filename), summary); conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None
                if output_lines: output_message_for_console = "\n".join(output_lines)
            elif final_state.get("chat_response"):
                agent_response_for_history = final_state["chat_response"]; output_message_for_console = f"\n{COLOR_OUTPUT}BioAgent: {agent_response_for_history}{COLOR_RESET}"; logger.info("Chat response generated.")
                saved_filename = f"chat_response_{interaction_count}.txt"; save_output(run_dir, os.path.join("results", saved_filename), agent_response_for_history); conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None
            elif isinstance(final_state.get("search_results"), list) and not final_state.get("search_results") and final_state.get("route_intent") == "literature_search" and not final_state.get("summary"):
                refined_query = final_state.get('refined_query', 'N/A'); no_results_msg = f"No lit results found for: '{refined_query}'"; output_message_for_console = f"{COLOR_WARN}{no_results_msg}{COLOR_RESET}"; logger.info(no_results_msg); agent_response_for_history = no_results_msg; saved_filename=f"search_results_{interaction_count}.txt"; save_output(run_dir, os.path.join("results", saved_filename), no_results_msg); conversation_state["last_search_results"] = []; conversation_state["last_summary"] = None
            elif not final_state.get("error"): # Fallback
                no_output_msg = "Agent finished processing."; output_message_for_console = f"{COLOR_WARN}{no_output_msg}{COLOR_RESET}"; logger.warning("No standard output generated."); agent_response_for_history = "Processing complete."; saved_filename=f"output_{interaction_count}_final_state.json"; save_output(run_dir, os.path.join("results", saved_filename), final_state)

            # --- Print Output to Console ---
            if output_message_for_console: print(output_message_for_console)
            elif not final_state.get("error"): print(f"{COLOR_INFO}Processing complete.{COLOR_RESET}")

            # --- Update Conversation History ---
            if agent_response_for_history is not None:
                conversation_state["history"].append((initial_query.strip(), agent_response_for_history));
                if len(conversation_state["history"]) > MAX_HISTORY_TURNS: conversation_state["history"] = conversation_state["history"][-MAX_HISTORY_TURNS:]; logger.debug(f"History trimmed.")
                save_output(run_dir, os.path.join("logs", "conversation_history.json"), conversation_state["history"])

        except Exception as e:
            tb_str = traceback.format_exc(); error_msg = f"FATAL: Unexpected error during interaction processing: {e}\nTraceback:\n{tb_str}"; print(f"\n{COLOR_ERROR}{Style.BRIGHT}{error_msg}{COLOR_RESET}"); logger.critical(f"Unexpected error during interaction processing: {e}", exc_info=True)
            try: state_at_error = final_state if final_state is not None else input_for_graph; logger.error("State at time of critical error: %s", json.dumps(state_at_error, indent=2, default=str))
            except Exception as dump_e: logger.error(f"Could not retrieve or serialize state at time of critical error: {dump_e}")

    # --- End of Main Loop ---
    logger.info("--- BioAgent Run Ended ---")
    print(f"\n{COLOR_INFO}BioAgent session ended.{COLOR_RESET}")

# === End of main.py ===
