import os
import sys
import json
import datetime
import logging
from io import StringIO
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple
from functools import partial

# Import colorama
from colorama import init as colorama_init, Fore, Style

from dotenv import load_dotenv

# LLM Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END, START

# Import config loader and utility functions/classes
from src.core.config_loader import load_config, get_config_value
# Import AgentState definition
try: from src.core.state import AgentState
except ImportError: print("Error: src.core.state not found. Please create the file."); sys.exit(1)
# Import tools and agents
from src.tools.llm_utils import initialize_llm
from src.agents.router import route_query, decide_next_node, decide_after_refine, decide_after_summary
from src.agents.refine import refine_query_node
from src.agents.literature import call_literature_agent
from src.agents.download import ask_download_preference, download_arxiv_pdfs, should_download
from src.agents.google_search import call_google_search_agent
from src.agents.summarize import summarize_results
from src.agents.synthesize import synthesize_results_agent
from src.agents.chat import call_chat_agent
from src.agents.coding import call_coding_agent

# --- Initialize Colorama ---
colorama_init(autoreset=True)

# --- Color Scheme ---
COLOR_INFO = Fore.CYAN; COLOR_INPUT = Fore.YELLOW; COLOR_OUTPUT = Fore.GREEN
COLOR_SUMMARY = Fore.MAGENTA; COLOR_ERROR = Fore.RED; COLOR_WARN = Fore.YELLOW
COLOR_DEBUG = Fore.BLUE; COLOR_RESET = Style.RESET_ALL; COLOR_FILE = Fore.LIGHTBLUE_EX
COLOR_QUESTION = Fore.BLUE + Style.BRIGHT; COLOR_CODE = Fore.LIGHTYELLOW_EX; COLOR_SYNTHESIS = Fore.LIGHTGREEN_EX

# --- Custom Colored Logging Formatter ---
class ColoredFormatter(logging.Formatter):
    LOG_COLORS = { logging.DEBUG: COLOR_DEBUG, logging.INFO: COLOR_INFO,
                   logging.WARNING: COLOR_WARN, logging.ERROR: COLOR_ERROR,
                   logging.CRITICAL: Fore.RED + Style.BRIGHT, }
    def format(self, record):
        log_color = self.LOG_COLORS.get(record.levelno, COLOR_RESET)
        log_fmt = f"{log_color}{record.levelname}: {record.getMessage()}{COLOR_RESET}"
        return log_fmt

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__); logger.propagate = False
console_handler = logging.StreamHandler(sys.stdout); console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter()); logger.addHandler(console_handler)
# File handler added in main block

# --- Configuration Loading ---
config = load_config(); logger.info("Configuration loaded.")

# --- Environment Setup ---
load_dotenv(); ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
if not ENTREZ_EMAIL: logger.critical("ENTREZ_EMAIL not found in .env file."); sys.exit(1)
from Bio import Entrez; Entrez.email = ENTREZ_EMAIL; logger.info(f"Entrez Email set to: {ENTREZ_EMAIL}")

# --- LLM Instantiation ---
llm = initialize_llm(config, "llm_provider", "llm_settings")
if llm is None: sys.exit("Failed to initialize main LLM.")
coding_llm = initialize_llm(config, "coding_agent_settings.llm_provider", "coding_agent_settings")
if coding_llm is None: logger.warning("Failed to initialize specific coding LLM, falling back to main LLM."); coding_llm = llm

# --- Search Settings ---
MAX_RESULTS_PER_SOURCE = get_config_value(config, "search_settings.max_results_per_source", 3)
MAX_ABSTRACTS_TO_SUMMARIZE = get_config_value(config, "search_settings.max_abstracts_to_summarize", 3)
NUM_GOOGLE_RESULTS = get_config_value(config, "search_settings.num_google_results", 5)
logger.info(f"Search settings: max_lit_results={MAX_RESULTS_PER_SOURCE}, max_google={NUM_GOOGLE_RESULTS}, max_abstracts={MAX_ABSTRACTS_TO_SUMMARIZE}")

# --- Prompt Templates ---
ROUTING_PROMPT_TEMPLATE = get_config_value(config, "prompts.routing_prompt", "Error: Routing prompt not found.")
REFINEMENT_PROMPT_TEMPLATE = get_config_value(config, "prompts.refinement_prompt", "Error: Refinement prompt not found.")
SUMMARIZATION_PROMPT_TEMPLATE = get_config_value(config, "prompts.summarization_prompt", "Error: Summarization prompt not found.")
CODE_GENERATION_PROMPT_TEMPLATE = get_config_value(config, "prompts.code_generation_prompt", "Error: Code generation prompt not found.")
SYNTHESIS_PROMPT_TEMPLATE = get_config_value(config, "prompts.synthesis_prompt", "Error: Synthesis prompt not found.")

# --- Graph Definition ---
# (Unchanged)
graph_builder = StateGraph(AgentState); graph_builder.add_node("router", partial(route_query, llm=llm, routing_prompt_template=ROUTING_PROMPT_TEMPLATE))
graph_builder.add_node("refine_query", partial(refine_query_node, llm=llm, refinement_prompt_template=REFINEMENT_PROMPT_TEMPLATE))
graph_builder.add_node("literature_agent", partial(call_literature_agent, max_pubmed=MAX_RESULTS_PER_SOURCE, max_arxiv=MAX_RESULTS_PER_SOURCE))
graph_builder.add_node("ask_download_preference", ask_download_preference); graph_builder.add_node("download_arxiv_pdfs", download_arxiv_pdfs)
graph_builder.add_node("summarizer", partial(summarize_results, llm=llm, summarization_prompt_template=SUMMARIZATION_PROMPT_TEMPLATE, max_abstracts=MAX_ABSTRACTS_TO_SUMMARIZE))
graph_builder.add_node("google_search", partial(call_google_search_agent, num_results=NUM_GOOGLE_RESULTS))
graph_builder.add_node("synthesizer", partial(synthesize_results_agent, llm=llm, synthesis_prompt_template=SYNTHESIS_PROMPT_TEMPLATE, app_config=config))
graph_builder.add_node("chat_agent", partial(call_chat_agent, llm=llm)); graph_builder.add_node("coding_agent", partial(call_coding_agent, coding_llm=coding_llm, code_generation_prompt_template=CODE_GENERATION_PROMPT_TEMPLATE))
graph_builder.add_edge(START, "router"); graph_builder.add_conditional_edges("router", decide_next_node, {"refine_query": "refine_query", "chat_agent": "chat_agent", "coding_agent": "coding_agent", END: END})
graph_builder.add_conditional_edges("refine_query", decide_after_refine, {"literature_agent": "literature_agent", END: END})
graph_builder.add_edge("literature_agent", "ask_download_preference"); graph_builder.add_conditional_edges("ask_download_preference", should_download, {"download_arxiv_pdfs": "download_arxiv_pdfs", "google_search": "google_search", "summarizer": "summarizer"})
graph_builder.add_edge("download_arxiv_pdfs", "summarizer"); graph_builder.add_conditional_edges("summarizer", decide_after_summary, {"google_search": "google_search", END: END})
graph_builder.add_edge("google_search", "synthesizer"); graph_builder.add_edge("synthesizer", END); graph_builder.add_edge("chat_agent", END); graph_builder.add_edge("coding_agent", END)
app = graph_builder.compile(); logger.info("Agent graph compiled.")

# --- Function to save results ---
def save_output(run_dir: str, relative_path: str, data: Any):
    # (Unchanged)
    if not run_dir: return
    filepath = os.path.join(run_dir, relative_path)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(data, str): f.write(data)
            else: json.dump(data, f, indent=2, default=str)
        logger.info(f"Output saved to: {filepath}")
    except Exception as e: logger.error(f"Failed to save output to {filepath}: {e}", exc_info=True)

# --- Function for Single-line Input ---
def get_input(prompt: str) -> str:
    # (Unchanged)
    try: line = input(prompt); return line
    except EOFError: return 'quit'

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Setup Run Directory & File Logging ---
    # (Unchanged)
    WORKPLACE_DIR = "workplace"; run_dir = None
    try:
        os.makedirs(WORKPLACE_DIR, exist_ok=True); timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(WORKPLACE_DIR, f"{timestamp}_run")
        logs_dir = os.path.join(run_dir, "logs"); results_dir = os.path.join(run_dir, "results"); temp_data_dir = os.path.join(run_dir, "temp_data")
        os.makedirs(logs_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True); os.makedirs(temp_data_dir, exist_ok=True)
        log_filepath = os.path.join(logs_dir, "run.log"); file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'); file_handler.setFormatter(file_formatter)
        module_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('src.')]
        if not module_loggers: module_loggers.append(logger)
        for mod_logger in module_loggers:
             if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_filepath for h in mod_logger.handlers):
                  mod_logger.addHandler(file_handler)
        logger.info("--- BioAgent CLI Run Started ---"); logger.info(f"Run directory created: {run_dir}")
        logger.info(f"File logging to: {log_filepath}")
        provider_for_log = get_config_value(config, "llm_provider", "unknown").lower(); logger.info(f"LLM Provider Config: {provider_for_log}")
        coding_provider_for_log = get_config_value(config, "coding_agent_settings.llm_provider", "default").lower()
        if coding_provider_for_log == 'default': coding_provider_for_log = provider_for_log
        logger.info(f"Coding LLM Provider Config: {coding_provider_for_log}")
    except Exception as setup_e: logger.critical(f"Failed to set up run environment: {setup_e}", exc_info=True); sys.exit(1)

    conversation_state = {"history": [], "last_search_results": None, "last_summary": None}
    MAX_HISTORY_TURNS = 5; interaction_count = 0

    while True:
        interaction_count += 1
        prompt_text = f"\n{COLOR_INPUT}Enter query {interaction_count} (or type 'quit'): {COLOR_RESET}"
        initial_query = get_input(prompt_text)
        if initial_query.strip().lower() == 'quit': logger.info("Quit command received. Exiting."); break
        if not initial_query.strip(): print(f"{COLOR_WARN}Please enter a query or message.{COLOR_RESET}"); interaction_count -= 1; continue
        logger.info(f"--- Starting Interaction #{interaction_count} ---")

        input_for_graph = AgentState(
            query=initial_query.strip(), history=conversation_state["history"], run_dir=run_dir,
            refined_query=None, search_results=None, summary=None, chat_response=None,
            error=None, next_node=None, arxiv_results_found=False, download_preference=None,
            code_request=None, generated_code=None, generated_code_language=None,
            google_results=None, synthesized_report=None, route_intent=None
        )

        logger.info(f"Invoking agent graph for query:\n{initial_query[:200]}...")
        final_state = None
        try:
            # Use stream to get intermediate states
            for event in app.stream(input_for_graph):
                final_state = list(event.values())[0]

            if final_state: logger.info("Graph execution complete.")
            else: logger.error("Graph execution did not produce a final state."); continue

            print(f"\n{COLOR_INFO}--- Agent Output (Interaction #{interaction_count}) ---{COLOR_RESET}")
            agent_response = None
            try: logger.debug("Final State: %s", json.dumps(final_state, indent=2, default=str))
            except Exception as dump_e: logger.warning(f"Could not serialize final state for logging: {dump_e}")

            # --- REVISED Output Handling Logic ---
            output_message = None
            saved_filename = None

            # Store results/summary from final_state into conversation_state *before* handling output
            # This ensures they are available even if only summary is in final_state
            if final_state.get("search_results") is not None:
                 conversation_state["last_search_results"] = final_state.get("search_results")
            if final_state.get("summary") is not None:
                 conversation_state["last_summary"] = final_state.get("summary")

            # Check for errors first
            if final_state.get("error"):
                error_msg = f"An error occurred: {final_state['error']}"
                output_message = f"{COLOR_ERROR}{error_msg}{COLOR_RESET}"
                logger.error(error_msg)
                agent_response = "Sorry, an error occurred."
                save_output(run_dir, os.path.join("results", f"error_{interaction_count}.txt"), error_msg)

            # Check for primary outputs in order of preference
            elif final_state.get("synthesized_report"):
                report = final_state["synthesized_report"]
                output_message = f"{COLOR_SYNTHESIS}--- Synthesized Report ---{COLOR_RESET}\n{COLOR_SYNTHESIS}{report}{COLOR_RESET}"
                logger.info("Synthesized report generated and displayed.")
                saved_filename = f"synthesized_report_{interaction_count}.txt"
                save_output(run_dir, os.path.join("results", saved_filename), report)
                agent_response = report
                # Note: last_search_results/last_summary already updated above if they were in final_state

            elif final_state.get("generated_code"):
                code = final_state["generated_code"]; language = final_state.get("generated_code_language", "text"); extension = {"python": "py", "r": "R"}.get(language, "txt"); filename = f"generated_code_{interaction_count}.{extension}"
                output_message = f"{COLOR_OUTPUT}Generated Code ({language}):{COLOR_RESET}\n{COLOR_CODE}```{language}\n{code}\n```"
                logger.info(f"Code ({language}) generated and displayed.")
                saved_filename = filename
                save_output(run_dir, os.path.join("results", saved_filename), code);
                agent_response = f"Generated {language} code snippet (saved to results/{saved_filename})."
                conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None # Clear search context

            elif final_state.get("summary"): # Check summary only if no synthesis/code
                summary = final_state["summary"]
                # <<< FIX: Get results from conversation_state, which was updated above >>>
                results = conversation_state.get("last_search_results", [])
                refined_query = final_state.get('refined_query', 'N/A') # Refined query is likely in final_state
                output_lines = []

                if results: # Only show summary if there were associated results
                    output_lines.append(f"{COLOR_OUTPUT}Found {len(results)} literature results (using refined query: '{refined_query}'):{COLOR_RESET}")
                    # Save search results (if not already saved by literature agent - maybe remove save there?)
                    save_output(run_dir, os.path.join("results", f"search_results_{interaction_count}.json"), results);
                    for i, result in enumerate(results):
                        output_lines.append(f"\n{Style.BRIGHT}--- Result {i+1} ({result.get('source', 'N/A')}) ---{Style.NORMAL}")
                        output_lines.append(f"{COLOR_OUTPUT}  Title: {result.get('title', 'N/A')}{COLOR_RESET}")
                        output_lines.append(f"  ID: {result.get('id', 'N/A')}"); output_lines.append(f"  URL: {result.get('url', '#')}")
                        if result.get("local_pdf_path"): output_lines.append(f"  {COLOR_FILE}Downloaded PDF: {result['local_pdf_path']}{COLOR_RESET}")
                    # Append summary
                    output_lines.append(f"\n{COLOR_SUMMARY}--- Summary of Results ---{COLOR_RESET}"); output_lines.append(f"{COLOR_SUMMARY}{summary}{COLOR_RESET}")
                    logger.info("Summary generated and displayed.")
                    saved_filename = f"summary_{interaction_count}.txt"
                    save_output(run_dir, os.path.join("results", saved_filename), summary);
                    agent_response = summary # Use summary for history
                else: # Case where summary node ran but no results were found/stored (less likely now)
                    output_message = f"{COLOR_WARN}Summary generated but no associated search results found.{COLOR_RESET}"
                    logger.warning("Summary node ran without associated search results.")
                    agent_response = summary
                    saved_filename = f"summary_{interaction_count}.txt"
                    save_output(run_dir, os.path.join("results", saved_filename), summary);
                    conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None # Clear search context

                if output_lines: output_message = "\n".join(output_lines)

            elif final_state.get("chat_response"):
                agent_response = final_state["chat_response"]; output_message = f"\n{COLOR_OUTPUT}BioAgent: {agent_response}{COLOR_RESET}"
                logger.info("Chat response generated and displayed.")
                saved_filename = f"chat_response_{interaction_count}.txt"
                save_output(run_dir, os.path.join("results", saved_filename), agent_response);
                conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None

            # Handle case where literature search ran but found nothing
            # Check search_results is empty list, not just None
            elif isinstance(final_state.get("search_results"), list) and not final_state.get("search_results"):
                 refined_query = final_state.get('refined_query', 'N/A');
                 no_results_msg_local = f"No literature results found from PubMed or ArXiv for refined query: '{refined_query}'"
                 output_message = f"{COLOR_WARN}{no_results_msg_local}{COLOR_RESET}";
                 logger.info(f"No literature results found for refined query: '{refined_query}'")
                 agent_response = no_results_msg_local; saved_filename=f"search_results_{interaction_count}.txt"; save_output(run_dir, os.path.join("results", saved_filename), no_results_msg_local)
                 conversation_state["last_search_results"] = []; conversation_state["last_summary"] = None

            # Fallback if absolutely no other output generated and no error
            elif not final_state.get("error"):
                 no_output_msg_local = "No specific output generated."; output_message = f"{COLOR_WARN}{no_output_msg_local}{COLOR_RESET}"
                 logger.warning("Graph finished without error but no standard output produced.")
                 agent_response = no_output_msg_local; saved_filename=f"output_{interaction_count}.txt"; save_output(run_dir, os.path.join("results", saved_filename), no_output_msg_local)
                 conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None

            # Print the final assembled output message
            if output_message: print(output_message)

            # --- Update History ---
            if agent_response is not None:
                 # Use the stripped initial query for history consistency
                 conversation_state["history"].append((initial_query.strip(), agent_response));
                 if len(conversation_state["history"]) > MAX_HISTORY_TURNS: conversation_state["history"] = conversation_state["history"][-MAX_HISTORY_TURNS:]
                 save_output(run_dir, os.path.join("logs", "conversation_history.json"), conversation_state["history"])

        except Exception as e:
            tb_str = traceback.format_exc(); error_msg = f"An unexpected error occurred: {e}\nTraceback:\n{tb_str}"
            print(f"\n{COLOR_ERROR}{error_msg}{COLOR_RESET}"); logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            try: state_at_error = final_state if final_state is not None else input_for_graph; logger.error("State at time of error: %s", json.dumps(state_at_error, indent=2, default=str))
            except Exception as dump_e: logger.error(f"Could not retrieve or serialize state at time of error: {dump_e}")

    logger.info("--- BioAgent Run Ended ---"); print(f"\n{COLOR_INFO}BioAgent session ended.{COLOR_RESET}")

