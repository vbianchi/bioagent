import os
import sys
import json
import datetime
import logging
from io import StringIO
import re
import traceback # Keep traceback module
from typing import TypedDict, List, Dict, Any, Optional, Tuple

# Import colorama
from colorama import init as colorama_init, Fore, Style

from dotenv import load_dotenv

# LLM Imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END, START

# Import BioPython, ArXiv, Pandas
from Bio import Entrez, Medline
import arxiv
import pandas as pd

# Import config loader
from src.core.config_loader import load_config, get_config_value

# --- Initialize Colorama ---
colorama_init(autoreset=True)

# --- Color Scheme ---
COLOR_INFO = Fore.CYAN; COLOR_INPUT = Fore.YELLOW; COLOR_OUTPUT = Fore.GREEN
COLOR_SUMMARY = Fore.MAGENTA; COLOR_ERROR = Fore.RED; COLOR_WARN = Fore.YELLOW
COLOR_DEBUG = Fore.BLUE; COLOR_RESET = Style.RESET_ALL; COLOR_FILE = Fore.LIGHTBLUE_EX
COLOR_QUESTION = Fore.BLUE + Style.BRIGHT; COLOR_CODE = Fore.LIGHTYELLOW_EX

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
load_dotenv(); OPENAI_API_KEY = os.getenv("OPENAI_API_KEY"); GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
if not ENTREZ_EMAIL: logger.critical("ENTREZ_EMAIL not found in .env file."); sys.exit(1)
Entrez.email = ENTREZ_EMAIL; logger.info(f"Entrez Email set to: {ENTREZ_EMAIL}")

# --- LLM Instantiation ---
def initialize_llm(provider_config_path: str, settings_config_path: str) -> Optional[BaseChatModel]:
    # (Unchanged)
    provider = get_config_value(config, provider_config_path, "default").lower()
    if provider == "default":
        provider = get_config_value(config, "llm_provider", "openai").lower()
        provider_config_path = "llm_provider"; settings_config_path = "llm_settings"
    logger.info(f"Attempting to initialize LLM for '{provider_config_path}': provider={provider}")
    temp = get_config_value(config, f"{settings_config_path}.temperature", 0); llm_instance = None
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY: logger.error(f"Provider is '{provider}' but OPENAI_API_KEY not found."); return None
        model = get_config_value(config, f"{settings_config_path}.openai_model_name", "gpt-3.5-turbo")
        try: llm_instance = ChatOpenAI(model=model, temperature=temp, openai_api_key=OPENAI_API_KEY); logger.info(f"Initialized OpenAI LLM: model={model}, temp={temp}")
        except Exception as e: logger.error(f"Error initializing OpenAI LLM: {e}", exc_info=True)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not GOOGLE_API_KEY: logger.error(f"Provider is '{provider}' but GOOGLE_API_KEY not found."); return None
        model = get_config_value(config, f"{settings_config_path}.gemini_model_name", "gemini-1.5-flash-latest")
        try: llm_instance = ChatGoogleGenerativeAI(model=model, temperature=temp, google_api_key=GOOGLE_API_KEY); logger.info(f"Initialized Google Gemini LLM: model={model}, temp={temp}")
        except Exception as e: logger.error(f"Error initializing Google Gemini LLM: {e}", exc_info=True)
    elif provider == "ollama":
        try: from langchain_ollama import ChatOllama
        except ImportError: logger.error("langchain-ollama package not found."); return None
        model = get_config_value(config, f"{settings_config_path}.ollama_model_name", "gemma3")
        base_url = get_config_value(config, f"{settings_config_path}.ollama_base_url")
        try:
            init_params = {"model": model, "temperature": temp};
            if base_url: init_params["base_url"] = base_url
            llm_instance = ChatOllama(**init_params); llm_instance.invoke("test connection")
            logger.info(f"Initialized Ollama LLM: model={model}, temp={temp}, base_url={base_url or 'default'}")
        except Exception as e: logger.error(f"Error initializing/connecting to Ollama LLM: {e}", exc_info=True); logger.error(f"Ensure Ollama is running and model '{model}' is available.")
    else: logger.error(f"Unknown LLM provider '{provider}' specified for '{provider_config_path}'.")
    return llm_instance

llm = initialize_llm("llm_provider", "llm_settings")
if llm is None: sys.exit("Failed to initialize main LLM.")
coding_llm = initialize_llm("coding_agent_settings.llm_provider", "coding_agent_settings")
if coding_llm is None: logger.warning("Failed to initialize specific coding LLM, falling back to main LLM."); coding_llm = llm

# --- Search Settings ---
MAX_RESULTS_PER_SOURCE = get_config_value(config, "search_settings.max_results_per_source", 3)
MAX_ABSTRACTS_TO_SUMMARIZE = get_config_value(config, "search_settings.max_abstracts_to_summarize", 3)
MAX_RESULTS_PUBMED = MAX_RESULTS_PER_SOURCE; MAX_RESULTS_ARXIV = MAX_RESULTS_PER_SOURCE
logger.info(f"Search settings: max_results_per_source={MAX_RESULTS_PER_SOURCE}, max_abstracts_to_summarize={MAX_ABSTRACTS_TO_SUMMARIZE}")

# --- Prompt Templates ---
ROUTING_PROMPT_TEMPLATE = get_config_value(config, "prompts.routing_prompt", "Error: Routing prompt not found.")
REFINEMENT_PROMPT_TEMPLATE = get_config_value(config, "prompts.refinement_prompt", "Error: Refinement prompt not found.")
SUMMARIZATION_PROMPT_TEMPLATE = get_config_value(config, "prompts.summarization_prompt", "Error: Summarization prompt not found.")
CODE_GENERATION_PROMPT_TEMPLATE = get_config_value(config, "prompts.code_generation_prompt", "Error: Code generation prompt not found.")

# --- Agent State Definition ---
class AgentState(TypedDict):
    query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
    chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]
    run_dir: str; arxiv_results_found: bool; download_preference: Optional[str]
    code_request: Optional[str]; generated_code: Optional[str]
    generated_code_language: Optional[str]

# --- Helper Functions ---
def _search_pubmed(query: str, max_results: int) -> List[Dict[str, Any]]:
    # (Unchanged)
    logger.info(f"Searching PubMed for '{query}' (max_results={max_results})...")
    results = [];
    try:
        handle_search = Entrez.esearch(db="pubmed", term=query, retmax=max_results); search_results_entrez = Entrez.read(handle_search); handle_search.close()
        id_list = search_results_entrez["IdList"];
        if not id_list: logger.info("No results found on PubMed."); return []
        logger.info(f"Found {len(id_list)} PMIDs on PubMed. Fetching details...")
        handle_fetch = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text"); records_text = handle_fetch.read(); handle_fetch.close()
        records = Medline.parse(StringIO(records_text)); count = 0
        for record in records:
            results.append({"id": record.get("PMID", "N/A"), "source": "PubMed", "title": record.get("TI", "No title found"), "abstract": record.get("AB", "No abstract found"), "journal": record.get("JT", "N/A"), "authors": record.get("AU", []), "url": f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}/"}); count += 1
        logger.info(f"Successfully fetched/parsed {count} PubMed records.")
    except Exception as e: logger.error(f"Error during PubMed search: {e}", exc_info=True)
    return results

def _search_arxiv(query: str, max_results: int) -> List[Dict[str, Any]]:
    # (Unchanged)
    logger.info(f"Searching ArXiv for '{query}' (max_results={max_results})...")
    results = []
    try:
        client = arxiv.Client(); search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance); arxiv_results = list(client.results(search))
        if not arxiv_results: logger.info("No results found on ArXiv."); return []
        logger.info(f"Found {len(arxiv_results)} results on ArXiv.")
        for result in arxiv_results:
            results.append({"id": result.entry_id.split('/')[-1], "source": "ArXiv", "title": result.title, "abstract": result.summary.replace('\n', ' '), "authors": [str(author) for author in result.authors], "published": str(result.published), "url": result.pdf_url, "pdf_url": result.pdf_url})
    except Exception as e: logger.error(f"Error during ArXiv search: {e}", exc_info=True)
    return results

# --- Agent Nodes ---
def call_literature_agent(state: AgentState) -> AgentState:
    # (Unchanged)
    logger.info("--- Calling Literature Agent ---"); original_query = state['query']; run_dir = state['run_dir']
    logger.info(f"Received original query: {original_query}"); error_message = None; refined_query_for_search = original_query; combined_results = []; arxiv_found = False
    try:
        logger.info("Refining query for literature search...")
        refinement_prompt = REFINEMENT_PROMPT_TEMPLATE.format(query=original_query)
        try: refinement_response = llm.invoke(refinement_prompt); refined_query_for_search = refinement_response.content.strip(); logger.info(f"Refined query: {refined_query_for_search}")
        except Exception as refine_e: logger.warning(f"LLM query refinement failed: {refine_e}. Using original query."); error_message = f"Query refinement failed: {refine_e}. "
        pubmed_results = _search_pubmed(refined_query_for_search, MAX_RESULTS_PUBMED); arxiv_results = _search_arxiv(refined_query_for_search, MAX_RESULTS_ARXIV)
        combined_results.extend(pubmed_results); combined_results.extend(arxiv_results); logger.info(f"Total combined results: {len(combined_results)}")
        if any(r['source'] == 'ArXiv' for r in combined_results): arxiv_found = True; logger.info("ArXiv results found.")
    except Exception as e: search_error = f"Literature search failed: {str(e)}"; logger.error(search_error, exc_info=True); error_message = (error_message + search_error) if error_message else search_error
    return { **state, "refined_query": refined_query_for_search, "search_results": combined_results, "error": error_message, "arxiv_results_found": arxiv_found }

def ask_download_preference(state: AgentState) -> AgentState:
    # (Unchanged)
    logger.info("--- Asking Download Preference ---"); arxiv_found = state.get("arxiv_results_found", False); preference = "no"
    if arxiv_found:
        try:
            user_input = input(f"{COLOR_QUESTION}ArXiv papers found. Download PDFs? (yes/no): {COLOR_RESET}").strip().lower()
            if user_input == "yes" or user_input == "y": preference = "yes"; logger.info("User chose to download ArXiv PDFs.")
            else: logger.info("User chose not to download ArXiv PDFs.")
        except EOFError: logger.warning("EOF received asking download preference. Defaulting to 'no'."); preference = "no"
    else: logger.info("No ArXiv papers found, skipping download prompt."); preference = "no"
    return {**state, "download_preference": preference}

def download_arxiv_pdfs(state: AgentState) -> AgentState:
    # (Unchanged)
    logger.info("--- Downloading ArXiv PDFs ---"); run_dir = state['run_dir']; search_results = state.get("search_results", [])
    results_save_dir = os.path.join(run_dir, "results"); download_count = 0; updated_search_results = []
    for result in search_results:
        if result.get("source") == "ArXiv":
            arxiv_id = result.get("id"); pdf_url = result.get("pdf_url")
            if not arxiv_id or not pdf_url: logger.warning(f"Skipping ArXiv result: {result.get('title')}"); updated_search_results.append(result); continue
            pdf_filename = f"arxiv_{arxiv_id}.pdf"; pdf_saveloc = os.path.join(results_save_dir, pdf_filename); local_pdf_path = None
            try:
                logger.info(f"Attempting to download ArXiv PDF: {arxiv_id}"); client = arxiv.Client(); fetch_search = arxiv.Search(id_list=[arxiv_id])
                fetched_results = list(client.results(fetch_search))
                if fetched_results: fetched_results[0].download_pdf(dirpath=results_save_dir, filename=pdf_filename); logger.info(f"Successfully downloaded PDF to: {pdf_saveloc}"); local_pdf_path = pdf_saveloc; download_count += 1
                else: logger.error(f"Could not re-fetch ArXiv result ID {arxiv_id} to download.")
            except Exception as pdf_e: logger.error(f"Failed to download ArXiv PDF {arxiv_id}: {pdf_e}", exc_info=True)
            result_copy = result.copy(); result_copy["local_pdf_path"] = local_pdf_path; updated_search_results.append(result_copy)
        else: updated_search_results.append(result)
    logger.info(f"Attempted to download PDFs. Successful downloads: {download_count}")
    return {**state, "search_results": updated_search_results}

def summarize_results(state: AgentState) -> AgentState:
    # (Unchanged)
    logger.info("--- Calling Summarizer ---"); search_results = state.get("search_results"); original_query = state.get("query")
    error_message = state.get("error"); summary_text = None
    if not search_results: logger.info("No search results to summarize."); return {**state, "summary": None}
    abstracts_to_summarize = []; logger.info(f"Preparing abstracts for summarization (max {MAX_ABSTRACTS_TO_SUMMARIZE})...")
    for i, result in enumerate(search_results):
        if i >= MAX_ABSTRACTS_TO_SUMMARIZE: break
        abstract = result.get("abstract");
        if abstract and abstract != "No abstract found": abstracts_to_summarize.append(f"Abstract {i+1} (Source: {result.get('source', 'N/A')}, ID: {result.get('id', 'N/A')}):\n{abstract}\n")
    if not abstracts_to_summarize: logger.info("No valid abstracts found."); return {**state, "summary": "No abstracts available to summarize."}
    abstracts_text = "\n---\n".join(abstracts_to_summarize); summarization_prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(query=original_query, abstracts_text=abstracts_text)
    logger.info(f"Sending {len(abstracts_to_summarize)} abstracts to LLM for summarization...")
    try: response = llm.invoke(summarization_prompt); summary_text = response.content.strip(); logger.info("LLM Summary generated.")
    except Exception as e: summary_error = f"Summarization failed: {str(e)}"; logger.error(summary_error, exc_info=True); error_message = (error_message + summary_error) if error_message else summary_error; summary_text = "Sorry, I couldn't generate a summary."
    return {**state, "summary": summary_text, "error": error_message}

def call_chat_agent(state: AgentState) -> AgentState:
    # (Unchanged)
    logger.info("--- Calling Chat Agent ---"); query = state['query']; history = state['history']
    error_message = None; chat_response_text = "Sorry, I couldn't generate a response."
    formatted_history = [];
    for user_msg, ai_msg in history: formatted_history.extend([HumanMessage(content=user_msg), AIMessage(content=ai_msg)])
    formatted_history.append(HumanMessage(content=query))
    logger.info(f"Received query: {query}"); logger.info(f"Using history (last {len(history)} turns)")
    try: response = llm.invoke(formatted_history); chat_response_text = response.content.strip(); logger.info(f"LLM chat response generated.")
    except Exception as e: error_message = f"Chat generation failed: {str(e)}"; logger.error(error_message, exc_info=True)
    return {**state, "chat_response": chat_response_text, "error": error_message}

def call_coding_agent(state: AgentState) -> AgentState:
    # (Unchanged)
    logger.info("--- Calling Coding Agent ---"); query = state['query']; history = state['history']
    error_message = None; generated_code_text = None; detected_language = "text"
    logger.info(f"Received code request: {query}")
    messages = [SystemMessage(content=CODE_GENERATION_PROMPT_TEMPLATE)]
    for user_msg, ai_msg in history:
        messages.append(HumanMessage(content=user_msg))
        if "Generated code snippet" in ai_msg:
             try: saved_file = ai_msg.split("results/")[-1].split(")")[0]; messages.append(AIMessage(content=f"(Generated code saved to {saved_file})"))
             except: messages.append(AIMessage(content=ai_msg))
        else: messages.append(AIMessage(content=ai_msg))
    messages.append(HumanMessage(content=query))
    logger.info(f"Using history (last {len(history)} turns) for code context")
    code_llm_to_use = coding_llm if coding_llm else llm; logger.info(f"Using LLM for code generation: {type(code_llm_to_use).__name__}")
    try:
        response = code_llm_to_use.invoke(messages); raw_code_response = response.content.strip()
        cleaned_code = raw_code_response
        match_py = re.match(r"```python\n?(.*)```", raw_code_response, re.DOTALL | re.IGNORECASE)
        match_r = re.match(r"```r\n?(.*)```", raw_code_response, re.DOTALL | re.IGNORECASE)
        match_generic = re.match(r"```\n?(.*)```", raw_code_response, re.DOTALL)
        if match_py: detected_language = "python"; cleaned_code = match_py.group(1).strip(); logger.info("Detected language: Python")
        elif match_r: detected_language = "r"; cleaned_code = match_r.group(1).strip(); logger.info("Detected language: R")
        elif match_generic:
             cleaned_code = match_generic.group(1).strip()
             if any(kw in cleaned_code for kw in ['library(', '<-', 'ggplot', 'dplyr']): detected_language = "r"; logger.info("Detected language: R (heuristic)")
             else: detected_language = "python"; logger.info(f"Detected language: {detected_language} (heuristic/default)")
        else:
             if any(kw in cleaned_code for kw in ['library(', '<-', 'ggplot', 'dplyr']): detected_language = "r"; logger.info("Detected language: R (heuristic, no backticks)")
             else: detected_language = "python"; logger.info(f"Detected language: {detected_language} (heuristic/default, no backticks)")
        generated_code_text = cleaned_code
        logger.info("LLM code generation complete."); logger.debug("Generated code (start):\n%s", "\n".join(generated_code_text.splitlines()[:5]))
    except Exception as e: error_message = f"Code generation failed: {str(e)}"; logger.error(error_message, exc_info=True); generated_code_text = f"# Error: {e}"; detected_language = "text"
    return {**state, "generated_code": generated_code_text, "generated_code_language": detected_language, "error": error_message}

def route_query(state: AgentState) -> AgentState:
    # (Unchanged)
    logger.info("--- Calling Router ---"); query = state['query']; logger.info(f"Routing query: {query}")
    prompt = ROUTING_PROMPT_TEMPLATE.format(query=query)
    try:
        response = llm.invoke(prompt); classification = response.content.strip().lower().replace("'", "").replace('"', '')
        logger.info(f"LLM Classification: {classification}"); next_node_decision = "chat_agent"
        if classification == "literature_search": logger.info("Routing to Literature Agent."); next_node_decision = "literature_agent"
        elif classification == "code_generation": logger.info("Routing to Coding Agent."); next_node_decision = "coding_agent"
        elif classification == "chat": logger.info("Routing to Chat Agent."); next_node_decision = "chat_agent"
        else: logger.warning(f"Unexpected classification '{classification}'. Defaulting to Chat Agent.")
        return {**state, "next_node": next_node_decision}
    except Exception as e: logger.error(f"Error during LLM routing: {e}. Defaulting to Chat Agent.", exc_info=True); return {**state, "next_node": "chat_agent", "error": f"Routing error: {e}"}

# --- Conditional Edge Logic ---
def decide_next_node(state: AgentState) -> str:
    # (Unchanged)
    next_node = state.get("next_node")
    if next_node not in ["literature_agent", "chat_agent", "coding_agent"]: logger.warning(f"Invalid next_node value '{next_node}' after router. Ending."); return END
    return next_node

def should_download(state: AgentState) -> str:
    # (Unchanged)
    preference = state.get("download_preference", "no")
    if preference == "yes": logger.info("Proceeding to download ArXiv PDFs."); return "download_arxiv_pdfs"
    else: logger.info("Skipping ArXiv PDF download."); return "summarizer"

# --- Graph Definition ---
# (Unchanged)
graph_builder = StateGraph(AgentState); graph_builder.add_node("router", route_query); graph_builder.add_node("literature_agent", call_literature_agent)
graph_builder.add_node("ask_download_preference", ask_download_preference); graph_builder.add_node("download_arxiv_pdfs", download_arxiv_pdfs)
graph_builder.add_node("summarizer", summarize_results); graph_builder.add_node("chat_agent", call_chat_agent); graph_builder.add_node("coding_agent", call_coding_agent)
graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges("router", decide_next_node, {"literature_agent": "literature_agent", "chat_agent": "chat_agent", "coding_agent": "coding_agent", END: END})
graph_builder.add_edge("literature_agent", "ask_download_preference")
graph_builder.add_conditional_edges("ask_download_preference", should_download, {"download_arxiv_pdfs": "download_arxiv_pdfs", "summarizer": "summarizer"})
graph_builder.add_edge("download_arxiv_pdfs", "summarizer"); graph_builder.add_edge("summarizer", END); graph_builder.add_edge("chat_agent", END); graph_builder.add_edge("coding_agent", END)
app = graph_builder.compile()

# --- Function to save results ---
def save_output(run_dir: str, relative_path: str, data: Any):
    # (Unchanged)
    filepath = os.path.join(run_dir, relative_path)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(data, str): f.write(data)
            else: json.dump(data, f, indent=2, default=str)
        logger.info(f"Output saved to: {filepath}")
    except Exception as e: logger.error(f"Failed to save output to {filepath}: {e}", exc_info=True)

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Setup Run Directory ---
    # (Unchanged)
    WORKPLACE_DIR = "workplace"; run_dir = None
    try:
        os.makedirs(WORKPLACE_DIR, exist_ok=True); timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(WORKPLACE_DIR, f"{timestamp}_run")
        logs_dir = os.path.join(run_dir, "logs"); results_dir = os.path.join(run_dir, "results"); temp_data_dir = os.path.join(run_dir, "temp_data")
        os.makedirs(logs_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True); os.makedirs(temp_data_dir, exist_ok=True)
        log_filepath = os.path.join(logs_dir, "run.log"); file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'); file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler); logger.info("--- BioAgent Run Started ---"); logger.info(f"Run directory created: {run_dir}")
        logger.info(f"File logging to: {log_filepath}")
        provider_for_log = get_config_value(config, "llm_provider", "unknown").lower(); logger.info(f"LLM Provider Config: {provider_for_log}")
    except Exception as setup_e: logger.critical(f"Failed to set up run environment: {setup_e}", exc_info=True); sys.exit(1)

    conversation_state = {"history": [], "last_search_results": None, "last_summary": None}
    MAX_HISTORY_TURNS = 5; interaction_count = 0

    while True:
        interaction_count += 1
        # --- Reverted to single-line input ---
        prompt_text = f"\n{COLOR_INPUT}Enter query {interaction_count} (or type 'quit'): {COLOR_RESET}"
        try: initial_query = input(prompt_text)
        except EOFError: initial_query = 'quit'

        # Use strip() in the quit check
        if initial_query.strip().lower() == 'quit':
            logger.info("Quit command received. Exiting.")
            break
        # Handle empty input after stripping
        if not initial_query.strip():
            print(f"{COLOR_WARN}Please enter a query or message.{COLOR_RESET}")
            interaction_count -= 1; continue

        logger.info(f"--- Starting Interaction #{interaction_count} ---")
        input_for_graph = { "query": initial_query.strip(), "history": conversation_state["history"], "run_dir": run_dir, "refined_query": None, "search_results": None, "summary": None, "chat_response": None, "error": None, "next_node": None, "arxiv_results_found": False, "download_preference": None, "code_request": None, "generated_code": None, "generated_code_language": None }
        logger.info(f"Invoking agent graph for query:\n{initial_query[:200]}...")
        final_state = None
        try:
            final_state = app.invoke(input_for_graph); logger.info("Graph execution complete.")
            print(f"\n{COLOR_INFO}--- Agent Output (Interaction #{interaction_count}) ---{COLOR_RESET}"); agent_response = None
            try: logger.debug("Final State: %s", json.dumps(final_state, indent=2, default=str))
            except Exception as dump_e: logger.warning(f"Could not serialize final state for logging: {dump_e}")

            # --- Simplified Output Handling Logic ---
            # Initialize agent_response to ensure it's always defined
            agent_response = None
            output_message = None # Message to print to console

            if final_state.get("error"):
                error_msg = f"An error occurred: {final_state['error']}"
                output_message = f"{COLOR_ERROR}{error_msg}{COLOR_RESET}"
                logger.error(error_msg) # Log full error to file
                agent_response = "Sorry, an error occurred." # Simplified response for history
                save_output(run_dir, os.path.join("results", f"error_{interaction_count}.txt"), error_msg)

            elif final_state.get("generated_code"):
                code = final_state["generated_code"]
                language = final_state.get("generated_code_language", "text")
                extension = {"python": "py", "r": "R"}.get(language, "txt")
                filename = f"generated_code_{interaction_count}.{extension}"
                output_message = f"{COLOR_OUTPUT}Generated Code ({language}):{COLOR_RESET}\n{COLOR_CODE}```{language}\n{code}\n```"
                logger.info(f"Code ({language}) generated and displayed.")
                save_output(run_dir, os.path.join("results", filename), code)
                agent_response = f"Generated {language} code snippet (saved to results/{filename})."
                conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None

            elif final_state.get("search_results"):
                results = final_state["search_results"]
                refined_query = final_state.get('refined_query', 'N/A')
                output_lines = []
                output_lines.append(f"{COLOR_OUTPUT}Found {len(results)} literature results (using refined query: '{refined_query}'):{COLOR_RESET}")
                logger.info(f"Found {len(results)} results for refined query: '{refined_query}'")
                save_output(run_dir, os.path.join("results", f"search_results_{interaction_count}.json"), results)
                conversation_state["last_search_results"] = results
                conversation_state["last_summary"] = final_state.get("summary")

                for i, result in enumerate(results):
                    output_lines.append(f"\n{Style.BRIGHT}--- Result {i+1} ({result.get('source', 'N/A')}) ---{Style.NORMAL}")
                    output_lines.append(f"{COLOR_OUTPUT}  Title: {result.get('title', 'N/A')}{COLOR_RESET}")
                    output_lines.append(f"  ID: {result.get('id', 'N/A')}")
                    output_lines.append(f"  URL: {result.get('url', '#')}")
                    if result.get("local_pdf_path"):
                        output_lines.append(f"  {COLOR_FILE}Downloaded PDF: {result['local_pdf_path']}{COLOR_RESET}")

                if final_state.get("summary"):
                    summary = final_state["summary"]
                    output_lines.append(f"\n{COLOR_SUMMARY}--- Summary of Results ---{COLOR_RESET}")
                    output_lines.append(f"{COLOR_SUMMARY}{summary}{COLOR_RESET}")
                    logger.info("Summary generated and displayed.")
                    save_output(run_dir, os.path.join("results", f"summary_{interaction_count}.txt"), summary)
                    agent_response = summary # Use summary for history
                else:
                    # Construct message if no summary
                    no_summary_msg_local = f"I found {len(results)} results."
                    if final_state.get("download_preference") == "yes": no_summary_msg_local += " PDF downloads attempted."
                    else: no_summary_msg_local += " PDFs were not downloaded."
                    if "Summarization failed" in final_state.get("error", ""): no_summary_msg_local += " Failed to generate summary."
                    output_lines.append(f"\n{COLOR_WARN}BioAgent: {no_summary_msg_local}{COLOR_RESET}")
                    logger.warning("Summary was not generated or failed.")
                    agent_response = no_summary_msg_local
                    save_output(run_dir, os.path.join("results", f"summary_{interaction_count}.txt"), no_summary_msg_local)

                output_message = "\n".join(output_lines)

            elif final_state.get("chat_response"):
                agent_response = final_state["chat_response"]
                output_message = f"\n{COLOR_OUTPUT}BioAgent: {agent_response}{COLOR_RESET}"
                logger.info("Chat response generated and displayed.")
                save_output(run_dir, os.path.join("results", f"chat_response_{interaction_count}.txt"), agent_response)
                conversation_state["last_search_results"] = None
                conversation_state["last_summary"] = None

            # Handle cases where search ran but found nothing
            elif final_state.get("next_node") == "literature_agent" and not final_state.get("error"):
                 refined_query = final_state.get('refined_query', 'N/A')
                 no_results_msg_local = f"No literature results found from PubMed or ArXiv for refined query: '{refined_query}'"
                 output_message = f"{COLOR_WARN}{no_results_msg_local}{COLOR_RESET}"
                 logger.info(f"No literature results found for refined query: '{refined_query}'")
                 agent_response = no_results_msg_local
                 save_output(run_dir, os.path.join("results", f"search_results_{interaction_count}.txt"), no_results_msg_local)
                 conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None

            # Fallback for other unhandled cases with no error
            elif not final_state.get("error"):
                 no_output_msg_local = "No specific output generated."
                 output_message = f"{COLOR_WARN}{no_output_msg_local}{COLOR_RESET}"
                 logger.warning("Graph finished without error but no standard output produced.")
                 agent_response = no_output_msg_local
                 save_output(run_dir, os.path.join("results", f"output_{interaction_count}.txt"), no_output_msg_local)
                 conversation_state["last_search_results"] = None; conversation_state["last_summary"] = None

            # Print the assembled output message if one was created
            if output_message:
                print(output_message)

            # --- Update History ---
            if agent_response is not None:
                 conversation_state["history"].append((initial_query, agent_response));
                 if len(conversation_state["history"]) > MAX_HISTORY_TURNS: conversation_state["history"] = conversation_state["history"][-MAX_HISTORY_TURNS:]
                 save_output(run_dir, os.path.join("logs", "conversation_history.json"), conversation_state["history"])

        except Exception as e:
            # --- Enhanced Traceback Logging ---
            tb_str = traceback.format_exc()
            error_msg = f"An unexpected error occurred during graph invocation: {e}\nTraceback:\n{tb_str}"
            print(f"\n{COLOR_ERROR}{error_msg}{COLOR_RESET}")
            # Log the full traceback as well
            logger.error(f"An unexpected error occurred during graph invocation: {e}", exc_info=True)
            # --- End Enhanced Traceback ---
            try:
                state_at_error = final_state if final_state is not None else input_for_graph
                logger.error("State at time of error: %s", json.dumps(state_at_error, indent=2, default=str))
            except Exception as dump_e:
                 logger.error(f"Could not retrieve or serialize state at time of error: {dump_e}")
            # Optionally add a generic error response to history
            # conversation_state["history"].append((initial_query, "Sorry, a critical error occurred."))
            # save_output(run_dir, os.path.join("logs", "conversation_history.json"), conversation_state["history"])


    logger.info("--- BioAgent Run Ended ---"); print(f"\n{COLOR_INFO}BioAgent session ended.{COLOR_RESET}")

