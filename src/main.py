import os
import sys
import json
import datetime
import logging
from io import StringIO
from typing import TypedDict, List, Dict, Any, Optional, Tuple

# Import colorama
from colorama import init as colorama_init, Fore, Style

from dotenv import load_dotenv

# LLM Imports - Now conditional based on provider
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START

# Import BioPython and ArXiv
from Bio import Entrez, Medline
import arxiv

# Import config loader
from src.core.config_loader import load_config, get_config_value

# --- Initialize Colorama ---
colorama_init(autoreset=True)

# --- Color Scheme ---
COLOR_INFO = Fore.CYAN
COLOR_INPUT = Fore.YELLOW
COLOR_OUTPUT = Fore.GREEN
COLOR_SUMMARY = Fore.MAGENTA
COLOR_ERROR = Fore.RED
COLOR_WARN = Fore.YELLOW
COLOR_DEBUG = Fore.BLUE
COLOR_RESET = Style.RESET_ALL

# --- Custom Colored Logging Formatter ---
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels for console output."""
    LOG_COLORS = {
        logging.DEBUG: COLOR_DEBUG,
        logging.INFO: COLOR_INFO,
        logging.WARNING: COLOR_WARN,
        logging.ERROR: COLOR_ERROR,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }
    def format(self, record):
        log_color = self.LOG_COLORS.get(record.levelno, COLOR_RESET)
        log_fmt = f"{log_color}{record.levelname}: {record.getMessage()}{COLOR_RESET}"
        return log_fmt

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.propagate = False
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)
# File handler added in main block

# --- Configuration Loading ---
config = load_config()
logger.info("Configuration loaded.")

# --- Environment Setup ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")

if not ENTREZ_EMAIL: logger.critical("ENTREZ_EMAIL not found in .env file. Please add it."); sys.exit(1)
Entrez.email = ENTREZ_EMAIL
logger.info(f"Entrez Email set to: {ENTREZ_EMAIL}")

# --- LLM Instantiation based on Config ---
llm_provider = get_config_value(config, "llm_provider", "openai").lower()
llm_temp = get_config_value(config, "llm_settings.temperature", 0)
llm = None

logger.info(f"Attempting to initialize LLM provider: {llm_provider}")

if llm_provider == "openai":
    from langchain_openai import ChatOpenAI
    if not OPENAI_API_KEY: logger.critical("LLM provider is 'openai' but OPENAI_API_KEY not found."); sys.exit(1)
    llm_model = get_config_value(config, "llm_settings.openai_model_name", "gpt-3.5-turbo")
    try:
        llm = ChatOpenAI(model=llm_model, temperature=llm_temp, openai_api_key=OPENAI_API_KEY)
        logger.info(f"LLM Initialized: OpenAI (model={llm_model}, temperature={llm_temp})")
    except Exception as e: logger.critical(f"Error initializing OpenAI LLM: {e}"); sys.exit(1)
elif llm_provider == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not GOOGLE_API_KEY: logger.critical("LLM provider is 'gemini' but GOOGLE_API_KEY not found."); sys.exit(1)
    llm_model = get_config_value(config, "llm_settings.gemini_model_name", "gemini-1.5-flash-latest")
    try:
        llm = ChatGoogleGenerativeAI(model=llm_model, temperature=llm_temp, google_api_key=GOOGLE_API_KEY)
        logger.info(f"LLM Initialized: Google Gemini (model={llm_model}, temperature={llm_temp})")
    except Exception as e: logger.critical(f"Error initializing Google Gemini LLM: {e}"); sys.exit(1)
elif llm_provider == "ollama":
    try: from langchain_ollama import ChatOllama
    except ImportError: logger.critical("langchain-ollama package not found. Run: pip install langchain-ollama"); sys.exit(1)
    llm_model = get_config_value(config, "llm_settings.ollama_model_name", "gemma3")
    ollama_base_url = get_config_value(config, "llm_settings.ollama_base_url")
    try:
        init_params = {"model": llm_model, "temperature": llm_temp}
        if ollama_base_url: init_params["base_url"] = ollama_base_url
        llm = ChatOllama(**init_params)
        llm.invoke("test connection")
        logger.info(f"LLM Initialized: Ollama (model={llm_model}, temperature={llm_temp}, base_url={ollama_base_url or 'default'})")
    except Exception as e:
        logger.critical(f"Error initializing or connecting to Ollama LLM: {e}")
        logger.critical(f"Ensure Ollama is running and model '{llm_model}' is available.")
        sys.exit(1)
else:
    logger.critical(f"Unknown llm_provider '{llm_provider}'. Use 'openai', 'gemini', or 'ollama'.")
    sys.exit(1)

# --- Search Settings from Config ---
MAX_RESULTS_PER_SOURCE = get_config_value(config, "search_settings.max_results_per_source", 3)
MAX_ABSTRACTS_TO_SUMMARIZE = get_config_value(config, "search_settings.max_abstracts_to_summarize", 3)
MAX_RESULTS_PUBMED = MAX_RESULTS_PER_SOURCE
MAX_RESULTS_ARXIV = MAX_RESULTS_PER_SOURCE
logger.info(f"Search settings: max_results_per_source={MAX_RESULTS_PER_SOURCE}, max_abstracts_to_summarize={MAX_ABSTRACTS_TO_SUMMARIZE}")

# --- Prompt Templates from Config ---
ROUTING_PROMPT_TEMPLATE = get_config_value(config, "prompts.routing_prompt", "Error: Routing prompt not found.")
REFINEMENT_PROMPT_TEMPLATE = get_config_value(config, "prompts.refinement_prompt", "Error: Refinement prompt not found.")
SUMMARIZATION_PROMPT_TEMPLATE = get_config_value(config, "prompts.summarization_prompt", "Error: Summarization prompt not found.")

# --- Agent State Definition ---
class AgentState(TypedDict):
    query: str; history: List[Tuple[str, str]]; refined_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]; summary: Optional[str]
    chat_response: Optional[str]; error: Optional[str]; next_node: Optional[str]

# --- Helper Functions for Literature Search ---
def _search_pubmed(query: str, max_results: int) -> List[Dict[str, Any]]:
    # (Unchanged from previous version)
    logger.info(f"Searching PubMed for '{query}' (max_results={max_results})...")
    results = []
    try:
        handle_search = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        search_results_entrez = Entrez.read(handle_search); handle_search.close()
        id_list = search_results_entrez["IdList"]
        if not id_list: logger.info("No results found on PubMed."); return []
        logger.info(f"Found {len(id_list)} PMIDs on PubMed. Fetching details...")
        handle_fetch = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records_text = handle_fetch.read(); handle_fetch.close()
        records = Medline.parse(StringIO(records_text))
        count = 0
        for record in records:
            results.append({
                "id": record.get("PMID", "N/A"), "source": "PubMed",
                "title": record.get("TI", "No title found"),
                "abstract": record.get("AB", "No abstract found"),
                "journal": record.get("JT", "N/A"), "authors": record.get("AU", []),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}/"
            })
            count += 1
        logger.info(f"Successfully fetched/parsed {count} PubMed records.")
    except Exception as e: logger.error(f"Error during PubMed search: {e}", exc_info=True)
    return results

def _search_arxiv(query: str, max_results: int) -> List[Dict[str, Any]]:
    # (Unchanged from previous version)
    logger.info(f"Searching ArXiv for '{query}' (max_results={max_results})...")
    results = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        arxiv_results = list(client.results(search))
        if not arxiv_results: logger.info("No results found on ArXiv."); return []
        logger.info(f"Found {len(arxiv_results)} results on ArXiv.")
        for result in arxiv_results:
            results.append({
                "id": result.entry_id.split('/')[-1], "source": "ArXiv",
                "title": result.title, "abstract": result.summary.replace('\n', ' '),
                "authors": [str(author) for author in result.authors],
                "published": str(result.published), "url": result.pdf_url
            })
    except Exception as e: logger.error(f"Error during ArXiv search: {e}", exc_info=True)
    return results

# --- Agent Nodes ---
def call_literature_agent(state: AgentState) -> AgentState:
    # (Unchanged from previous version)
    logger.info("--- Calling Literature Agent ---")
    original_query = state['query']; logger.info(f"Received original query: {original_query}")
    error_message = None; refined_query_for_search = original_query
    try:
        logger.info("Refining query for literature search...")
        refinement_prompt = REFINEMENT_PROMPT_TEMPLATE.format(query=original_query)
        try:
            refinement_response = llm.invoke(refinement_prompt)
            refined_query_for_search = refinement_response.content.strip()
            logger.info(f"Refined query: {refined_query_for_search}")
        except Exception as refine_e:
            logger.warning(f"LLM query refinement failed: {refine_e}. Using original query.")
            error_message = f"Query refinement failed: {refine_e}. "
        pubmed_results = _search_pubmed(refined_query_for_search, MAX_RESULTS_PUBMED)
        arxiv_results = _search_arxiv(refined_query_for_search, MAX_RESULTS_ARXIV)
        combined_results = pubmed_results + arxiv_results
        logger.info(f"Total combined results: {len(combined_results)}")
    except Exception as e:
        search_error = f"Literature search failed: {str(e)}"; logger.error(search_error, exc_info=True)
        error_message = (error_message + search_error) if error_message else search_error
        combined_results = []
    return {
        "refined_query": refined_query_for_search, "search_results": combined_results,
        "error": error_message, "history": state['history']
    }

def summarize_results(state: AgentState) -> AgentState:
    # (Unchanged from previous version)
    logger.info("--- Calling Summarizer ---")
    search_results = state.get("search_results"); original_query = state.get("query")
    error_message = state.get("error"); summary_text = None
    if not search_results:
        logger.info("No search results to summarize.")
        return {"summary": None, "error": error_message, "history": state['history']}
    abstracts_to_summarize = []
    logger.info(f"Preparing abstracts for summarization (max {MAX_ABSTRACTS_TO_SUMMARIZE})...")
    for i, result in enumerate(search_results):
        if i >= MAX_ABSTRACTS_TO_SUMMARIZE: break
        abstract = result.get("abstract")
        if abstract and abstract != "No abstract found":
            abstracts_to_summarize.append(f"Abstract {i+1} (Source: {result.get('source', 'N/A')}, ID: {result.get('id', 'N/A')}):\n{abstract}\n")
    if not abstracts_to_summarize:
        logger.info("No valid abstracts found in results to summarize.")
        return {"summary": "No abstracts available to summarize.", "error": error_message, "history": state['history']}
    abstracts_text = "\n---\n".join(abstracts_to_summarize)
    summarization_prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(query=original_query, abstracts_text=abstracts_text)
    logger.info(f"Sending {len(abstracts_to_summarize)} abstracts to LLM for summarization...")
    try:
        response = llm.invoke(summarization_prompt)
        summary_text = response.content.strip()
        logger.info("LLM Summary generated.")
    except Exception as e:
        summary_error = f"Summarization failed: {str(e)}"; logger.error(summary_error, exc_info=True)
        error_message = (error_message + summary_error) if error_message else summary_error
        summary_text = "Sorry, I couldn't generate a summary."
    return {"summary": summary_text, "error": error_message, "history": state['history']}

def call_chat_agent(state: AgentState) -> AgentState:
    # (Unchanged from previous version)
    logger.info("--- Calling Chat Agent ---")
    query = state['query']; history = state['history']
    error_message = None; chat_response_text = "Sorry, I couldn't generate a response."
    formatted_history = []
    for user_msg, ai_msg in history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    formatted_history.append(HumanMessage(content=query))
    logger.info(f"Received query: {query}")
    logger.info(f"Using history (last {len(history)} turns)")
    try:
        response = llm.invoke(formatted_history)
        chat_response_text = response.content.strip()
        logger.info(f"LLM chat response generated.")
    except Exception as e:
        error_message = f"Chat generation failed: {str(e)}"; logger.error(error_message, exc_info=True)
    return {"chat_response": chat_response_text, "error": error_message, "history": history}

def route_query(state: AgentState) -> AgentState:
    # (Unchanged from previous version)
    logger.info("--- Calling Router ---")
    query = state['query']; logger.info(f"Routing query: {query}")
    prompt = ROUTING_PROMPT_TEMPLATE.format(query=query)
    try:
        response = llm.invoke(prompt)
        classification = response.content.strip().lower().replace("'", "").replace('"', '')
        logger.info(f"LLM Classification: {classification}")
        next_node_decision = "chat_agent"
        if classification == "literature_search":
            logger.info("Routing to Literature Agent."); next_node_decision = "literature_agent"
        elif classification == "chat":
             logger.info("Routing to Chat Agent."); next_node_decision = "chat_agent"
        else: logger.warning(f"Unexpected classification '{classification}'. Defaulting to Chat Agent.")
        return {"next_node": next_node_decision, "history": state['history']}
    except Exception as e:
        logger.error(f"Error during LLM routing: {e}. Defaulting to Chat Agent.", exc_info=True)
        return {"next_node": "chat_agent", "error": f"Routing error: {e}", "history": state['history']}

# --- Conditional Edge Logic ---
def decide_next_node(state: AgentState) -> str:
    # (Unchanged from previous version)
    next_node = state.get("next_node")
    if next_node not in ["literature_agent", "chat_agent"]:
        logger.warning(f"Invalid next_node value '{next_node}'. Ending.")
        return END
    return next_node

# --- Graph Definition ---
# (Unchanged from previous version)
graph_builder = StateGraph(AgentState)
graph_builder.add_node("router", route_query)
graph_builder.add_node("literature_agent", call_literature_agent)
graph_builder.add_node("summarizer", summarize_results)
graph_builder.add_node("chat_agent", call_chat_agent)
graph_builder.add_edge(START, "router")
graph_builder.add_conditional_edges(
    "router", decide_next_node,
    {"literature_agent": "literature_agent", "chat_agent": "chat_agent", END: END}
)
graph_builder.add_edge("literature_agent", "summarizer")
graph_builder.add_edge("summarizer", END)
graph_builder.add_edge("chat_agent", END)
app = graph_builder.compile()

# --- Function to save results ---
def save_output(run_dir: str, relative_path: str, data: Any):
    # (Unchanged from previous version)
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
    # (Unchanged from previous version)
    WORKPLACE_DIR = "workplace"; run_dir = None
    try:
        os.makedirs(WORKPLACE_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(WORKPLACE_DIR, f"{timestamp}_run")
        logs_dir = os.path.join(run_dir, "logs"); results_dir = os.path.join(run_dir, "results")
        temp_data_dir = os.path.join(run_dir, "temp_data")
        os.makedirs(logs_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True); os.makedirs(temp_data_dir, exist_ok=True)
        log_filepath = os.path.join(logs_dir, "run.log")
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'); file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info("--- BioAgent Run Started ---"); logger.info(f"Run directory created: {run_dir}")
        logger.info(f"File logging to: {log_filepath}"); logger.info(f"LLM Provider: {llm_provider}")
    except Exception as setup_e: logger.critical(f"Failed to set up run environment: {setup_e}", exc_info=True); sys.exit(1)

    conversation_state = {"history": [], "last_search_results": None, "last_summary": None}
    MAX_HISTORY_TURNS = 5
    interaction_count = 0 # <<< Initialize interaction counter

    while True:
        try:
            initial_query = input(f"\n{COLOR_INPUT}Enter query {interaction_count + 1} (or type 'quit'): {COLOR_RESET}")
        except EOFError: initial_query = 'quit'

        if not initial_query.strip():
            print(f"{COLOR_WARN}Please enter a query or message.{COLOR_RESET}")
            continue
        if initial_query.lower() == 'quit':
            logger.info("Quit command received. Exiting.")
            break

        interaction_count += 1 # <<< Increment counter for each valid interaction
        logger.info(f"--- Starting Interaction #{interaction_count} ---")

        input_for_graph = {
            "query": initial_query, "history": conversation_state["history"],
            "refined_query": None, "search_results": None, "summary": None,
            "chat_response": None, "error": None, "next_node": None
        }

        logger.info(f"Invoking agent graph for query: {initial_query}")
        final_state = None
        try:
            final_state = app.invoke(input_for_graph)
            logger.info("Graph execution complete.")

            print(f"\n{COLOR_INFO}--- Agent Output (Interaction #{interaction_count}) ---{COLOR_RESET}")
            agent_response = None

            try: logger.debug("Final State: %s", json.dumps(final_state, indent=2, default=str))
            except Exception as dump_e: logger.warning(f"Could not serialize final state for logging: {dump_e}")

            if final_state.get("error"):
                error_msg = f"An error occurred: {final_state['error']}"
                print(f"{COLOR_ERROR}{error_msg}{COLOR_RESET}")
                logger.error(error_msg)
                agent_response = f"Sorry, an error occurred."
                # Save error state
                save_output(run_dir, os.path.join("results", f"error_{interaction_count}.txt"), error_msg)


            if final_state.get("search_results"):
                results = final_state["search_results"]
                refined_query = final_state.get('refined_query', 'N/A')
                print(f"{COLOR_OUTPUT}Found {len(results)} literature results (using refined query: '{refined_query}'):{COLOR_RESET}")
                logger.info(f"Found {len(results)} results for refined query: '{refined_query}'")
                # Use interaction count in filename
                save_output(run_dir, os.path.join("results", f"search_results_{interaction_count}.json"), results)

                conversation_state["last_search_results"] = results
                conversation_state["last_summary"] = final_state.get("summary")

                for i, result in enumerate(results):
                    print(f"\n{Style.BRIGHT}--- Result {i+1} ({result.get('source', 'N/A')}) ---{Style.NORMAL}")
                    print(f"{COLOR_OUTPUT}  Title: {result.get('title', 'N/A')}{COLOR_RESET}")
                    print(f"  ID: {result.get('id', 'N/A')}")
                    print(f"  URL: {result.get('url', '#')}")

                if final_state.get("summary"):
                    summary = final_state["summary"]
                    print(f"\n{COLOR_SUMMARY}--- Summary of Results ---{COLOR_RESET}")
                    print(f"{COLOR_SUMMARY}{summary}{COLOR_RESET}")
                    logger.info("Summary generated and displayed.")
                    # Use interaction count in filename
                    save_output(run_dir, os.path.join("results", f"summary_{interaction_count}.txt"), summary)
                    agent_response = summary
                else:
                    no_summary_msg = f"I found {len(results)} results but couldn't generate a summary."
                    print(f"\n{COLOR_WARN}BioAgent: {no_summary_msg}{COLOR_RESET}")
                    logger.warning("Summary could not be generated.")
                    agent_response = no_summary_msg
                    # Use interaction count in filename
                    save_output(run_dir, os.path.join("results", f"summary_{interaction_count}.txt"), no_summary_msg)

            elif final_state.get("chat_response"):
                agent_response = final_state["chat_response"]
                print(f"\n{COLOR_OUTPUT}BioAgent: {agent_response}{COLOR_RESET}")
                logger.info("Chat response generated and displayed.")
                # Use interaction count in filename
                save_output(run_dir, os.path.join("results", f"chat_response_{interaction_count}.txt"), agent_response)
                conversation_state["last_search_results"] = None
                conversation_state["last_summary"] = None

            elif final_state.get("next_node") == "literature_agent" and not final_state.get("error"):
                 refined_query = final_state.get('refined_query', 'N/A')
                 no_results_msg = f"No literature results found from PubMed or ArXiv for refined query: '{refined_query}'"
                 print(f"{COLOR_WARN}{no_results_msg}{COLOR_RESET}")
                 logger.info(f"No literature results found for refined query: '{refined_query}'")
                 agent_response = no_results_msg
                 # Use interaction count in filename
                 save_output(run_dir, os.path.join("results", f"search_results_{interaction_count}.txt"), no_results_msg)
                 conversation_state["last_search_results"] = None
                 conversation_state["last_summary"] = None

            elif not final_state.get("error"):
                 no_output_msg = "No specific output generated."
                 print(f"{COLOR_WARN}{no_output_msg}{COLOR_RESET}")
                 logger.warning("Graph finished without error but no standard output produced.")
                 agent_response = no_output_msg
                 # Use interaction count in filename
                 save_output(run_dir, os.path.join("results", f"output_{interaction_count}.txt"), no_output_msg)
                 conversation_state["last_search_results"] = None
                 conversation_state["last_summary"] = None

            # --- Update History ---
            if agent_response is not None:
                 conversation_state["history"].append((initial_query, agent_response))
                 if len(conversation_state["history"]) > MAX_HISTORY_TURNS:
                     conversation_state["history"] = conversation_state["history"][-MAX_HISTORY_TURNS:]
                 # Save cumulative history (still overwrites previous history state for the run)
                 save_output(run_dir, os.path.join("logs", "conversation_history.json"), conversation_state["history"])


        except Exception as e:
            error_msg = f"An unexpected error occurred during graph invocation: {e}"
            print(f"\n{COLOR_ERROR}{error_msg}{COLOR_RESET}")
            logger.exception(error_msg)
            try:
                state_at_error = final_state if final_state is not None else input_for_graph
                logger.error("State at time of error: %s", json.dumps(state_at_error, indent=2, default=str))
            except Exception as dump_e:
                 logger.error(f"Could not retrieve or serialize state at time of error: {dump_e}")

    logger.info("--- BioAgent Run Ended ---")
    print(f"\n{COLOR_INFO}BioAgent session ended.{COLOR_RESET}")