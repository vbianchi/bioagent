import os
import sys
import json
from typing import TypedDict, List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# LLM Imports - Now conditional based on provider
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START

# Import BioPython and ArXiv
from Bio import Entrez, Medline
import arxiv

# Import config loader
from src.core.config_loader import load_config, get_config_value

# --- Configuration Loading ---
config = load_config() # Load config from config/settings.yaml

# --- Environment Setup ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")

if not ENTREZ_EMAIL: sys.exit("Error: ENTREZ_EMAIL not found in .env file. Please add it.")
Entrez.email = ENTREZ_EMAIL

# --- LLM Instantiation based on Config ---
llm_provider = get_config_value(config, "llm_provider", "openai").lower()
llm_temp = get_config_value(config, "llm_settings.temperature", 0)
llm = None

print(f"Attempting to initialize LLM provider: {llm_provider}")

if llm_provider == "openai":
    from langchain_openai import ChatOpenAI
    if not OPENAI_API_KEY: sys.exit("Error: LLM provider is 'openai' but OPENAI_API_KEY not found in .env file.")
    llm_model = get_config_value(config, "llm_settings.openai_model_name", "gpt-3.5-turbo")
    try:
        llm = ChatOpenAI(model=llm_model, temperature=llm_temp, openai_api_key=OPENAI_API_KEY)
        print(f"LLM Initialized: OpenAI (model={llm_model}, temperature={llm_temp})")
    except Exception as e:
        sys.exit(f"Error initializing OpenAI LLM: {e}")

elif llm_provider == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not GOOGLE_API_KEY: sys.exit("Error: LLM provider is 'gemini' but GOOGLE_API_KEY not found in .env file.")
    llm_model = get_config_value(config, "llm_settings.gemini_model_name", "gemini-1.5-flash-latest") # Updated default
    try:
        # Removed deprecated convert_system_message_to_human=True
        llm = ChatGoogleGenerativeAI(model=llm_model, temperature=llm_temp, google_api_key=GOOGLE_API_KEY)
        print(f"LLM Initialized: Google Gemini (model={llm_model}, temperature={llm_temp})")
    except Exception as e:
        sys.exit(f"Error initializing Google Gemini LLM: {e}")

elif llm_provider == "ollama":
    # Use the new dedicated package
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        sys.exit("Error: langchain-ollama package not found. Please install it: pip install langchain-ollama")

    llm_model = get_config_value(config, "llm_settings.ollama_model_name", "llama3")
    ollama_base_url = get_config_value(config, "llm_settings.ollama_base_url")
    try:
        init_params = {"model": llm_model, "temperature": llm_temp}
        if ollama_base_url:
            init_params["base_url"] = ollama_base_url
        llm = ChatOllama(**init_params)
        llm.invoke("test connection") # Test connection
        print(f"LLM Initialized: Ollama (model={llm_model}, temperature={llm_temp}, base_url={ollama_base_url or 'default'})")
    except Exception as e:
        print(f"Error initializing or connecting to Ollama LLM: {e}")
        print("Please ensure the Ollama service is running and the specified model ('{llm_model}') is available.")
        sys.exit(1)

else:
    sys.exit(f"Error: Unknown llm_provider '{llm_provider}'. Use 'openai', 'gemini', or 'ollama'.")

# --- Search Settings from Config ---
# Corrected key used here:
MAX_RESULTS_PER_SOURCE = get_config_value(config, "search_settings.max_results_per_source", 3)
MAX_ABSTRACTS_TO_SUMMARIZE = get_config_value(config, "search_settings.max_abstracts_to_summarize", 3)
# Assign the single setting to both variables used in helpers
MAX_RESULTS_PUBMED = MAX_RESULTS_PER_SOURCE
MAX_RESULTS_ARXIV = MAX_RESULTS_PER_SOURCE
print(f"Search settings: max_results_per_source={MAX_RESULTS_PER_SOURCE}, max_abstracts_to_summarize={MAX_ABSTRACTS_TO_SUMMARIZE}")


# --- Prompt Templates from Config ---
ROUTING_PROMPT_TEMPLATE = get_config_value(config, "prompts.routing_prompt", "Error: Routing prompt not found.")
REFINEMENT_PROMPT_TEMPLATE = get_config_value(config, "prompts.refinement_prompt", "Error: Refinement prompt not found.")
SUMMARIZATION_PROMPT_TEMPLATE = get_config_value(config, "prompts.summarization_prompt", "Error: Summarization prompt not found.")

# --- Agent State Definition ---
class AgentState(TypedDict):
    query: str
    history: List[Tuple[str, str]]
    refined_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    chat_response: Optional[str]
    error: Optional[str]
    next_node: Optional[str]

# --- Helper Functions for Literature Search ---
# (Unchanged - use MAX_RESULTS_PUBMED/ARXIV which are now set correctly)
def _search_pubmed(query: str, max_results: int) -> List[Dict[str, Any]]:
    print(f"Searching PubMed for '{query}' (max_results={max_results})...")
    results = []
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        search_results_entrez = Entrez.read(handle)
        handle.close()
        id_list = search_results_entrez["IdList"]
        if not id_list: return []
        print(f"Found {len(id_list)} PMIDs on PubMed. Fetching details...")
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records = Medline.parse(handle)
        for record in records:
            results.append({
                "id": record.get("PMID", "N/A"), "source": "PubMed",
                "title": record.get("TI", "No title found"),
                "abstract": record.get("AB", "No abstract found"),
                "journal": record.get("JT", "N/A"), "authors": record.get("AU", []),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}/"
            })
        handle.close()
        print(f"Successfully fetched/parsed {len(results)} PubMed records.")
    except Exception as e: print(f"Error during PubMed search: {e}")
    return results

def _search_arxiv(query: str, max_results: int) -> List[Dict[str, Any]]:
    print(f"Searching ArXiv for '{query}' (max_results={max_results})...")
    results = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        arxiv_results = list(client.results(search))
        if not arxiv_results: return []
        print(f"Found {len(arxiv_results)} results on ArXiv.")
        for result in arxiv_results:
            results.append({
                "id": result.entry_id.split('/')[-1], "source": "ArXiv",
                "title": result.title, "abstract": result.summary.replace('\n', ' '),
                "authors": [str(author) for author in result.authors],
                "published": str(result.published), "url": result.pdf_url
            })
    except Exception as e: print(f"Error during ArXiv search: {e}")
    return results

# --- Agent Nodes ---
# (Unchanged - use global llm and prompt templates)
def call_literature_agent(state: AgentState) -> AgentState:
    print("--- Calling Literature Agent ---")
    original_query = state['query']
    error_message = None
    refined_query_for_search = original_query
    try:
        print("Refining query for literature search...")
        refinement_prompt = REFINEMENT_PROMPT_TEMPLATE.format(query=original_query)
        try:
            refinement_response = llm.invoke(refinement_prompt)
            refined_query_for_search = refinement_response.content.strip()
            print(f"Refined query: {refined_query_for_search}")
        except Exception as refine_e:
            print(f"Warning: LLM query refinement failed: {refine_e}. Using original query.")
            error_message = f"Query refinement failed: {refine_e}. "
        pubmed_results = _search_pubmed(refined_query_for_search, MAX_RESULTS_PUBMED)
        arxiv_results = _search_arxiv(refined_query_for_search, MAX_RESULTS_ARXIV)
        combined_results = pubmed_results + arxiv_results
        print(f"Total combined results: {len(combined_results)}")
    except Exception as e:
        search_error = f"Literature search failed: {str(e)}"
        error_message = (error_message + search_error) if error_message else search_error
        combined_results = []
    return {
        "refined_query": refined_query_for_search, "search_results": combined_results,
        "error": error_message, "history": state['history']
    }

def summarize_results(state: AgentState) -> AgentState:
    print("--- Calling Summarizer ---")
    search_results = state.get("search_results")
    original_query = state.get("query")
    error_message = state.get("error")
    summary_text = None
    if not search_results:
        return {"summary": None, "error": error_message, "history": state['history']}
    abstracts_to_summarize = []
    print(f"Preparing abstracts for summarization (max {MAX_ABSTRACTS_TO_SUMMARIZE})...")
    for i, result in enumerate(search_results):
        if i >= MAX_ABSTRACTS_TO_SUMMARIZE: break
        abstract = result.get("abstract")
        if abstract and abstract != "No abstract found":
            abstracts_to_summarize.append(f"Abstract {i+1} (Source: {result.get('source', 'N/A')}, ID: {result.get('id', 'N/A')}):\n{abstract}\n")
    if not abstracts_to_summarize:
        return {"summary": "No abstracts available to summarize.", "error": error_message, "history": state['history']}
    abstracts_text = "\n---\n".join(abstracts_to_summarize)
    summarization_prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(query=original_query, abstracts_text=abstracts_text)
    print(f"Sending {len(abstracts_to_summarize)} abstracts to LLM for summarization...")
    try:
        response = llm.invoke(summarization_prompt)
        summary_text = response.content.strip()
        print("LLM Summary generated.")
    except Exception as e:
        summary_error = f"Summarization failed: {str(e)}"
        error_message = (error_message + summary_error) if error_message else summary_error
        summary_text = "Sorry, I couldn't generate a summary."
    return {"summary": summary_text, "error": error_message, "history": state['history']}

def call_chat_agent(state: AgentState) -> AgentState:
    print("--- Calling Chat Agent ---")
    query = state['query']
    history = state['history']
    error_message = None
    chat_response_text = "Sorry, I couldn't generate a response."
    formatted_history = []
    for user_msg, ai_msg in history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))
    formatted_history.append(HumanMessage(content=query))
    print(f"Using history (last {len(history)} turns)")
    try:
        response = llm.invoke(formatted_history)
        chat_response_text = response.content.strip()
        print(f"LLM chat response: {chat_response_text}")
    except Exception as e:
        error_message = f"Chat generation failed: {str(e)}"
    return {"chat_response": chat_response_text, "error": error_message, "history": history}

def route_query(state: AgentState) -> AgentState:
    print("--- Calling Router ---")
    query = state['query']
    prompt = ROUTING_PROMPT_TEMPLATE.format(query=query)
    try:
        response = llm.invoke(prompt)
        classification = response.content.strip().lower().replace("'", "").replace('"', '')
        print(f"LLM Classification: {classification}")
        next_node_decision = "chat_agent"
        if classification == "literature_search": next_node_decision = "literature_agent"
        elif classification == "chat": next_node_decision = "chat_agent"
        else: print(f"Warning: Unexpected classification '{classification}'. Defaulting to Chat Agent.")
        return {"next_node": next_node_decision, "history": state['history']}
    except Exception as e:
        print(f"Error during LLM routing: {e}. Defaulting to Chat Agent.")
        return {"next_node": "chat_agent", "error": f"Routing error: {e}", "history": state['history']}

# --- Conditional Edge Logic ---
def decide_next_node(state: AgentState) -> str:
    next_node = state.get("next_node")
    return next_node if next_node in ["literature_agent", "chat_agent"] else END

# --- Graph Definition ---
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

# --- Main Execution Block ---
if __name__ == "__main__":
    print("BioAgent Co-Pilot Initializing...")
    print(f"Using Entrez Email: {Entrez.email}")
    conversation_state = {"history": [], "last_search_results": None, "last_summary": None}
    MAX_HISTORY_TURNS = 5
    while True:
        initial_query = input("\nEnter your research query or message (or type 'quit'): ")
        if initial_query.lower() == 'quit': break
        input_for_graph = {
            "query": initial_query, "history": conversation_state["history"],
            "refined_query": None, "search_results": None, "summary": None,
            "chat_response": None, "error": None, "next_node": None
        }
        print("\nInvoking the agent graph...")
        final_state = None
        try:
            final_state = app.invoke(input_for_graph)
            print("\n--- Graph Execution Complete ---")
            print("Final State Output:")
            agent_response = None
            if final_state.get("error"):
                print(f"An error occurred: {final_state['error']}")
                agent_response = f"Sorry, an error occurred: {final_state['error']}"
            if final_state.get("search_results"):
                results = final_state["search_results"]
                print(f"Found {len(results)} literature results (using refined query: '{final_state.get('refined_query', 'N/A')}'):")
                conversation_state["last_search_results"] = results
                conversation_state["last_summary"] = final_state.get("summary")
                for i, result in enumerate(results):
                    print(f"\n--- Result {i+1} ({result.get('source', 'N/A')}) ---")
                    print(f"  Title: {result.get('title', 'N/A')}")
                    print(f"  ID: {result.get('id', 'N/A')}")
                    print(f"  URL: {result.get('url', '#')}")
                if final_state.get("summary"):
                    print("\n--- Summary of Results ---")
                    print(final_state["summary"])
                    agent_response = final_state["summary"]
                else:
                    agent_response = f"I found {len(results)} results but couldn't generate a summary."
                    print(f"\nBioAgent: {agent_response}")
            elif final_state.get("chat_response"):
                agent_response = final_state["chat_response"]
                print(f"\nBioAgent: {agent_response}")
                conversation_state["last_search_results"] = None
                conversation_state["last_summary"] = None
            elif final_state.get("next_node") == "literature_agent" and not final_state.get("error"):
                 no_results_msg = f"No literature results found from PubMed or ArXiv for refined query: '{final_state.get('refined_query', 'N/A')}'"
                 print(no_results_msg)
                 agent_response = no_results_msg
                 conversation_state["last_search_results"] = None
                 conversation_state["last_summary"] = None
            elif not final_state.get("error"):
                 no_output_msg = "No specific output generated."
                 print(no_output_msg)
                 agent_response = no_output_msg
                 conversation_state["last_search_results"] = None
                 conversation_state["last_summary"] = None

            if agent_response is not None:
                 conversation_state["history"].append((initial_query, agent_response))
                 if len(conversation_state["history"]) > MAX_HISTORY_TURNS:
                     conversation_state["history"] = conversation_state["history"][-MAX_HISTORY_TURNS:]
        except Exception as e:
            print(f"\nAn unexpected error occurred during graph invocation: {e}")
            try:
                state_at_error = final_state if final_state is not None else input_for_graph
                print("State at time of error:", json.dumps(state_at_error, indent=2, default=str))
            except Exception as dump_e: print(f"Could not retrieve or serialize state at time of error: {dump_e}")
    print("\nBioAgent session ended.")