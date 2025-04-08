import os
import sys
import json
from typing import TypedDict, List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage # For history formatting

# Import BioPython modules for PubMed searching
from Bio import Entrez
from Bio import Medline # To parse fetched records

# Import arxiv library for ArXiv searching
import arxiv

# --- Environment Setup ---

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Retrieve the email for NCBI Entrez (REQUIRED by NCBI)
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")

# Basic check for OpenAI API key
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY not found in .env file.")
    sys.exit(1)

# Basic check and setup for Entrez email
if not ENTREZ_EMAIL:
    print("Warning: ENTREZ_EMAIL not found in .env file.")
    print("NCBI Entrez requires an email address for identification.")
    print("Please add 'ENTREZ_EMAIL=your.email@example.com' to your .env file.")
    sys.exit(1)
else:
    Entrez.email = ENTREZ_EMAIL

# Instantiate the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

# --- Agent State Definition ---

class AgentState(TypedDict):
    """
    Represents the state of our agent graph, including history.
    """
    query: str
    history: List[Tuple[str, str]] # List of (user_query, agent_response) tuples
    refined_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    chat_response: Optional[str]
    error: Optional[str]
    next_node: Optional[str]

# --- Helper Functions for Literature Search ---
# (Unchanged from previous version)
def _search_pubmed(query: str, max_results: int) -> List[Dict[str, Any]]:
    # ... (code unchanged) ...
    print(f"Searching PubMed for '{query}'...")
    results = []
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        search_results_entrez = Entrez.read(handle)
        handle.close()
        id_list = search_results_entrez["IdList"]
        if not id_list:
            print("No results found on PubMed.")
            return []
        print(f"Found {len(id_list)} PMIDs on PubMed. Fetching details...")
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records = Medline.parse(handle)
        for record in records:
            pmid = record.get("PMID", "N/A")
            title = record.get("TI", "No title found")
            abstract = record.get("AB", "No abstract found")
            journal = record.get("JT", "N/A")
            authors = record.get("AU", [])
            results.append({
                "id": pmid, "source": "PubMed", "title": title,
                "abstract": abstract, "journal": journal, "authors": authors,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
        handle.close()
        print(f"Successfully fetched and parsed {len(results)} PubMed records.")
    except Exception as e:
        print(f"Error during PubMed search: {e}")
    return results

def _search_arxiv(query: str, max_results: int) -> List[Dict[str, Any]]:
    # (Unchanged from previous version)
    # ... (code unchanged) ...
    print(f"Searching ArXiv for '{query}'...")
    results = []
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
        arxiv_results = list(client.results(search))
        if not arxiv_results:
             print("No results found on ArXiv.")
             return []
        print(f"Found {len(arxiv_results)} results on ArXiv.")
        for result in arxiv_results:
            arxiv_id = result.entry_id.split('/')[-1]
            results.append({
                "id": arxiv_id, "source": "ArXiv", "title": result.title,
                "abstract": result.summary.replace('\n', ' '),
                "authors": [str(author) for author in result.authors],
                "published": str(result.published),
                "url": result.pdf_url
            })
    except Exception as e:
        print(f"Error during ArXiv search: {e}")
    return results

# --- Agent Nodes ---

def call_literature_agent(state: AgentState) -> AgentState:
    """
    Literature Agent node: Refines query, searches PubMed & ArXiv, combines results.
    (Code is unchanged)
    """
    # (Code is unchanged)
    # ... (code unchanged) ...
    print("--- Calling Literature Agent ---")
    original_query = state['query']
    print(f"Received original query: {original_query}")
    combined_results = []
    error_message = None
    refined_query_for_search = original_query
    max_results_per_source = 3
    try:
        print("Refining query for literature search...")
        refinement_prompt = f"""Given the user's query, extract the core topic or keywords suitable for searching scientific databases like PubMed and ArXiv. Focus on nouns, technical terms, and essential concepts. Remove conversational phrases like "find papers on", "search for", "tell me about". Respond ONLY with the refined search query string.

User Query: "{original_query}"
Refined Search Query:"""
        try:
            refinement_response = llm.invoke(refinement_prompt)
            refined_query_for_search = refinement_response.content.strip()
            print(f"Refined query: {refined_query_for_search}")
        except Exception as refine_e:
            print(f"Warning: LLM query refinement failed: {refine_e}. Using original query.")
            error_message = f"Query refinement failed: {refine_e}. "
        pubmed_results = _search_pubmed(refined_query_for_search, max_results_per_source)
        arxiv_results = _search_arxiv(refined_query_for_search, max_results_per_source)
        combined_results.extend(pubmed_results)
        combined_results.extend(arxiv_results)
        print(f"Total combined results: {len(combined_results)}")
    except Exception as e:
        print(f"An unexpected error occurred in literature agent: {e}")
        search_error = f"Literature search failed: {str(e)}"
        error_message = (error_message + search_error) if error_message else search_error
    # Return history unchanged
    return {
        "refined_query": refined_query_for_search,
        "search_results": combined_results,
        "error": error_message,
        "history": state['history'] # Pass history through
    }


def summarize_results(state: AgentState) -> AgentState:
    """
    Summarizer node that uses the LLM to summarize abstracts from search results.
    (Code is essentially unchanged, just passes history through)
    """
    # (Code is essentially unchanged)
    # ... (code unchanged) ...
    print("--- Calling Summarizer ---")
    search_results = state.get("search_results")
    original_query = state.get("query")
    error_message = state.get("error") # Carry over previous errors
    summary_text = None
    if not search_results:
        print("No search results to summarize.")
        # Return history unchanged
        return {"summary": None, "error": error_message, "history": state['history']}
    max_abstracts_to_summarize = 3
    abstracts_to_summarize = []
    print(f"Preparing abstracts for summarization (max {max_abstracts_to_summarize})...")
    for i, result in enumerate(search_results):
        if i >= max_abstracts_to_summarize: break
        abstract = result.get("abstract")
        if abstract and abstract != "No abstract found":
            abstracts_to_summarize.append(f"Abstract {i+1} (Source: {result.get('source', 'N/A')}, ID: {result.get('id', 'N/A')}):\n{abstract}\n")
    if not abstracts_to_summarize:
        print("No valid abstracts found in results to summarize.")
        # Return history unchanged
        return {"summary": "No abstracts available to summarize.", "error": error_message, "history": state['history']}
    abstracts_text = "\n---\n".join(abstracts_to_summarize)
    summarization_prompt = f"""Given the user's original query and the following abstracts from search results, provide a concise summary of the key findings relevant to the query.

Original User Query: "{original_query}"

Abstracts:
---
{abstracts_text}
---

Concise Summary:"""
    print(f"Sending {len(abstracts_to_summarize)} abstracts to LLM for summarization...")
    try:
        response = llm.invoke(summarization_prompt)
        summary_text = response.content.strip()
        print("LLM Summary generated.")
    except Exception as e:
        print(f"Error during LLM summarization: {e}")
        summary_error = f"Summarization failed: {str(e)}"
        error_message = (error_message + summary_error) if error_message else summary_error
        summary_text = "Sorry, I couldn't generate a summary."
    # Return history unchanged
    return {"summary": summary_text, "error": error_message, "history": state['history']}


def call_chat_agent(state: AgentState) -> AgentState:
    """
    Chat Agent node that uses the LLM to generate a response,
    now considering conversation history.
    """
    print("--- Calling Chat Agent ---")
    query = state['query']
    history = state['history'] # Get history from state
    error_message = None
    chat_response_text = "Sorry, I couldn't generate a response."

    # Format history for the LLM prompt
    formatted_history = []
    for user_msg, ai_msg in history:
        formatted_history.append(HumanMessage(content=user_msg))
        formatted_history.append(AIMessage(content=ai_msg))

    # Add the current user query
    formatted_history.append(HumanMessage(content=query))

    print(f"Received query: {query}")
    print(f"Using history (last {len(history)} turns)") # Log history usage

    try:
        # Invoke LLM with history
        response = llm.invoke(formatted_history) # Pass formatted history + query
        chat_response_text = response.content.strip()
        print(f"LLM chat response: {chat_response_text}")
    except Exception as e:
        print(f"Error during LLM chat generation: {e}")
        error_message = f"Chat generation failed: {str(e)}"

    # Return history unchanged in this node's output dict
    # History will be updated in the main loop after the full graph invocation
    return {"chat_response": chat_response_text, "error": error_message, "history": history}


def route_query(state: AgentState) -> AgentState:
    """
    Router node that classifies the user query using an LLM.
    (Code is unchanged, but needs to pass history through)
    """
    print("--- Calling Router ---")
    query = state['query']
    print(f"Routing query: {query}")
    prompt = f"""Classify the user's query into one of the following categories: 'literature_search' or 'chat'. Respond ONLY with the category name.

    - 'literature_search': The user is asking to find papers, articles, publications, search results, or specific information likely found by searching scientific literature databases (like PubMed or ArXiv). Keywords often include "find papers", "search for articles", "publications on", "literature about".
        Examples:
            "Find papers on CRISPR" -> literature_search
            "Search PubMed for gene editing" -> literature_search
            "What articles discuss LLMs in bioinformatics?" -> literature_search
            "Show me recent studies on vaccine effectiveness" -> literature_search

    - 'chat': The user is asking a general question, asking for a definition or explanation, making a statement, greeting, or having a conversation. These queries typically don't require searching literature databases directly. Keywords often include "what is", "explain", "tell me about", "hello", "hi", "can you". It might also refer to previous turns like "what did you find?".
        Examples:
            "Hello" -> chat
            "Explain what an LLM is" -> chat
            "What is bioinformatics?" -> chat
            "Tell me about DNA sequencing" -> chat
            "What did you find in the last search?" -> chat
            "Can you help me write some code?" -> chat

    User Query: "{query}"
    Classification:"""
    try:
        response = llm.invoke(prompt)
        classification = response.content.strip().lower().replace("'", "").replace('"', '')
        print(f"LLM Classification: {classification}")
        next_node_decision = "chat_agent" # Default to chat
        if classification == "literature_search":
            print("Routing to Literature Agent.")
            next_node_decision = "literature_agent"
        elif classification == "chat":
             print("Routing to Chat Agent.")
             next_node_decision = "chat_agent"
        else:
            print(f"Warning: Unexpected classification '{classification}'. Defaulting to Chat Agent.")
            next_node_decision = "chat_agent"
        # Return history unchanged
        return {"next_node": next_node_decision, "history": state['history']}

    except Exception as e:
        print(f"Error during LLM routing: {e}")
        print("Defaulting to Chat Agent due to error.")
        # Return history unchanged
        return {"next_node": "chat_agent", "error": f"Routing error: {e}", "history": state['history']}


# --- Conditional Edge Logic ---

def decide_next_node(state: AgentState) -> str:
    """
    Determines the next node to visit based on the 'next_node' field in the state.
    (Code is unchanged)
    """
    # (Code is unchanged)
    # ... (code unchanged) ...
    next_node = state.get("next_node")
    if next_node in ["literature_agent", "chat_agent"]:
        return next_node
    else:
        print(f"Warning: Invalid next_node value '{next_node}'. Ending.")
        return END


# --- Graph Definition ---
# (Unchanged from previous version)
# Instantiate the LangGraph StateGraph
graph_builder = StateGraph(AgentState)
# Add nodes
graph_builder.add_node("router", route_query)
graph_builder.add_node("literature_agent", call_literature_agent)
graph_builder.add_node("summarizer", summarize_results)
graph_builder.add_node("chat_agent", call_chat_agent)
# Define entry point
graph_builder.add_edge(START, "router")
# Add conditional edges
graph_builder.add_conditional_edges(
    "router", decide_next_node,
    {"literature_agent": "literature_agent", "chat_agent": "chat_agent", END: END}
)
# Define remaining edges
graph_builder.add_edge("literature_agent", "summarizer")
graph_builder.add_edge("summarizer", END)
graph_builder.add_edge("chat_agent", END)
# Compile the graph
app = graph_builder.compile()

# --- Main Execution Block ---

if __name__ == "__main__":
    print("BioAgent Co-Pilot Initializing...")
    print(f"Using Entrez Email: {Entrez.email}")

    # --- Maintain state across loop iterations ---
    # Initialize conversation state (including history) before the loop
    conversation_state = {
        "history": [],
        # We might store last search results here if needed across turns,
        # but for now, history is the main cross-turn state.
        "last_search_results": None,
        "last_summary": None
    }
    MAX_HISTORY_TURNS = 5 # Limit history length

    while True:
        initial_query = input("\nEnter your research query or message (or type 'quit'): ")
        if initial_query.lower() == 'quit':
            break

        # Prepare input for this graph invocation, including current history
        input_for_graph = {
            "query": initial_query,
            "history": conversation_state["history"],
            # Reset fields that should be populated by this run
            "refined_query": None,
            "search_results": None,
            "summary": None,
            "chat_response": None,
            "error": None,
            "next_node": None
        }

        print("\nInvoking the agent graph...")
        final_state = None # Define final_state before try block
        try:
            # Run the graph
            final_state = app.invoke(input_for_graph)

            print("\n--- Graph Execution Complete ---")
            print("Final State Output:")

            # --- Determine response and update history ---
            agent_response = None
            if final_state.get("error"):
                print(f"An error occurred: {final_state['error']}")
                agent_response = f"Sorry, an error occurred: {final_state['error']}" # Store error as response

            if final_state.get("search_results"):
                results = final_state["search_results"]
                print(f"Found {len(results)} literature results (using refined query: '{final_state.get('refined_query', 'N/A')}'):")
                # Store results/summary in conversation_state for potential future reference
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
                    agent_response = final_state["summary"] # Use summary as the 'response' for history
                else:
                    # Provide a generic response if summary failed or wasn't generated
                    agent_response = f"I found {len(results)} results but couldn't generate a summary."
                    print(f"\nBioAgent: {agent_response}")


            elif final_state.get("chat_response"):
                agent_response = final_state["chat_response"]
                print(f"\nBioAgent: {agent_response}")
                # Clear last search results if this was a chat response
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

            # --- Update History ---
            if agent_response is not None:
                 conversation_state["history"].append((initial_query, agent_response))
                 # Limit history length
                 if len(conversation_state["history"]) > MAX_HISTORY_TURNS:
                     conversation_state["history"] = conversation_state["history"][-MAX_HISTORY_TURNS:]
            # --- ---

        except Exception as e:
            print(f"\nAn unexpected error occurred during graph invocation: {e}")
            try:
                # Use final_state if defined, otherwise input_for_graph
                state_at_error = final_state if final_state is not None else input_for_graph
                print("State at time of error:", json.dumps(state_at_error, indent=2, default=str)) # Use default=str for non-serializable
            except Exception as dump_e:
                 print(f"Could not retrieve or serialize state at time of error: {dump_e}")

    print("\nBioAgent session ended.")