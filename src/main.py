import os
import sys
import json
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

# Import BioPython modules for PubMed searching
from Bio import Entrez
from Bio import Medline # To parse fetched records

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

# Instantiate the LLM for routing and potentially other tasks
# Use temperature 0 for more deterministic routing/refinement
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

# --- Agent State Definition ---

# Define the structure for the data that flows through the graph
class AgentState(TypedDict):
    """
    Represents the state of our agent graph.
    """
    query: str # The initial user query
    refined_query: Optional[str] # Store the LLM-refined query for PubMed
    search_results: Optional[List[Dict[str, Any]]] # Store structured search results
    chat_response: Optional[str] # Store responses from the chat agent
    error: Optional[str] # To store potential errors
    next_node: Optional[str] # Field to guide routing

# --- Agent Nodes ---

def call_literature_agent(state: AgentState) -> AgentState:
    """
    Literature Agent node that searches PubMed using Bio.Entrez.
    Includes an LLM step to refine the user query into effective search terms.
    """
    print("--- Calling Literature Agent ---")
    original_query = state['query']
    print(f"Received original query: {original_query}")
    results = []
    error_message = None
    refined_query_for_search = original_query # Default to original if refinement fails
    max_results = 5

    try:
        # --- LLM Query Refinement Step ---
        print("Refining query for PubMed search...")
        refinement_prompt = f"""Given the user's query, extract the core topic or keywords suitable for a PubMed search. Focus on nouns, technical terms, and essential concepts. Remove conversational phrases like "find papers on", "search for", "tell me about". Respond ONLY with the refined search query string.

User Query: "{original_query}"
Refined PubMed Query:"""

        try:
            refinement_response = llm.invoke(refinement_prompt)
            refined_query_for_search = refinement_response.content.strip()
            print(f"Refined query: {refined_query_for_search}")
        except Exception as refine_e:
            print(f"Warning: LLM query refinement failed: {refine_e}. Using original query.")
            error_message = f"Query refinement failed: {refine_e}. " # Add to potential errors

        # --- PubMed Search using Refined Query ---
        print(f"Searching PubMed for PMIDs using term '{refined_query_for_search}' (max_results={max_results})...")
        handle = Entrez.esearch(db="pubmed", term=refined_query_for_search, retmax=max_results)
        search_results_entrez = Entrez.read(handle)
        handle.close()
        id_list = search_results_entrez["IdList"]

        if not id_list:
            print("No results found on PubMed for the refined query.")
            # Store the refined query in the state even if no results
            return {"refined_query": refined_query_for_search, "search_results": [], "error": error_message}

        print(f"Found {len(id_list)} PMIDs. Fetching details...")
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records = Medline.parse(handle)

        for record in records:
            pmid = record.get("PMID", "N/A")
            title = record.get("TI", "No title found")
            abstract = record.get("AB", "No abstract found")
            journal = record.get("JT", "N/A")
            authors = record.get("AU", [])
            results.append({
                "pmid": pmid, "title": title, "abstract": abstract,
                "journal": journal, "authors": authors,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
        handle.close()
        print(f"Successfully fetched and parsed details for {len(results)} records.")

    except Exception as e:
        print(f"Error during PubMed search: {e}")
        # Append search error to any refinement error
        search_error = f"An error occurred during PubMed search: {str(e)}"
        error_message = (error_message + search_error) if error_message else search_error
        # Store the refined query and error
        return {"refined_query": refined_query_for_search, "search_results": [], "error": error_message}

    # Update the state with the refined query and results
    return {"refined_query": refined_query_for_search, "search_results": results, "error": error_message}


def call_chat_agent(state: AgentState) -> AgentState:
    """
    Placeholder for a Chat Agent node.
    (Code is unchanged)
    """
    print("--- Calling Chat Agent ---")
    query = state['query']
    print(f"Received query: {query}")
    # TODO: Implement actual chat logic using the LLM if needed, or just respond
    dummy_response = f"Placeholder chat response to: '{query}'"
    print(f"Returning dummy response: {dummy_response}")
    return {"chat_response": dummy_response}

def route_query(state: AgentState) -> AgentState:
    """
    Router node that classifies the user query using an LLM.
    Determines whether the query is for literature search or general chat.
    Uses an improved prompt for better classification.
    (Code is unchanged)
    """
    print("--- Calling Router ---")
    query = state['query']
    print(f"Routing query: {query}")

    # --- IMPROVED PROMPT ---
    prompt = f"""Classify the user's query into one of the following categories: 'literature_search' or 'chat'. Respond ONLY with the category name.

    - 'literature_search': The user is asking to find papers, articles, publications, search results, or specific information likely found by searching scientific literature databases (like PubMed). Keywords often include "find papers", "search for articles", "publications on", "literature about".
        Examples:
            "Find papers on CRISPR" -> literature_search
            "Search PubMed for gene editing" -> literature_search
            "What articles discuss LLMs in bioinformatics?" -> literature_search
            "Show me recent studies on vaccine effectiveness" -> literature_search

    - 'chat': The user is asking a general question, asking for a definition or explanation, making a statement, greeting, or having a conversation. These queries typically don't require searching literature databases directly. Keywords often include "what is", "explain", "tell me about", "hello", "hi", "can you".
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

        if classification == "literature_search":
            print("Routing to Literature Agent.")
            return {"next_node": "literature_agent"}
        elif classification == "chat":
             print("Routing to Chat Agent.")
             return {"next_node": "chat_agent"}
        else:
            print(f"Warning: Unexpected classification '{classification}'. Defaulting to Chat Agent.")
            return {"next_node": "chat_agent"}

    except Exception as e:
        print(f"Error during LLM routing: {e}")
        print("Defaulting to Chat Agent due to error.")
        return {"next_node": "chat_agent", "error": f"Routing error: {e}"}


# --- Conditional Edge Logic ---

def decide_next_node(state: AgentState) -> str:
    """
    Determines the next node to visit based on the 'next_node' field in the state.
    (Code is unchanged)
    """
    next_node = state.get("next_node")
    if next_node in ["literature_agent", "chat_agent"]:
        return next_node
    else:
        print(f"Warning: Invalid next_node value '{next_node}'. Ending.")
        return END


# --- Graph Definition ---

# Instantiate the LangGraph StateGraph
graph_builder = StateGraph(AgentState)

# Add nodes to the graph
graph_builder.add_node("router", route_query)
graph_builder.add_node("literature_agent", call_literature_agent)
graph_builder.add_node("chat_agent", call_chat_agent)

# Define the entry point
graph_builder.add_edge(START, "router")

# Add conditional edges from the router
graph_builder.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "literature_agent": "literature_agent",
        "chat_agent": "chat_agent",
        END: END
    }
)

# Define edges leading to the end
graph_builder.add_edge("literature_agent", END)
graph_builder.add_edge("chat_agent", END)

# Compile the graph into a runnable application
app = graph_builder.compile()

# --- Main Execution Block ---

if __name__ == "__main__":
    print("BioAgent Co-Pilot Initializing...")
    print(f"Using Entrez Email: {Entrez.email}")

    while True:
        initial_query = input("\nEnter your research query or message (or type 'quit'): ")
        if initial_query.lower() == 'quit':
            break

        # Initialize state, including new refined_query field
        initial_state = {
            "query": initial_query,
            "refined_query": None,
            "search_results": None,
            "chat_response": None,
            "error": None,
            "next_node": None
        }

        print("\nInvoking the agent graph...")
        try:
            final_state = app.invoke(initial_state)

            print("\n--- Graph Execution Complete ---")
            print("Final State Output:")

            if final_state.get("error"):
                # Print accumulated errors (could be from refinement and/or search)
                print(f"An error occurred: {final_state['error']}")

            if final_state.get("search_results"):
                results = final_state["search_results"]
                print(f"Found {len(results)} literature results (using refined query: '{final_state.get('refined_query', 'N/A')}'):")
                for i, result in enumerate(results):
                    print(f"\n--- Result {i+1} ---")
                    print(f"  Title: {result.get('title', 'N/A')}")
                    print(f"  PMID: {result.get('pmid', 'N/A')} ({result.get('url', '#')})")
            elif final_state.get("chat_response"):
                print(f"Chat response: {final_state['chat_response']}")
            # Handle case where literature search ran but found 0 results (no error)
            elif final_state.get("next_node") == "literature_agent" and not final_state.get("error"):
                 print(f"No literature results found for refined query: '{final_state.get('refined_query', 'N/A')}'")
            elif not final_state.get("error"): # Catch other cases with no output
                 print("No specific output generated (check agent logic).")


        except Exception as e:
            print(f"\nAn unexpected error occurred during graph invocation: {e}")
            # Print state if available for debugging unexpected errors
            try:
                print("State at time of error:", json.dumps(final_state, indent=2))
            except: # Handle case where final_state might not be defined/serializable
                 print("Could not retrieve state at time of error.")


    print("\nBioAgent session ended.")
