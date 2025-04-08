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
    # You might choose to exit here, or proceed with a default/warning
    # For now, let's exit if email is not provided, as it's good practice
    sys.exit(1)
else:
    # Set the email for Entrez requests
    Entrez.email = ENTREZ_EMAIL

# --- Agent State Definition ---

# Define the structure for the data that flows through the graph
class AgentState(TypedDict):
    """
    Represents the state of our agent graph.
    """
    query: str # The initial user query
    search_results: Optional[List[Dict[str, Any]]] # Store structured search results
    error: Optional[str] # To store potential errors

# --- Agent Nodes ---

def call_literature_agent(state: AgentState) -> AgentState:
    """
    Literature Agent node that searches PubMed using Bio.Entrez.
    """
    print("--- Calling Literature Agent ---")
    query = state['query']
    print(f"Received query: {query}")
    results = []
    error_message = None
    max_results = 5 # Limit the number of results for now

    try:
        # 1. Search PubMed to get PMIDs (PubMed IDs)
        print(f"Searching PubMed for PMIDs (max_results={max_results})...")
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        search_results = Entrez.read(handle)
        handle.close()
        id_list = search_results["IdList"]

        if not id_list:
            print("No results found on PubMed for the query.")
            return {"search_results": [], "error": None}

        print(f"Found {len(id_list)} PMIDs. Fetching details...")

        # 2. Fetch details for the found PMIDs
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
        records = Medline.parse(handle) # Parse MEDLINE formatted records

        # 3. Extract relevant information from records
        for record in records:
            # Extract desired fields (handle potential missing keys)
            pmid = record.get("PMID", "N/A")
            title = record.get("TI", "No title found")
            abstract = record.get("AB", "No abstract found")
            journal = record.get("JT", "N/A")
            authors = record.get("AU", []) # Authors is a list

            results.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "authors": authors,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" # Construct PubMed URL
            })
        handle.close()
        print(f"Successfully fetched and parsed details for {len(results)} records.")

    except Exception as e:
        print(f"Error during PubMed search: {e}")
        error_message = f"An error occurred during PubMed search: {str(e)}"
        # Return empty results and the error message
        return {"search_results": [], "error": error_message}

    # Update the state with the results
    return {"search_results": results, "error": None}


# --- Graph Definition ---

# Instantiate the LangGraph StateGraph
graph_builder = StateGraph(AgentState)

# Add nodes to the graph
graph_builder.add_node("literature_agent", call_literature_agent)

# Define the entry and exit points of the graph
graph_builder.add_edge(START, "literature_agent")
graph_builder.add_edge("literature_agent", END)

# Compile the graph into a runnable application
app = graph_builder.compile()

# --- Main Execution Block ---

if __name__ == "__main__":
    print("BioAgent Co-Pilot Initializing...")
    print(f"Using Entrez Email: {Entrez.email}") # Confirm email is set

    initial_query = input("Enter your research query: ")
    initial_state = {"query": initial_query, "search_results": None, "error": None} # Initialize state

    print("\nInvoking the agent graph...")
    final_state = app.invoke(initial_state)

    print("\n--- Graph Execution Complete ---")
    print("Final State:")

    if final_state.get("error"):
        print(f"An error occurred: {final_state['error']}")
    elif final_state.get("search_results"):
        results = final_state["search_results"]
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  PMID: {result.get('pmid', 'N/A')} ({result.get('url', '#')})")
            # Optionally print abstract or other details
            # print(f"  Abstract: {result.get('abstract', 'N/A')[:200]}...") # Print first 200 chars
    else:
        print("No results found or state is unexpected.")

    # Optionally print the full final state for debugging
    # print("\nFull Final State Dictionary:")
    # print(json.dumps(final_state, indent=2))
