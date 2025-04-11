from typing import TypedDict, List, Dict, Any, Optional, Tuple

# Define AgentState centrally
class AgentState(TypedDict):
    """
    Represents the state of our agent graph.
    Includes fields for various agent outputs and control flow.
    """
    # Core input/output
    query: str
    history: List[Tuple[str, str]]
    error: Optional[str]
    next_node: Optional[str] # For routing decisions
    run_dir: Optional[str] # Directory for saving files (used in CLI)

    # Literature Search Path
    refined_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]] # Combined PubMed/ArXiv results
    arxiv_results_found: bool
    download_preference: Optional[str]
    summary: Optional[str] # Summary of literature results

    # Coding Path
    code_request: Optional[str] # Potentially parsed request
    generated_code: Optional[str]
    generated_code_language: Optional[str]

    # Chat Path
    chat_response: Optional[str]

    # Deep Research Path
    google_results: Optional[List[Dict[str, Any]]] # Store google search snippets/links
    synthesized_report: Optional[str] # Final report from deep research

