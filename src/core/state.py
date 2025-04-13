# src/core/state.py
# Central definition for the Agent State TypedDict

from typing import TypedDict, List, Dict, Any, Optional, Tuple

class AgentState(TypedDict):
    """
    Represents the shared state passed between nodes in the LangGraph agent.
    Each field stores information relevant to different parts of the workflow.
    Using TypedDict provides type hinting and improves code clarity.
    """
    # --- Core Input/Output ---
    query: str                      # The user's most recent query.
    history: List[Tuple[str, str]]  # Conversation history (user_msg, ai_msg).
    error: Optional[str]            # Stores accumulated error messages during a run.
    run_dir: Optional[str]          # Path to the current run's directory (used by CLI for saving).

    # --- Routing Information ---
    route_intent: Optional[str]     # Intent classified by the router (e.g., 'literature_search').
    next_node: Optional[str]        # Explicitly set by router for conditional edges.

    # --- Literature Search Path (Also used by Deep Research) ---
    refined_query: Optional[str]    # Query refined by LLM for searching (initial refinement).
    search_results: Optional[List[Dict[str, Any]]] # Stores combined results from the *latest* literature search round.
    arxiv_results_found: bool       # Flag indicating if any ArXiv results were returned in the latest lit search.
    download_preference: Optional[str] # User's choice ('yes'/'no') for downloading ArXiv PDFs (CLI only).
    summary: Optional[str]          # LLM-generated summary of literature abstracts (used mainly by literature_search path).

    # --- Coding Path ---
    code_request: Optional[str]     # Can store a parsed/specific version of the code request if needed later.
    generated_code: Optional[str]   # The code snippet generated by the coding agent.
    generated_code_language: Optional[str] # Detected language ('python', 'r', 'text').

    # --- Chat Path ---
    chat_response: Optional[str]    # Conversational response from the chat agent.

    # --- Deep Research Path ---
    google_results: Optional[List[Dict[str, Any]]] # Stores results from the *latest* Google search round.
    synthesized_report: Optional[str] # Final report generated by synthesizing all evidence.

    # --- Fields for NEW Iterative Deep Research Workflow ---
    hypothesis: Optional[str]       # The current research question or hypothesis being investigated.
    search_plan: List[str]          # The *current* list of sub-topics/questions to search for this iteration.
    evidence_log: List[Dict[str, Any]] # Accumulates *all* relevant findings (snippets, abstracts) with source info across iterations. E.g., {'plan_item': 'original plan item text', 'source_type': ..., 'source_id': ..., 'content': ...}
    research_iterations: int        # Counter for the number of research loops performed.
    evaluation_summary: Optional[str] # LLM's assessment of findings after a search iteration.
    more_research_needed: bool      # Flag set by evaluation node (True/False) to control the loop.
    plan_approved: Optional[str]    # User's decision ('yes'/'mod'/'no') on the research plan.
    plan_modifications: Optional[str] # User's suggested modifications to the plan.

