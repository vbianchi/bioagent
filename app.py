import streamlit as st
import sys
import os
import datetime
import logging
import traceback # Import traceback for detailed error logging
from typing import List, Tuple, Dict, Any

# --- Set Page Config FIRST ---
st.set_page_config(page_title="BioAgent Co-Pilot", layout="wide")

# --- Add src to path for imports ---
app_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(app_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Import Agent Components ---
try:
    from main import create_agent_app, AgentState
    from core.config_loader import load_config, get_config_value
except ImportError as e:
    st.error(f"Error importing agent components from src: {e}")
    st.error(f"Ensure app.py is in the project root directory ({app_dir}) and the src directory exists at {src_path}.")
    st.stop()

# --- Configuration and Initialization ---
config = load_config()
llm_provider = get_config_value(config, "llm_provider", "N/A")
coding_provider = get_config_value(config, "coding_agent_settings.llm_provider", "default")
if coding_provider == 'default': coding_provider = llm_provider

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Session State Management for Run Directory and Logging ---
if "run_dir" not in st.session_state:
    WORKPLACE_DIR = "workplace_streamlit"
    os.makedirs(WORKPLACE_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.run_dir = os.path.join(WORKPLACE_DIR, f"{timestamp}_streamlit_run")
    st.session_state.logs_dir = os.path.join(st.session_state.run_dir, "logs")
    st.session_state.results_dir = os.path.join(st.session_state.run_dir, "results")
    os.makedirs(st.session_state.logs_dir, exist_ok=True)
    os.makedirs(st.session_state.results_dir, exist_ok=True)
    st.session_state.log_filepath = os.path.join(st.session_state.logs_dir, "run.log")
    print(f"Session Run Directory: {st.session_state.run_dir}") # Debug print
    print(f"Session Log File: {st.session_state.log_filepath}")

    # Configure File Logging for this session
    handler_exists = any(isinstance(h, logging.FileHandler) and h.baseFilename == st.session_state.log_filepath for h in logger.handlers)
    if not handler_exists:
        file_handler = logging.FileHandler(st.session_state.log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"--- Streamlit Session Started ({st.session_state.run_dir}) ---")
        logger.info(f"File logging for this session directed to: {st.session_state.log_filepath}")
    else:
         logger.info("File handler already exists for this session.")

# --- Create Agent App ---
@st.cache_resource
def get_compiled_app():
    logger.info("Compiling agent graph...")
    try:
        app = create_agent_app()
        logger.info("Agent graph compiled successfully.")
        return app
    except Exception as e:
        st.error(f"Failed to compile agent graph: {e}")
        logger.critical(f"Failed to compile agent graph: {e}", exc_info=True)
        return None

agent_app = get_compiled_app()

if agent_app is None:
     st.error("Agent application failed to initialize. Cannot proceed.")
     st.stop()

# --- CSS for Scrollable Chat ---
# Inject CSS to make the chat history container scrollable
st.markdown("""
    <style>
    #chat-history-container {
        /* Adjust height as needed - vh units are relative to viewport height */
        max-height: 70vh;
        overflow-y: auto;
        /* Add some spacing below the chat history */
        margin-bottom: 15px;
        /* Optional: Add a border for visual separation */
        /* border: 1px solid #e6e6e6; */
        /* border-radius: 5px; */
        /* padding: 10px; */
    }
    /* Ensure columns take up height */
    /* These selectors might be brittle */
     div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        height: 90vh; /* Try forcing column height */
        display: flex;
        flex-direction: column;
     }
     /* Make chat input container not shrink */
     div[data-testid="stChatInput"] {
         flex-shrink: 0;
     }
     /* Make chat history container grow */
     #chat-history-container {
         flex-grow: 1;
     }

    </style>
""", unsafe_allow_html=True)

# --- Streamlit Page Title & Layout ---
st.title("🧬 BioAgent Co-Pilot 🤖")
st.caption(f"Powered by LangGraph | Main LLM: {llm_provider} | Coding LLM: {coding_provider}")

col1, col2 = st.columns([2, 1]) # Main area takes 2/3, log view takes 1/3

# --- Main Chat Area ---
with col1:
    st.header("Chat Interaction")
    # Initialize session states if they don't exist
    if "messages" not in st.session_state: st.session_state.messages = []
    if "conversation_state_history" not in st.session_state: st.session_state.conversation_state_history = []

    # Wrap the chat history display in a div with the ID targeted by CSS
    st.markdown("<div id='chat-history-container'>", unsafe_allow_html=True)
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display different content types
            if message["type"] == "text": st.markdown(message["content"])
            elif message["type"] == "code": st.code(message["content"], language=message.get("language", "plaintext"))
            elif message["type"] == "search_results":
                st.markdown(f"Found {len(message['content'])} results (Refined Query: '{message.get('refined_query', 'N/A')}')")
                for i, res in enumerate(message["content"]):
                     with st.expander(f"Result {i+1}: {res.get('title', 'N/A')} ({res.get('source', 'N/A')})"):
                          st.markdown(f"**ID:** {res.get('id', 'N/A')}")
                          st.markdown(f"**URL:** [{res.get('url', '#')}]({res.get('url', '#')})")
                          st.markdown(f"**Abstract:** {res.get('abstract', 'N/A')}")
                if message.get("summary"): st.markdown("---"); st.markdown(f"**Summary:**\n{message['summary']}")
            elif message["type"] == "error": st.error(message["content"])
    st.markdown("</div>", unsafe_allow_html=True) # Close the wrapping div

    # --- Handle User Input (Placed *after* the history container) ---
    if prompt := st.chat_input("Enter your query or message..."):
        # Add user message to display history
        st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
        # Rerun immediately to show user message before processing
        st.rerun()

# --- Log View Panel ---
with col2:
    st.subheader("Agent Log")
    # Create a container for the log display
    log_container = st.container()
    with log_container:
        log_content = ""
        log_file = st.session_state.get("log_filepath")
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    log_content = "".join(lines[-100:]) # Show last 100 lines
            except Exception as e:
                log_content = f"Error reading log file: {e}"
        elif log_file: log_content = f"Log file not found yet: {log_file}"
        else: log_content = "Log file path not set in session state."

        # Use height parameter for text_area scrolling
        st.text_area("Log Output", value=log_content, height=600, disabled=True, key="log_view_area")


# --- Agent Processing Logic (Triggered by rerun after input) ---
# Check if the last message is from the user to process it
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]

    # Prepare input for the agent graph
    current_history = st.session_state.conversation_state_history
    input_for_graph = {
        "query": user_prompt, "history": current_history, "run_dir": None, # UI mode
        "refined_query": None, "search_results": None, "summary": None,
        "chat_response": None, "error": None, "next_node": None,
        "arxiv_results_found": False, "download_preference": "no",
        "code_request": None, "generated_code": None, "generated_code_language": None
    }

    # Invoke Agent Graph & Prepare Response Message
    final_state = None
    agent_response_for_history = None
    response_message = {"role": "assistant", "type": "text", "content": ""} # Default structure

    try:
        logger.info(f"Invoking agent graph for query: {user_prompt[:100]}...")
        with st.spinner("Agent is thinking..."): # Add spinner
            final_state = agent_app.invoke(input_for_graph)
        logger.info("Graph execution complete.")

        # Determine the primary output type and content
        if final_state.get("error"):
            response_message["type"] = "error"
            response_message["content"] = f"An error occurred: {final_state['error']}"
            agent_response_for_history = "Sorry, an error occurred."
        elif final_state.get("generated_code"):
            response_message["type"] = "code"
            response_message["content"] = final_state["generated_code"]
            response_message["language"] = final_state.get("generated_code_language", "plaintext")
            agent_response_for_history = f"Generated {response_message['language']} code snippet."
        elif final_state.get("search_results") is not None:
            response_message["type"] = "search_results"
            response_message["content"] = final_state["search_results"]
            response_message["refined_query"] = final_state.get("refined_query")
            response_message["summary"] = final_state.get("summary")
            agent_response_for_history = final_state.get("summary") or f"Found {len(final_state['search_results'])} literature results."
        elif final_state.get("chat_response"):
            response_message["type"] = "text"
            response_message["content"] = final_state["chat_response"]
            agent_response_for_history = final_state["chat_response"]
        else:
            response_message["content"] = "No specific output generated by the agent."
            agent_response_for_history = "No specific output generated."

    except Exception as e:
        logger.critical(f"An unexpected error occurred during graph invocation: {e}", exc_info=True)
        response_message["type"] = "error"
        response_message["content"] = f"An unexpected error occurred: {e}"
        agent_response_for_history = "Sorry, a critical error occurred."

    # --- Update conversation state history ---
    if agent_response_for_history is not None:
        new_history = current_history + [(user_prompt, agent_response_for_history)]
        MAX_HISTORY_TURNS = 5
        if len(new_history) > MAX_HISTORY_TURNS:
            st.session_state.conversation_state_history = new_history[-(MAX_HISTORY_TURNS*2):]
        else:
             st.session_state.conversation_state_history = new_history

    # Add the final response message to the display list
    st.session_state.messages.append(response_message)
    # Rerun the script one last time to update the display with the assistant message
    st.rerun()

