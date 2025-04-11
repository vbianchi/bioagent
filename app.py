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

# --- Import colorama and Initialize ---
from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

# --- Color Scheme ---
COLOR_INFO = Fore.CYAN; COLOR_INPUT = Fore.YELLOW; COLOR_OUTPUT = Fore.GREEN
COLOR_SUMMARY = Fore.MAGENTA; COLOR_ERROR = Fore.RED; COLOR_WARN = Fore.YELLOW
COLOR_DEBUG = Fore.BLUE; COLOR_RESET = Style.RESET_ALL; COLOR_FILE = Fore.LIGHTBLUE_EX
COLOR_QUESTION = Fore.BLUE + Style.BRIGHT; COLOR_CODE = Fore.LIGHTYELLOW_EX

# --- Custom Colored Logging Formatter ---
# (Keep class definition as before)
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
for handler in logger.handlers[:]: logger.removeHandler(handler) # Remove existing handlers to avoid duplication on rerun
console_handler = logging.StreamHandler(sys.stdout); console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter()); logger.addHandler(console_handler)
# File handler added later in session state setup

# --- Configuration and Initialization ---
config = load_config(); logger.info("Configuration loaded.")
llm_provider = get_config_value(config, "llm_provider", "N/A")
coding_provider = get_config_value(config, "coding_agent_settings.llm_provider", "default")
if coding_provider == 'default': coding_provider = llm_provider

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
    print(f"Session Run Directory: {st.session_state.run_dir}")
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
    try: app = create_agent_app(); logger.info("Agent graph compiled successfully."); return app
    except Exception as e: st.error(f"Failed to compile agent graph: {e}"); logger.critical(f"Failed to compile agent graph: {e}", exc_info=True); return None

agent_app = get_compiled_app()

if agent_app is None: st.error("Agent application failed to initialize."); st.stop()

# --- CSS for Scrollable Chat ---
chat_container_css = """
<style>
#chat-history-container { max-height: 65vh; overflow-y: auto; margin-bottom: 15px; }
div[data-testid="stVerticalBlock"]:has(textarea[aria-label="Log Output:"]) { max-height: 85vh; overflow-y: auto; }
</style>
"""
st.markdown(chat_container_css, unsafe_allow_html=True)

# --- Streamlit Page Title & Layout ---
st.title("🧬 BioAgent Co-Pilot 🤖")
st.caption(f"Powered by LangGraph | Main LLM: {llm_provider} | Coding LLM: {coding_provider}")

col1, col2 = st.columns([2, 1])

# --- Main Chat Area ---
with col1:
    st.header("Chat Interaction")
    if "messages" not in st.session_state: st.session_state.messages = []
    if "conversation_state_history" not in st.session_state: st.session_state.conversation_state_history = []

    # Wrap the chat history display in a div with the ID targeted by CSS
    st.markdown("<div id='chat-history-container'>", unsafe_allow_html=True)
    # Display Chat History
    for message in st.session_state.messages:
        # Remove explicit key parameter
        with st.chat_message(message["role"]):
            if message["type"] == "text": st.markdown(message["content"], unsafe_allow_html=True)
            elif message["type"] == "code": st.code(message["content"], language=message.get("language", "plaintext"))
            elif message["type"] == "search_results":
                st.markdown(f"Found {len(message['content'])} results (Refined Query: '{message.get('refined_query', 'N/A')}')")
                for j, res in enumerate(message["content"]):
                     # Keep key for expander if needed, make it simpler
                     res_key = f"res_{j}_{res.get('id', j)}"
                     with st.expander(f"Result {j+1}: {res.get('title', 'N/A')} ({res.get('source', 'N/A')})", key=res_key):
                          st.markdown(f"**ID:** {res.get('id', 'N/A')}")
                          st.markdown(f"**URL:** [{res.get('url', '#')}]({res.get('url', '#')})")
                          st.markdown(f"**Abstract:** {res.get('abstract', 'N/A')}")
                          if res.get("local_pdf_path"): st.info(f"Downloaded PDF: {res['local_pdf_path']}")
                if message.get("summary"): st.markdown("---"); st.markdown(f"**Summary:**\n{message['summary']}")
            elif message["type"] == "error": st.error(message["content"])
    st.markdown("</div>", unsafe_allow_html=True) # Close the wrapping div

    # --- Handle User Input ---
    if prompt := st.chat_input("Enter your query or message..."):
        st.session_state.messages.append({"role": "user", "type": "text", "content": prompt})
        st.rerun()

# --- Log View Panel ---
with col2:
    st.subheader("Agent Log")
    log_container = st.container()
    with log_container:
        log_content = ""; log_file = st.session_state.get("log_filepath")
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f: lines = f.readlines(); log_content = "".join(lines[-100:])
            except Exception as e: log_content = f"Error reading log file: {e}"
        elif log_file: log_content = f"Log file not found yet: {log_file}"
        else: log_content = "Log file path not set in session state."
        st.text_area("Log Output:", value=log_content, height=600, disabled=True, key="log_view_area")


# --- Agent Processing Logic ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    current_history = st.session_state.conversation_state_history
    input_for_graph = {
        "query": user_prompt, "history": current_history, "run_dir": None,
        "refined_query": None, "search_results": None, "summary": None,
        "chat_response": None, "error": None, "next_node": None,
        "arxiv_results_found": False, "download_preference": "no",
        "code_request": None, "generated_code": None, "generated_code_language": None
    }
    final_state = None; agent_response_for_history = None
    response_message = {"role": "assistant", "type": "text", "content": ""}
    try:
        logger.info(f"Invoking agent graph for query: {user_prompt[:100]}...")
        with st.spinner("Agent is thinking..."): final_state = agent_app.invoke(input_for_graph)
        logger.info("Graph execution complete.")
        # (Determine output type/content - unchanged)
        if final_state.get("error"):
            response_message["type"] = "error"; response_message["content"] = f"An error occurred: {final_state['error']}"
            agent_response_for_history = "Sorry, an error occurred."
        elif final_state.get("generated_code"):
            response_message["type"] = "code"; response_message["content"] = final_state["generated_code"]
            response_message["language"] = final_state.get("generated_code_language", "plaintext")
            agent_response_for_history = f"Generated {response_message['language']} code snippet."
        elif final_state.get("search_results") is not None:
            response_message["type"] = "search_results"; response_message["content"] = final_state["search_results"]
            response_message["refined_query"] = final_state.get("refined_query"); response_message["summary"] = final_state.get("summary")
            agent_response_for_history = final_state.get("summary") or f"Found {len(final_state['search_results'])} literature results."
        elif final_state.get("chat_response"):
            response_message["type"] = "text"; response_message["content"] = final_state["chat_response"]
            agent_response_for_history = final_state["chat_response"]
        else:
            response_message["content"] = "No specific output generated by the agent."; agent_response_for_history = "No specific output generated."
    except Exception as e:
        logger.critical(f"An unexpected error occurred during graph invocation: {e}", exc_info=True)
        response_message["type"] = "error"; response_message["content"] = f"An unexpected error occurred: {e}"
        agent_response_for_history = "Sorry, a critical error occurred."

    # Update history
    if agent_response_for_history is not None:
        new_history = current_history + [(user_prompt, agent_response_for_history)]
        MAX_HISTORY_TURNS = 5
        if len(new_history) > MAX_HISTORY_TURNS: st.session_state.conversation_state_history = new_history[-(MAX_HISTORY_TURNS*2):]
        else: st.session_state.conversation_state_history = new_history
    # Add final response and rerun
    st.session_state.messages.append(response_message)
    st.rerun()

