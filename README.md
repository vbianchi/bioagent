# BioAgent Co-Pilot (Working Title)

A project to develop an agentic AI co-pilot for epidemiology and bioinformatics research tasks. It routes queries among chat, literature search, and code generation intents. It searches PubMed/ArXiv, optionally downloads ArXiv PDFs, summarizes results, generates Python/R code snippets (using conversation history), and maintains conversation history. Supports multiple LLM providers with per-task configuration. Each run creates a timestamped folder in `workplace/` with logs and results. Console output is colored. The project structure is modular, separating agents and tools.

## Description

(Add a more detailed description of the project goals and functionalities here.)

## Current Status (CLI Version - v1.31 / UI v1.37)

The command-line interface (CLI) and basic Streamlit UI currently support the following features in the underlying agent logic:

* **Multi-Agent Routing:** Classifies user queries into 'chat', 'literature_search', or 'code_generation' intents and routes to the appropriate agent.
* **Literature Search:**
    * Refines natural language queries into search terms using an LLM.
    * Searches both PubMed and ArXiv.
    * Prompts the user to optionally download PDFs for found ArXiv papers (CLI only).
    * Summarizes the abstracts of found papers using an LLM.
* **Coding Assistance:**
    * Generates Python or R code snippets based on user requests.
    * Uses conversation history for context (e.g., modifying previous code).
    * Detects the language of generated code and saves it with the correct file extension (.py or .R) (CLI only for saving).
    * **Note:** Code is generated but *not* executed by the agent.
* **Chat:**
    * Provides conversational responses using an LLM.
    * Maintains basic conversation history within a run for context.
* **LLM Flexibility:**
    * Supports OpenAI, Google Gemini, and local Ollama models as LLM providers.
    * Allows configuring a separate LLM provider/model specifically for coding tasks (`config/settings.yaml`).
* **Workflow & Output:**
    * Uses LangGraph for defining the agent workflow.
    * Loads settings and prompts from `config/settings.yaml`.
    * Creates a timestamped directory in `workplace_streamlit/` (for UI) or `workplace/` (for CLI) for each run/session.
    * Logs agent operations to `run.log` within the run directory.
    * Saves interaction results (CLI only currently) and cumulative history (both) to files.
    * Provides colored console output for CLI.
    * Handles basic errors and empty input.
* **Streamlit UI:** Basic chat interface, multi-column layout with live log view panel, displays formatted agent outputs (chat, search results, summaries, code). Handles multi-line input.

## Next Steps

1.  Refine Streamlit UI: Improve display of results, add copy buttons for code, enhance layout (scrolling, fixed elements).
2.  Re-integrate features like conditional PDF download using Streamlit widgets (buttons, etc.).
3.  Implement file saving for results generated via the UI.
4.  Resume development of advanced agent features (Deep Research, Analysis Agent execution, Tool Execution) within the UI framework.

## Setup

This project uses `uv` for environment and package management and supports multiple LLM backends.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vbianchi/bioagent.git](https://github.com/vbianchi/bioagent.git)
    cd bioagent
    ```
2.  **Install `uv` (if you don't have it):**
    `uv` is a fast Python package installer and resolver. Follow the official installation instructions: [https://github.com/astral-sh/uv#installation](https://github.com/astral-sh/uv#installation)
    (Typically involves running a curl or pip command).

3.  **Create and activate the virtual environment:**
    ```bash
    # Create the environment using uv
    uv venv --python 3.12 # Or your desired Python version
    # Activate the environment (Linux/Mac/WSL)
    source .venv/bin/activate
    # (For Windows CMD: .venv\Scripts\activate.bat)
    # (For Windows PowerShell: .venv\Scripts\Activate.ps1)
    ```
4.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
5.  **Configuration File (`config/settings.yaml`):**
    * Ensure the `config/` directory exists (it should after cloning).
    * Review `config/settings.yaml`. You can modify prompts or settings here later.
    * **Choose LLM Providers:** Edit `llm_provider` (for general tasks) and `coding_agent_settings.llm_provider` (for coding tasks). Options: `"openai"`, `"gemini"`, `"ollama"`, or `"default"` (to use the main provider).
    * **Set Model Names:** Adjust the corresponding model name settings (e.g., `openai_model_name`, `gemini_model_name`, `ollama_model_name`, and those under `coding_agent_settings`). Ensure models are available for the chosen provider.

6.  **API Keys & Environment (`.env`):**
    * Create a `.env` file in the project root directory (you can copy `.env.example` if it exists, or create a new file).
    * Add the following lines, filling in your actual keys/email:
      ```dotenv
      # Needed if using llm_provider = "openai" OR coding_agent_settings.llm_provider = "openai"
      OPENAI_API_KEY="sk-..."

      # Needed if using llm_provider = "gemini" OR coding_agent_settings.llm_provider = "gemini"
      GOOGLE_API_KEY="AIza..."

      # ALWAYS Required for PubMed searches
      ENTREZ_EMAIL="your.actual.email@example.com"
      ```
    * Get API keys from:
        * OpenAI: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
        * Google AI Studio (for Gemini): [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
    * **Important:** Ensure `.env` is listed in your `.gitignore` file (it should be by default) to avoid committing your secret keys.

7.  **Ollama Setup (if using `ollama` provider):**
    * **Install Ollama:** Follow the instructions on the Ollama website: [https://ollama.com/](https://ollama.com/)
    * **Run Ollama:** Ensure the Ollama application or background service is running.
    * **Pull Models:** Download the specific models you intend to use (as specified in `config/settings.yaml`) via your terminal:
      ```bash
      # Example models mentioned in config:
      ollama pull gemma3
      ollama pull codellama:13b
      # Example for another coding model you mentioned:
      # ollama pull qwen2:7b # Or other qwen2 variants
      ```

## Usage

Activate the environment (`source .venv/bin/activate`).

**To run the Streamlit Web UI:**

```bash
streamlit run app.py
```
Interacts via the terminal. Each run creates a directory in `workplace/`.

Project Structure
-----------------
```
├── .venv/            # Virtual environment managed by uv
├── config/           # Configuration files
│   └── settings.yaml # Agent settings and prompts
├── data/             # Sample data, inputs
├── docs/             # Project documentation
├── notebooks/        # Jupyter notebooks
├── src/              # Main source code
│   ├── core/         # Core logic (config loader, state - TODO)
│   ├── agents/       # Agent node logic
│   ├── tools/        # Tool functions (APIs, utils)
│   └── main.py       # CLI entry point & Graph Creation Function
├── tests/            # Unit and integration tests
├── workplace/        # Timestamped directories for CLI runs (IGNORED BY GIT)
├── workplace_streamlit/ # Timestamped directories for UI runs (IGNORED BY GIT - Add this!)
├── .env.example      # Example environment file
├── .gitignore        # Files ignored by Git
├── app.py            # Streamlit UI application
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

Contributing
------------
(Add contribution guidelines if applicable.)

License
-------
(Specify project license, e.g., MIT, Apache 2.0.)

---

Testing Plan (Updated)
----------------------
Includes testing the Streamlit UI basics:

1. Streamlit UI Test:
   * Run `streamlit run app.py`.
   * Expected: App loads with title, chat input, and log panel.
   * Perform chat, literature search, code generation queries.
   * Expected: Input appears, agent responds, output (text, results list, summary, code block) appears correctly formatted in the main panel. Log panel updates with INFO/ERROR messages. App does not hang on literature search.
2. Coding Agent Context Test (via CLI or UI)
3. Code Extension Test (via CLI or UI - check saved file if UI saving implemented)
4. Multi-line Input Test (via UI)
   * Expected: Pasting multi-line text into Streamlit chat input works.
5. Per-Task LLM Test (Check logs via UI or CLI run)
6. Conditional Download Test (via CLI - UI skips prompt)
7. Numbered Results Test (via CLI - UI doesn't save numbered files yet)
8. Workplace Directory Test (Check CLI runs and `workplace_streamlit` for UI logs)
9. Color Output Test (via CLI)
10. LLM Provider Test (via UI or CLI, check logs/config)
11. Routing Test (via UI or CLI)
12. Query Refinement Test (via UI or CLI, check logs)
13. Literature Search Test (PubMed & ArXiv) (via UI or CLI)
14. Summarization Test (via UI or CLI)
15. Chat Functionality Test (via UI or CLI)
16. Conversational Context Test (Chat) (via UI or CLI)
17. Configuration Test
18. Empty Input Test (via UI or CLI)
19. Error Handling (Basic)
20. Exit Test (via CLI 'quit', or stopping Streamlit server)

