BioAgent Co-Pilot (Working Title)

A project to develop an agentic AI co-pilot for epidemiology and bioinformatics research tasks. It features a Streamlit web UI and a command-line interface (CLI).

Current Focus: Web UI Development
----------------------------------

Description
-----------
(Add a more detailed description of the project goals and functionalities here.)

Project Status
--------------
Development is currently focused on building a web UI using Streamlit (`app.py`) to provide a better user experience, handle multi-line inputs effectively, and display results more clearly. Core agent features developed in the CLI version are being integrated into the UI.

Implemented Features (in underlying logic, accessible via CLI & basic UI)
-------------------------------------------------------------------------
* Multi-Agent Routing: Classifies user queries into 'chat', 'literature_search', or 'code_generation'.
* Literature Search: Refines queries, searches PubMed/ArXiv, optionally downloads ArXiv PDFs (via CLI prompt), summarizes results.
* Coding Assistance: Generates Python/R code snippets with history context (no execution). Saves with correct extension (CLI only).
* Chat: Conversational responses using LLM and history.
* LLM Flexibility: Supports OpenAI, Gemini, Ollama via config, with per-task LLM for coding.
* Workflow & Output (CLI): LangGraph workflow, external config (`config/settings.yaml`), timestamped `workplace/` directory for logs/results per run, colored console output for CLI.
* Streamlit UI: Basic chat interface, multi-column layout with live log view panel, displays formatted agent outputs (chat, search results, summaries, code).

Next Steps
----------
1. Refine Streamlit UI: Improve display of results, add copy buttons for code, enhance layout.
2. Re-integrate features like conditional PDF download using Streamlit widgets (buttons, etc.).
3. Implement file saving for results generated via the UI.
4. Resume development of advanced agent features (Deep Research, Analysis Agent, Tool Execution) within the UI framework.

Setup
-----
This project uses `uv` for environment and package management.

1. Clone: `git clone https://github.com/vbianchi/bioagent.git && cd bioagent`
2. Environment: `uv venv --python 3.12 && source .venv/bin/activate`
3. Install Dependencies: `uv pip install -r requirements.txt` (includes `streamlit`)
4. Configuration File (`config/settings.yaml`): Create/update `config/settings.yaml`. Choose LLM providers/models.
5. API Keys (`.env`): Create/update `.env` with `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ENTREZ_EMAIL`.
6. Ollama Setup: If using Ollama, install/run service and pull models.

Usage
-----
Activate the environment (`source .venv/bin/activate`).

To run the new Streamlit Web UI:
```
streamlit run app.py
```
Open the provided URL in your browser. Interact via the chat input. Logs appear in the right panel.

To run the original Command-Line Interface (CLI):
Run from the project's root directory:
```
python -m src.main
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

