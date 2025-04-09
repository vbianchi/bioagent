# BioAgent Co-Pilot (Working Title)

A project to develop an agentic AI co-pilot for epidemiology and bioinformatics research tasks. It routes queries among chat, literature search, and code generation intents. It searches PubMed/ArXiv, optionally downloads ArXiv PDFs, summarizes results, generates Python/R code snippets (using conversation history), and maintains conversation history. Supports multiple LLM providers with per-task configuration. Each run creates a timestamped folder in `workplace/` with logs and results. Console output is colored. The project structure is modular, separating agents and tools.

## Description

(Add a more detailed description of the project goals and functionalities here.)

## Setup

This project uses `uv` for environment and package management.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vbianchi/bioagent.git](https://github.com/vbianchi/bioagent.git)
    cd bioagent
    ```
2.  **Create and activate the virtual environment:**
    ```bash
    # Create the environment
    uv venv --python 3.12 # Or your desired Python version
    # Activate
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
4.  **Configuration File (`config/settings.yaml`):**
    * Ensure `config/` directory exists.
    * Create or update `config/settings.yaml`.
    * Choose LLM Providers/Models for general and coding tasks.
5.  **API Keys & Environment (`.env`):**
    * Create or update `.env` in the root directory.
    * Add required keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`).
    * Add `ENTREZ_EMAIL='your.email@example.com'`.
    * Ensure `.env` is listed in `.gitignore`.
6.  **Ollama Setup (if using `ollama` provider):**
    * Install and run Ollama: [https://ollama.com/](https://ollama.com/)
    * Pull the desired model(s): e.g., `ollama pull gemma3`, `ollama pull codellama:13b`

## Usage

Activate the environment (`source .venv/bin/activate`). **Run from the project's root directory**:

```bash
python -m src.main
```

*```(Using python -m src.main ensures that Python can correctly find the src package for imports like from src.core.config_loader import ...)```*

The script initializes the agent and prompts for input. Each run creates a new directory in `workplace/` named `YYYYMMDD_HHMMSS_run/`. This directory contains:
-   `logs/run.log`: Detailed log of the agent's operations for that run.
-   `results/`: Output files like `search_results.json` or `summary.txt`.
-   `temp_data/`: Placeholder for any intermediate files (currently unused).

The agent classifies input:
-   'literature_search': Refines query, searches PubMed/ArXiv, displays results, summarizes, saves outputs to the run directory.
-   'chat': Generates a response using history, saves response to the run directory.
-   Empty input: Prompts again.
-   'quit': Exits the session.

## Project Structure
```
├── .venv/            # Virtual environment managed by uv
├── config/           # Configuration files
│   └── settings.yaml # Agent settings and prompts
├── data/             # Sample data, inputs (use .gitignore for large/sensitive data)
├── docs/             # Project documentation
├── notebooks/        # Jupyter notebooks for experiments, EDA
├── src/              # Main source code
│   ├── core/         # Core logic (e.g., config loader)
│   │   ├── __init__.py
│   │   └── config_loader.py
│   ├── agents/       # Agent node logic (router, literature, download, etc.)
│   │   ├── __init__.py
│   │   ├── router.py
│   │   ├── literature.py
│   │   ├── download.py
│   │   ├── summarize.py
│   │   ├── chat.py
│   │   └── coding.py
│   ├── tools/        # Tool functions (API wrappers, utilities)
│   │   ├── __init__.py
│   │   ├── literature_search.py
│   │   ├── pdf_downloader.py
│   │   └── llm_utils.py
│   └── main.py       # Main application entry point (graph definition, loop)
├── tests/            # Unit and integration tests
├── workplace/        # Timestamped directories for each run (IGNORED BY GIT)
│   └──YYYYMMDD_HHMMSS_run/
│       ├── logs/
│       │   ├── run.log
│       │   └── conversation_history.json
│       ├── results/
│       │   └── (e.g., search_results_1.json, summary_1.txt, generated_code_2.py, ...)
│       └── temp_data/
├── .gitignore        # Files and directories ignored by Git
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## Contributing
(Add contribution guidelines if applicable.)

## License
(Specify project license, e.g., MIT, Apache 2.0.)

---

## Testing Plan (Updated)

Includes testing coding agent context:

1.  **Coding Agent Context Test:**
    * Generate a Python script (e.g., "write python function for GC content").
    * On the next turn, ask for a modification or translation (e.g., "rewrite that function in R", "add error handling to the previous function").
    * *Expected:* The agent uses the history to understand "that function" or "the previous function" and generates the relevant R code or modified Python code.
2.  **Code Extension Test**
3.  **Multi-line Input Test** (Verify standard single-line input works)
4.  **Per-Task LLM Test**
5.  **Conditional Download Test**
6.  **Numbered Results Test**
7.  **Workplace Directory Test**
8.  **Color Output Test**
9.  **LLM Provider Test**
10. **Routing Test**
11. **Query Refinement Test**
12. **Literature Search Test (PubMed & ArXiv)**
13. **Summarization Test**
14. **Chat Functionality Test**
15. **Conversational Context Test (Chat)**
16. **Configuration Test**
17. **Empty Input Test**
18. **Error Handling (Basic)**
19. **Exit Test**