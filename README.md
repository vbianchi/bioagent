# BioAgent Co-Pilot (Working Title)

A project to develop an agentic AI co-pilot for epidemiology and bioinformatics research tasks. It routes queries among chat, literature search, and code generation intents. It searches PubMed/ArXiv, optionally downloads ArXiv PDFs, summarizes results, generates Python code snippets, and maintains conversation history. Supports multiple LLM providers (OpenAI, Gemini, Ollama) with per-task configuration. Each run creates a timestamped folder in `workplace/` with logs and results. Console output is colored.

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
    Make sure `uv` is installed.
    ```bash
    # Create the environment (uses .venv directory by default)
    uv venv --python 3.12 # Or your desired Python version

    # Activate the environment
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
4.  **Configuration File (`config/settings.yaml`):**
    * Ensure `config/` directory exists.
    * Create or update `config/settings.yaml`.
    * **Choose LLM Providers:** Edit `llm_provider` (for general tasks) and `coding_agent_settings.llm_provider` (for coding tasks). Use `"openai"`, `"gemini"`, `"ollama"`, or `"default"` (to use the main provider).
    * **Set Model Names:** Adjust corresponding model names (e.g., `openai_model_name`, `gemini_model_name`, `ollama_model_name`, and those under `coding_agent_settings`). Ensure models are available/pulled.
5.  **API Keys & Environment (`.env`):**
    * Create or update `.env` in the root directory.
    * Add required keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY` depending on providers chosen).
    * Add `ENTREZ_EMAIL='your.email@example.com'`.
    * Ensure `.env` is listed in `.gitignore`.
6.  **Ollama Setup (if using `ollama` provider):**
    * Install and run Ollama: [https://ollama.com/](https://ollama.com/)
    * Pull the desired model(s): e.g., `ollama pull gemma3`, `ollama pull codellama`

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
│   ├── agents/       # Code for specialized agents - TODO Refactor
│   ├── tools/        # Wrappers for external APIs - TODO Refactor
│   └── main.py       # Main application entry point
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

Includes testing the coding agent:

1.  **Coding Agent Test:**
    * Configure a suitable coding LLM in `config/settings.yaml` (e.g., set `coding_agent_settings.llm_provider` to `openai` and `openai_model_name` to `gpt-4-turbo-preview`, or use `ollama` with `codellama`). Ensure API keys/Ollama setup is correct.
    * Input a code generation request (e.g., "write python function to calculate GC content", "generate matplotlib code to plot column 'A' vs 'B' from a pandas dataframe named 'my_df'").
    * *Expected:* Router classifies as 'code_generation', `call_coding_agent` runs, generated Python code is printed to the console and saved to `results/generated_code_X.py`. Verify the code looks reasonable (it won't be executed).
2.  **Per-Task LLM Test:** Verify that the correct LLM (main vs. coding) is reported as being used in the logs when routing to different agents.
3.  **Conditional Download Test**
4.  **Numbered Results Test**
5.  **Workplace Directory Test**
6.  **Color Output Test**
7.  **LLM Provider Test**
8.  **Routing Test**
9.  **Query Refinement Test**
10. **Literature Search Test (PubMed & ArXiv)**
11. **Summarization Test**
12. **Chat Functionality Test**
13. **Conversational Context Test**
14. **Configuration Test**
15. **Empty Input Test**
16. **Error Handling (Basic)**
17. **Exit Test**