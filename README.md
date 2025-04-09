# BioAgent Co-Pilot (Working Title)

A project to develop an agentic AI co-pilot for epidemiology and bioinformatics research tasks. It routes queries between chat and literature search intents. For literature searches, it refines the query, searches PubMed/ArXiv, and summarizes the results. Chat responses consider conversation history. The agent supports OpenAI, Google Gemini, and local Ollama models, configured via `config/settings.yaml`. Each run creates a timestamped folder in the `workplace/` directory containing logs and results. Console output is colored for readability.

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
    * Ensure the `config/` directory exists.
    * Create or update `config/settings.yaml`.
    * **Choose LLM Provider:** Edit `llm_provider` setting (`openai`, `gemini`, `ollama`).
    * **Set Model Names:** Adjust corresponding model name setting. Default Ollama model is `gemma3`. Ensure model is pulled locally if using Ollama.
5.  **API Keys & Environment (`.env`):**
    * Create or update `.env` in the root directory.
    * Add required keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY` depending on provider).
    * Add `ENTREZ_EMAIL='your.email@example.com'`.
    * Ensure `.env` is listed in `.gitignore`.
6.  **Ollama Setup (if using `ollama` provider):**
    * Install and run Ollama: [https://ollama.com/](https://ollama.com/)
    * Pull the desired model: e.g., `ollama pull gemma3`

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
│   ├── agents/       # Code for specialized agents - TODO
│   ├── tools/        # Wrappers for external APIs - TODO Refactor
│   └── main.py       # Main application entry point
├── tests/            # Unit and integration tests
├── workplace/        # Timestamped directories for each run (IGNORED BY GIT)
│   └── YYYYMMDD_HHMMSS_run/
│       ├── logs/
│       │   └── run.log
│       ├── results/
│       │   └── (e.g., search_results.json, summary.txt)
│       └── temp_data/
├── .gitignore        # Files and directories ignored by Git
├── requirements.txt  # Project dependencies
└── README.md         # This file       # This file
```

## Contributing
(Add contribution guidelines if applicable.)

## License
(Specify project license, e.g., MIT, Apache 2.0.)

---

## Testing Plan (Updated)

Includes testing numbered result files:

1.  **Numbered Results Test:**
    * Run the script (`python -m src.main`).
    * Perform a literature search (interaction 1).
    * Perform a chat query (interaction 2).
    * Perform another literature search (interaction 3).
    * *Expected:* Check the `workplace/RUN_DIR/results/` folder. It should contain files like `search_results_1.json`, `summary_1.txt`, `chat_response_2.txt`, `search_results_3.json`, `summary_3.txt`. Check `logs/conversation_history.json` contains all 3 interactions.
2.  **Workplace Directory Test**
3.  **Color Output Test**
4.  **LLM Provider Test**
5.  **Routing Test**
6.  **Query Refinement Test**
7.  **Literature Search Test (PubMed & ArXiv)**
8.  **Summarization Test**
9.  **Chat Functionality Test**
10. **Conversational Context Test**
11. **Configuration Test**
12. **Empty Input Test**
13. **Error Handling (Basic)**
14. **Exit Test**
