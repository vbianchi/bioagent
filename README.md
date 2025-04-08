# BioAgent Co-Pilot (Working Title)

A project to develop an agentic AI co-pilot for epidemiology and bioinformatics research tasks. It routes queries between chat and literature search intents. For literature searches, it refines the query, searches PubMed/ArXiv, and summarizes the results. Chat responses consider conversation history. The agent supports OpenAI, Google Gemini, and local Ollama models, configured via `config/settings.yaml`.

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
    Create or update the `requirements.txt` file with the content specified in this document. **Note the change from `langchain-community` to `langchain-ollama`**.
    ```bash
    # Ensure old community version potentially installed is removed if causing conflicts
    # uv pip uninstall langchain-community
    uv pip install -r requirements.txt
    ```
4.  **Configuration File (`config/settings.yaml`):**
    * Ensure the `config/` directory exists.
    * Create or update `config/settings.yaml` with the content provided in this document.
    * **Choose your LLM Provider:** Edit the `llm_provider` setting to `"openai"`, `"gemini"`, or `"ollama"`.
    * **Set Model Names:** Adjust the corresponding model name setting (e.g., `openai_model_name`, `gemini_model_name`, `ollama_model_name`). For Ollama, ensure the model is pulled locally.
5.  **API Keys & Environment (`.env`):**
    * Create or update `.env` in the root directory.
    * **OpenAI:** If using `openai`, add `OPENAI_API_KEY='...'`.
    * **Gemini:** If using `gemini`, add `GOOGLE_API_KEY='...'`.
    * **NCBI Entrez:** Add `ENTREZ_EMAIL='your.email@example.com'`.
    * Ensure `.env` is listed in `.gitignore`.
6.  **Ollama Setup (if using `ollama` provider):**
    * Install and run Ollama locally: [https://ollama.com/](https://ollama.com/)
    * Pull the desired model specified in `config/settings.yaml`: e.g., `ollama pull llama3`

## Usage

Activate the environment (`source .venv/bin/activate`). **Run the script from the project's root directory**:

```bash
python -m src.main
```

The script will initialize the LLM specified in `config/settings.yaml` and prompt for input in a loop (type 'quit' to exit). It classifies input:
-   If 'literature_search', it refines the query, searches PubMed/ArXiv, displays results, and provides a summary.
-   If 'chat', it routes to the chat function which uses the LLM and conversation history to respond.

## Project Structure
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
├── .gitignore        # Files and directories ignored by Git
├── requirements.txt  # Project dependencies
└── README.md         # This file


## Contributing
(Add contribution guidelines if applicable.)

## License
(Specify project license, e.g., MIT, Apache 2.0.)
