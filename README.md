# BioAgent Co-Pilot (Working Title)

A project to develop an agentic AI co-pilot for epidemiology and bioinformatics research tasks. Currently supports basic PubMed literature search.

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
    Create or update the `requirements.txt` file with the content specified in this document.
    ```bash
    uv pip install -r requirements.txt
    ```
4.  **API Key & Configuration Setup:**
    * Create a file named `.env` in the root directory (`bioagent/`).
    * Add your OpenAI API key: `OPENAI_API_KEY='your-actual-openai-api-key-here'`
    * **Add your email address for NCBI Entrez:** `ENTREZ_EMAIL='your.email@example.com'` (NCBI requires this for identification when using their APIs).
    * Ensure `.env` is listed in your `.gitignore` file.

## Usage

Activate the environment (`source .venv/bin/activate`) and run the main script:

```bash
python src/main.py
```

## Project Structure
├── .venv/            # Virtual environment managed by uv
├── config/           # Configuration files (API keys, settings - use .gitignore!)
├── data/             # Sample data, inputs (use .gitignore for large/sensitive data)
├── docs/             # Project documentation
├── notebooks/        # Jupyter notebooks for experiments, EDA
├── src/              # Main source code
│   ├── agents/       # Code for specialized agents (Literature, Coding, etc.)
│   ├── core/         # Core orchestration logic (e.g., LangGraph setup)
│   ├── tools/        # Wrappers for external APIs, bioinformatics tools
│   └── main.py       # Main application entry point
├── tests/            # Unit and integration tests
├── .gitignore        # Files and directories ignored by Git
└── README.md         # This file


## Contributing
(Add contribution guidelines if applicable.)

## License
(Specify project license, e.g., MIT, Apache 2.0.)
