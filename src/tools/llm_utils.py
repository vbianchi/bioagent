import sys
import os # <<< Added missing import
import logging
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

# Import config loader utils
from src.core.config_loader import get_config_value # Assuming config is loaded globally in main

logger = logging.getLogger(__name__)

def initialize_llm(config: dict, provider_config_path: str, settings_config_path: str) -> Optional[BaseChatModel]:
    """
    Initializes an LLM provider based on configuration.

    Args:
        config: The loaded configuration dictionary.
        provider_config_path: Path to the provider setting (e.g., "llm_provider").
        settings_config_path: Path to the settings section for this provider (e.g., "llm_settings").

    Returns:
        An initialized BaseChatModel instance or None if failed.
    """
    provider = get_config_value(config, provider_config_path, "default").lower()
    main_provider = get_config_value(config, "llm_provider", "openai").lower() # Get the global provider

    # If provider is 'default', use the main llm_provider setting
    if provider == "default":
        provider = main_provider
        # Adjust settings path to use the main settings if provider was default
        settings_config_path = "llm_settings"
        logger.info(f"Using default LLM provider for '{provider_config_path}': {provider}")
    else:
         logger.info(f"Attempting to initialize LLM for '{provider_config_path}': provider={provider}")

    temp = get_config_value(config, f"{settings_config_path}.temperature", 0)
    llm_instance = None

    # Dynamically get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not openai_api_key: logger.error(f"Provider is '{provider}' but OPENAI_API_KEY not found."); return None
        model = get_config_value(config, f"{settings_config_path}.openai_model_name", "gpt-3.5-turbo")
        try:
            llm_instance = ChatOpenAI(model=model, temperature=temp, openai_api_key=openai_api_key)
            logger.info(f"Initialized OpenAI LLM: model={model}, temp={temp}")
        except Exception as e: logger.error(f"Error initializing OpenAI LLM: {e}", exc_info=True)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not google_api_key: logger.error(f"Provider is '{provider}' but GOOGLE_API_KEY not found."); return None
        model = get_config_value(config, f"{settings_config_path}.gemini_model_name", "gemini-1.5-flash-latest")
        try:
            llm_instance = ChatGoogleGenerativeAI(model=model, temperature=temp, google_api_key=google_api_key)
            logger.info(f"Initialized Google Gemini LLM: model={model}, temp={temp}")
        except Exception as e: logger.error(f"Error initializing Google Gemini LLM: {e}", exc_info=True)

    elif provider == "ollama":
        try: from langchain_ollama import ChatOllama
        except ImportError: logger.error("langchain-ollama package not found."); return None
        model = get_config_value(config, f"{settings_config_path}.ollama_model_name", "gemma3")
        base_url = get_config_value(config, f"{settings_config_path}.ollama_base_url")
        try:
            init_params = {"model": model, "temperature": temp};
            if base_url: init_params["base_url"] = base_url
            llm_instance = ChatOllama(**init_params); llm_instance.invoke("test connection")
            logger.info(f"Initialized Ollama LLM: model={model}, temp={temp}, base_url={base_url or 'default'}")
        except Exception as e:
            logger.error(f"Error initializing/connecting to Ollama LLM: {e}", exc_info=False) # Don't need full traceback for connection error
            logger.error(f"Ensure Ollama is running and model '{model}' is available.")
            llm_instance = None # Ensure instance is None if connection fails
    else:
        logger.error(f"Unknown LLM provider '{provider}' specified for '{provider_config_path}'.")

    return llm_instance
