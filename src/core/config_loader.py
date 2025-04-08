import yaml
import os
import sys
from typing import Any # <<< Added missing import here

DEFAULT_CONFIG_PATH = "config/settings.yaml"

def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Loads the YAML configuration file.

    Args:
        path: The path to the configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    if not os.path.exists(path):
        print(f"Error: Configuration file not found at {path}")
        print("Please ensure 'config/settings.yaml' exists in the project root.")
        sys.exit(1)

    try:
        with open(path, 'r') as stream:
            config = yaml.safe_load(stream)
        if config is None:
            print(f"Warning: Configuration file at {path} is empty.")
            return {}
        print(f"Configuration loaded successfully from {path}")
        return config
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML configuration file at {path}: {exc}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading configuration from {path}: {e}")
        sys.exit(1)

# Example of how to access nested keys safely
def get_config_value(config: dict, key_path: str, default: Any = None) -> Any:
    """
    Safely retrieves a value from the nested config dictionary using a dot-separated path.

    Args:
        config: The configuration dictionary.
        key_path: Dot-separated path (e.g., "llm_settings.model_name").
        default: The default value to return if the key is not found.

    Returns:
        The configuration value or the default.
    """
    keys = key_path.split('.')
    value = config
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        print(f"Warning: Config key '{key_path}' not found. Using default: {default}")
        return default
