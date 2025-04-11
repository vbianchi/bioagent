import yaml
import os
import sys
from typing import Any
# Import colorama for colored warnings
from colorama import Fore, Style, init as colorama_init

# Initialize colorama in this module as well
colorama_init(autoreset=True)

DEFAULT_CONFIG_PATH = "config/settings.yaml"

# Define colors used in this module
COLOR_ERROR = Fore.RED
COLOR_WARN = Fore.YELLOW
COLOR_RESET = Style.RESET_ALL

def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Loads the YAML configuration file."""
    if not os.path.exists(path):
        # Use plain print for critical startup errors before logging might be fully set up
        print(f"{COLOR_ERROR}Error: Configuration file not found at {path}{COLOR_RESET}")
        print("Please ensure 'config/settings.yaml' exists in the project root.")
        sys.exit(1)
    try:
        with open(path, 'r') as stream:
            config = yaml.safe_load(stream)
        if config is None:
            print(f"{COLOR_WARN}Warning: Configuration file at {path} is empty.{COLOR_RESET}")
            return {}
        # Keep console clean on successful load
        # print(f"Configuration loaded successfully from {path}")
        return config
    except yaml.YAMLError as exc:
        print(f"{COLOR_ERROR}Error parsing YAML configuration file at {path}: {exc}{COLOR_RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{COLOR_ERROR}An unexpected error occurred while loading config: {e}{COLOR_RESET}")
        sys.exit(1)

def get_config_value(config: dict, key_path: str, default: Any = None) -> Any:
    """Safely retrieves a value from the nested config dictionary."""
    keys = key_path.split('.')
    value = config
    try:
        if not isinstance(config, dict):
             raise TypeError("Config object is not a dictionary")
        for key in keys:
            if not isinstance(value, dict):
                 raise TypeError(f"Intermediate key '{key}' accessed on non-dictionary: {value}")
            value = value[key] # Accesses nested dict
        # Return the found value if it's not None, otherwise return the default
        # Handles cases where key exists but value is explicitly null in YAML
        return value if value is not None else default
    except (KeyError, TypeError) as e:
        # <<< Re-enabled warning print with color >>>
        # Print directly to console as logger might not be fully configured when this is called initially
        print(f"{COLOR_WARN}Warning: Config key '{key_path}' not found or invalid structure in provided config dict (Error: {e}). Using default: {default}{COLOR_RESET}")
        return default

