"""
This file is a slightly modified version of
https://github.com/Soh-Lab/imager_python/blob/main/hardware/config/__init__.py
"""

import os
from typing import Optional
from loguru import logger
from pydantic import ValidationError
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only used on Python < 3.11
    import tomli as tomllib

from .config_schema import Config

config: Optional[Config] = None


def load_config(config_path: Optional[str] = None):
    global config
    if config_path is None:
        # Default path: config/example_config.toml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "example_config.toml")

    if not os.path.exists(config_path):
        error_message = f"Configuration file not found at {config_path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    try:
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        error_message = f"Error parsing TOML file: {e}"
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        new_config = Config(**config_data)
    except ValidationError as e:
        error_message = f"Configuration validation error: {e}"
        logger.error(error_message)
        raise ValueError(error_message)

    if config is not None:
        # Update the existing config object in place
        config.__dict__.update(new_config.__dict__)
    else:
        # First-time initialization
        config = new_config


# Optionally load the default config if it exists (for backward compatibility with CLI tools)
# but do not fail if it doesn't exist (e.g. when importing aswift package)
_default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_config.toml")
if os.path.exists(_default_config_path):
    try:
        load_config(_default_config_path)
    except Exception as e:
        logger.info(f"Could not load default config from {_default_config_path}: {e}")
