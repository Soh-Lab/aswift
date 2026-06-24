"""
This file is a slightly modified version of
https://github.com/Soh-Lab/imager_python/blob/main/hardware/config/__init__.py
"""

import os
from typing import Optional
from loguru import logger
import tomli
from pydantic import ValidationError

from peak_extraction.config.config_schema import Config

config: Config


def load_config(config_path: Optional[str] = None):
    global config
    if config_path is None:
        # Default path: config/wells1_config.toml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "example_config.toml")

    if not os.path.exists(config_path):
        error_message = f"Configuration file not found at {config_path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    try:
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)
    except tomli.TOMLDecodeError as e:
        error_message = f"Error parsing TOML file: {e}"
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        new_config = Config(**config_data)
    except ValidationError as e:
        error_message = f"Configuration validation error: {e}"
        logger.error(error_message)
        raise ValueError(error_message)

    if 'config' in globals():
        # Update the existing config object in place
        config.__dict__.update(new_config.__dict__)
    else:
        # First-time initialization
        config = new_config


# Initial load of the config
load_config()
