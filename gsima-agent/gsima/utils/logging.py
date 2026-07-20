"""Logging configuration and utilities."""
import logging
import os
from datetime import datetime
from gsima.utils import config

def setup_logging():
    """Sets up the logger to write to the outputs/logs directory."""
    env_name = config.GYM_ENVIRONMENT or "UnknownEnv"
    log_dir = os.path.join("outputs", "logs", env_name)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"agent_run_{timestamp}.log")

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")
