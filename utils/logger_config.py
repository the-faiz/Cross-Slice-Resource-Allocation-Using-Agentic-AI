import logging
import os

def setup_logger(log_file="logs/project.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure global logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

def get_logger(name: str):
    """Get a logger for a specific module/component."""
    return logging.getLogger(name)
