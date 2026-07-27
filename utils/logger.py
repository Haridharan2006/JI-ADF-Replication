# utils/logger.py

import logging
import os


def get_logger(log_dir="logs", log_file="training.log"):
    """
    Create and configure a logger.

    Logs are written both to:
    1. Console
    2. Log file
    """

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("JIADF")

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_file)
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger