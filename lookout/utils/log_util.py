"""
log_util.py: Provides centralized logging across the application.

Enhanced to support both console and optional file logging, facilitating easier
identification of message sources and maintaining consistent logging practices.
"""

import logging
import os


def app_logger(name, level=None, log_file=None):
    """
    Configures a logger with a specified name, format, and log level.
    Optionally supports logging to a file.

    :param name: Logger name, typically __name__ from the importing module.
    :param level: Logging level or None to use LOGLEVEL env.
    :param log_file: Optional. If provided, logs will also be written to this file.
    :return: Configured logger instance.
    """
    # Resolve level from env or fallback
    env_level = os.getenv("LOGLEVEL", "").upper()
    resolved_level = getattr(logging, env_level, None) if env_level else None
    level = level or resolved_level or logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
