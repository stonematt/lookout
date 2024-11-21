"""
log_util.py: Provides centralized logging across the application.

Enhanced to support both console and optional file logging, facilitating easier
identification of message sources and maintaining consistent logging practices.
"""

import logging


def app_logger(name, level=logging.INFO, log_file=None):
    """
    Configures a logger with a specified name, format, and log level.
    Optionally supports logging to a file.

    :param name: Logger name, typically __name__ from the importing module.
    :param level: Logging level, defaults to logging.INFO.
    :param log_file: Optional. If provided, logs will also be written to this file.
    :return: Configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # Clear existing handlers to avoid duplicates

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
