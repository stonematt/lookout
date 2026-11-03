"""
log_util.py: Provides centralized logging across the application.

Enhanced to support both console and optional file logging, facilitating easier
identification of message sources and maintaining consistent logging practices.

Functions and names it offers:
- app_logger(name, level, log_file): a module's logger, console and optional
  file, in `LOG_FORMAT`.
- station_context(name, mac): a context manager; every line any app logger
  writes inside it carries the station's name and MAC (#271), so a run that
  handles several stations says which one each line is about. Existing call
  sites stay unchanged: a `StationTag` filter on every handler reads the
  context variable the manager sets.
- LOG_FORMAT, DATE_FORMAT, StationTag.
"""

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s: %(station)s%(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# The station the current code is working on, as its log tag; "" outside one.
_station_tag: ContextVar[str] = ContextVar("lookout_station_tag", default="")


@contextmanager
def station_context(name: Optional[str], mac: str) -> Iterator[None]:
    """
    Tag every log line written inside the block with one station.

    :param name: The station's Ambient name, or None when unnamed.
    :param mac: Its MAC address.
    :return: A context manager; the tag is restored on exit, however the
        block ends.
    """
    tag = f"[{name} {mac}] " if name else f"[{mac}] "
    token = _station_tag.set(tag)
    try:
        yield
    finally:
        _station_tag.reset(token)


class StationTag(logging.Filter):
    """
    Put the current station's tag on each record as `station`.

    Set on every handler `app_logger` makes, so `LOG_FORMAT` can always name
    `%(station)s`; the tag is "" outside `station_context`. It never drops a
    record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Tag the record.

        :param record: The record.
        :return: True, always.
        """
        record.station = _station_tag.get()
        return True


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

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(StationTag())
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(StationTag())
        logger.addHandler(file_handler)

    return logger
