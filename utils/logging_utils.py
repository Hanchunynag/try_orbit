"""Logging helpers."""

from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logger(output_dir: str, logger_name: str = "orbit_experiment") -> logging.Logger:
    """Create a logger that writes both to console and to a log file."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_section(logger: logging.Logger, title: str) -> None:
    """Print a visible section title into the logs."""
    logger.info("=" * 20 + " %s " + "=" * 20, title)
