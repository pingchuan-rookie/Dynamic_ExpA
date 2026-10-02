"""Dyad logger: routes runtime information that used to be printed to a log file.

Design:
- Standalone logger (name "dyad", propagate=False), not attached to the verl/root logger,
  so it neither affects verl's own logging nor bubbles info records up to the console.
- File handler: both info and debug go to DYAD_LOG_FILE when explicitly set.
  Otherwise they use the shared artifact root's outputs/dyad.log, never cwd/outputs.
  Per-run diagnostics are configured separately by the public training entrypoints.
  The level comes from DYAD_LOG_LEVEL, default INFO; per-token debug output only lands
  when it is set to DEBUG.
- Console handler: only WARNING/ERROR pass through, keeping the training output clean.

Usage:
    from agent_system.utils.logging import get_dyad_logger
    dyad_logger = get_dyad_logger()
    dyad_logger.info("...")     # file only
    dyad_logger.debug("...")    # file only, and only when DYAD_LOG_LEVEL=DEBUG
    dyad_logger.warning("...")  # file + console
"""

import logging
import os
import sys

from agent_system.utils.artifact_paths import artifact_root

_LOGGER_NAME = "dyad"
_configured = False


def get_dyad_logger() -> logging.Logger:
    """Return the Dyad logger, configuring it lazily. Safe to call repeatedly."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    level_name = os.environ.get("DYAD_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False  # keep records off the root logger so verl's logging stays untouched

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s:%(process)d] %(message)s"
    )

    log_file = os.environ.get("DYAD_LOG_FILE") or str(artifact_root() / "outputs" / "dyad.log")
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except Exception as exc:  # a failed log file is not fatal: degrade to console WARNING only
        sys.stderr.write(f"[dyad_logging] cannot open log file {log_file}: {exc!r}\n")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    _configured = True
    logger.info("Dyad logger initialized: file=%s level=%s", log_file, level_name)
    return logger
