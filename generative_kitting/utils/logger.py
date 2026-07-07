"""
Structured logging for the Generative Kitting System.

Uses loguru for rich, structured log output with file rotation,
coloured terminal output, and per-module context tagging.
"""

import os
import sys
from datetime import datetime
from loguru import logger


def setup_logger(config: dict) -> "logger":
    """
    Configure the global loguru logger from the logging section of config.yaml.

    Parameters
    ----------
    config : dict
        The full parsed config.yaml dict.  Expected keys under ``logging``:
        - level : str          (e.g. "INFO", "DEBUG")
        - log_file : str       (filename inside log_dir)
        - log_dir : str        (directory for log files)
        - save_vlm_images : bool
        - save_task_plans : bool

    Returns
    -------
    logger
        The configured loguru logger instance.
    """
    log_cfg = config.get("logging", {})
    level = log_cfg.get("level", "INFO").upper()
    log_dir = log_cfg.get("log_dir", "logs")
    log_file = log_cfg.get("log_file", "kitting_session.log")

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Remove default handler
    logger.remove()

    # ── Console handler ──────────────────────────────────────
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # ── File handler (with rotation) ─────────────────────────
    log_path = os.path.join(log_dir, log_file)
    logger.add(
        log_path,
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    logger.info(f"Logger initialised — level={level}, file={log_path}")
    return logger


def get_session_id() -> str:
    """Generate a unique session ID based on the current timestamp."""
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


# ── Convenience re-export ────────────────────────────────────
# Modules can do:  from utils.logger import log
# and use log.info(), log.warning(), etc.
log = logger
