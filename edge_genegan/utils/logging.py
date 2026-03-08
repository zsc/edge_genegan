"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("edge_genegan")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    if log_file is not None:
        f = Path(log_file)
        f.parent.mkdir(parents=True, exist_ok=True)
        fhandler = logging.FileHandler(str(f))
        fhandler.setFormatter(fmt)
        logger.addHandler(fhandler)
    return logger
