"""Logging setup."""

from __future__ import annotations

import logging


def get_logger(
    name: str = "basicstfm",
    *,
    rank: int = 0,
    is_main_process: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if is_main_process else logging.WARNING)
    logger.propagate = False
    return logger
