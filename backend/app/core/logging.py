"""Centralized logging configuration."""

import logging

from app.core.config import settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure root logging once, using the level from settings."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, ensuring logging is configured."""
    configure_logging()
    return logging.getLogger(name)
