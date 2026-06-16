"""Lightweight, app-wide logging.

There was previously no logging anywhere — errors went to ``print`` (which does not
reach Streamlit Cloud / Docker logs reliably) or were swallowed. ``get_logger`` gives
a configured, leveled, timestamped logger; the level comes from the ``LOG_LEVEL`` env
var (default ``INFO``) so a deployment can turn up verbosity without code changes.
"""
import logging
import os

_configured = False


def get_logger(name="data_analysis_platform"):
    """Return a configured logger (configures the root handler once, idempotently)."""
    global _configured
    if not _configured:
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, level_name, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _configured = True
    return logging.getLogger(name)
