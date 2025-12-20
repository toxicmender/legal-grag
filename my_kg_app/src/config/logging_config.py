"""Logging configuration helper."""

import logging


def configure_logging(level: str = "INFO"):
    logging.basicConfig(level=level)
