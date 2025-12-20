"""Serving configuration helpers."""

from pydantic import BaseSettings

class ServingConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
