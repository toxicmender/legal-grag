"""Dependency injection for db clients, config, etc."""

from ..config.settings import Settings


def get_settings() -> Settings:
    return Settings()