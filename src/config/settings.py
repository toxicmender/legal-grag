"""Load settings from environment using python-dotenv or pydantic.

Placeholder `Settings` dataclass.
"""

from pydantic import BaseSettings

class Settings(BaseSettings):
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    openai_api_key: str | None = None

    class Config:
        env_file = ".env"

