"""Application environment configuration."""

from functools import lru_cache

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def load_environment() -> None:
    """Load values from the project's .env file once."""

    load_dotenv()
