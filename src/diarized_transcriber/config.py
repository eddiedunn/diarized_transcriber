"""Service configuration via environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class ServiceSettings(BaseSettings):
    """Configuration for the diarized transcriber service.

    All settings can be overridden via environment variables
    prefixed with ``DIARIZED_``.
    """

    redis_url: str = "redis://localhost:6379"
    model_size: str = "large-v3-turbo"
    host: str = "0.0.0.0"
    port: int = 8000
    idle_timeout: int = 300
    extract_embeddings: bool = True
    speaker_db_path: str = "./speaker_profiles_db"
    identify_speakers: bool = False
    auto_enroll_speakers: bool = False
    match_threshold: float = 0.4

    model_config = {"env_prefix": "DIARIZED_"}


@lru_cache
def get_settings() -> ServiceSettings:
    """Return cached service settings."""
    return ServiceSettings()
