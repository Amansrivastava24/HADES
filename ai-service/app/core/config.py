from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration and environment settings."""
    model_config = SettingsConfigDict(
        env_file=(".env", "ai-service/.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    ENVIRONMENT: str = "development"

    # OpenAI LLM Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_REASONING_MODEL_ID: str = "gpt-4o"

    # Embedding Provider Configuration
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_DIMENSION: int = 1536

    # PostgreSQL Database Configuration
    DATABASE_URL: Optional[str] = None
    DATABASE_POOL_MIN_SIZE: int = 1
    DATABASE_POOL_MAX_SIZE: int = 10
    DATABASE_POOL_TIMEOUT: float = 30.0

    # Tavily Search Configuration
    TAVILY_API_KEY: Optional[str] = None
    TAVILY_API_URL: str = "https://api.tavily.com/search"


settings = Settings()
