from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a local .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://docs_qa:docs_qa@localhost:5432/docs_qa"
    docs_paths: str = "sample_docs,README.md"

    # Chunking. Words are a cheap proxy for tokens (about 1.3 tokens per English word).
    chunk_max_words: int = 400
    chunk_overlap_words: int = 60

    embedder: Literal["fastembed", "hash", "openai"] = "fastembed"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_cache_dir: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embed_model: str = "text-embedding-3-small"

    llm_provider: Literal["extractive", "anthropic", "openai"] = "extractive"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    openai_chat_model: str = "gpt-4o-mini"

    top_k: int = 5

    @property
    def doc_paths(self) -> list[str]:
        return [p.strip() for p in self.docs_paths.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
