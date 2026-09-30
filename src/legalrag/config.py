from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables live here; override any of them with RAG_<NAME> env vars or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    data_dir: Path = Path("data")
    corpus_url: str = "https://lecomarquage.service-public.fr/vdd/3.4/part/zip/vosdroits-latest.zip"

    # Chunking (characters, not tokens: keeps the pipeline tokenizer-agnostic)
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = False
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    candidates: int = 30  # pool size fed to fusion / reranking
    top_k: int = 5  # chunks passed to the LLM

    # Generation
    llm_provider: Literal["anthropic", "openai_compat"] = "anthropic"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # For openai_compat: LM Studio defaults to http://localhost:1234/v1, Ollama to :11434/v1
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "not-needed"
    max_tokens: int = 4096

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"
