"""Application configuration.

Single source of truth for every environment-driven setting. All values are
loaded from environment variables (or a local ``.env`` during development) via
pydantic-settings. Filesystem paths are resolved relative to the backend root so
the service runs identically regardless of the current working directory.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> parents[2] == backend/
BASE_DIR: Path = Path(__file__).resolve().parents[2]


def _env_snapshot() -> dict[str, str]:
    """Merge .env file values with the live process environment (process wins)."""
    dot: dict[str, str] = {}
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        dot = {k: v for k, v in dotenv_values(str(env_path)).items() if v is not None}
    return {**dot, **os.environ}


class Settings(BaseSettings):
    """Typed application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "WellBeing Agent"
    APP_VERSION: str = "3.0.0"
    ENVIRONMENT: str = Field(default="development")  # development | production
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # ── CORS ───────────────────────────────────────────────────────────────
    # Comma-separated list of allowed frontend origins. Defaults to "*" (all
    # origins) since this is a public API with no credentials. Set to a specific
    # Vercel URL in production via the ALLOWED_ORIGINS env var if needed.
    ALLOWED_ORIGINS: str = "*"

    # ── LLM ────────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "openrouter"
    LLM_MODEL: str = "nvidia/nemotron-3-super-120b-a12b:free"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MAX_TOKENS: int = 1500
    LLM_TEMPERATURE: float = 0.3
    LLM_TIMEOUT_SECONDS: float = 60.0

    # Primary key fallback (used only when no OPENROUTER_API_KEY<N> vars are found).
    LLM_API_KEY: str = ""
    LLM_API_KEY_2: str = ""  # legacy — superseded by OPENROUTER_API_KEY<N>
    LLM_API_KEY_3: str = ""  # legacy — superseded by OPENROUTER_API_KEY<N>

    # ── Embeddings / Index ─────────────────────────────────────────────────
    # Run through fastembed (ONNX Runtime) — no PyTorch, ~50 MB runtime RAM,
    # so the same local-quality embedding model works on the Render free plan.
    # MUST match the model the persisted index was built with; index.py reads
    # data/cancer_index_store/index_metadata.json and overrides this if they
    # disagree, so retrieval never silently breaks.
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_PATH: str = ""  # resolved below; defaults to data/cancer_index_store
    DATASET_PATH: str = ""  # resolved below; defaults to data/DataSet/...
    SIMILARITY_TOP_K: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Cache ──────────────────────────────────────────────────────────────
    CACHE_TTL_HOURS: int = 24
    CACHE_SIMILARITY_THRESHOLD: float = 0.85
    # Hard caps. Both files are read and rewritten in full on every turn, so
    # without a bound a long-running instance would grow its memory and per-
    # request latency without limit. Oldest entries are dropped first.
    CACHE_MAX_ENTRIES: int = 500
    CONVERSATION_LOG_MAX_ENTRIES: int = 1000

    # ── Voice (faster-whisper) ──────────────────────────────────────────────
    # CTranslate2-based Whisper, CPU-friendly. "small" (~500 MB int8) is the
    # default; use "base" (~150 MB) on very memory-constrained instances.
    # Voice endpoint degrades gracefully and tells users to type if disabled
    # or unavailable.
    ENABLE_VOICE: bool = False
    WHISPER_MODEL_ID: str = "small"
    MAX_AUDIO_SIZE_MB: int = 10
    MAX_QUERY_LENGTH: int = 2000

    # ── Runtime (writable) directories ─────────────────────────────────────
    # On ephemeral hosts (Render) this lives on the container's writable disk.
    RUNTIME_DIR: str = ""  # resolved below; defaults to backend/var

    # ── Derived values ─────────────────────────────────────────────────────
    @property
    def cors_origins(self) -> List[str]:
        v = self.ALLOWED_ORIGINS.strip()
        if v in ("", "*"):
            return ["*"]
        return [o.strip() for o in v.split(",") if o.strip()]

    # ── Resolved paths ─────────────────────────────────────────────────────
    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def index_dir(self) -> Path:
        if self.INDEX_PATH:
            p = Path(self.INDEX_PATH)
            return p if p.is_absolute() else BASE_DIR / p
        return self.data_dir / "cancer_index_store"

    @property
    def dataset_file(self) -> Path:
        if self.DATASET_PATH:
            p = Path(self.DATASET_PATH)
            return p if p.is_absolute() else BASE_DIR / p
        return self.data_dir / "DataSet" / "breast_cancer_comprehensive.json"

    @property
    def runtime_dir(self) -> Path:
        if self.RUNTIME_DIR:
            p = Path(self.RUNTIME_DIR)
            return p if p.is_absolute() else BASE_DIR / p
        return BASE_DIR / "var"

    @property
    def cache_file(self) -> Path:
        return self.runtime_dir / "cache" / "response_cache.json"

    @property
    def conversations_file(self) -> Path:
        return self.runtime_dir / "conversations.json"

    @property
    def recordings_dir(self) -> Path:
        return self.runtime_dir / "recorded_audio"

    @property
    def api_keys(self) -> List[str]:
        """Return all OPENROUTER_API_KEY<N> values in numbered order, falling back to LLM_API_KEY."""
        env = _env_snapshot()
        keys: List[str] = []
        i = 1
        while True:
            v = env.get(f"OPENROUTER_API_KEY{i}", "").strip()
            if not v:
                break
            keys.append(v)
            i += 1
        return keys if keys else ([self.LLM_API_KEY] if self.LLM_API_KEY else [])

    @property
    def models(self) -> List[str]:
        """Return all OPENROUTER_MODEL_<N> values in numbered order, falling back to LLM_MODEL."""
        env = _env_snapshot()
        models: List[str] = []
        i = 1
        while True:
            v = env.get(f"OPENROUTER_MODEL_{i}", "").strip()
            if not v:
                break
            models.append(v)
            i += 1
        return models if models else [self.LLM_MODEL]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton settings instance."""
    return Settings()


settings = get_settings()
