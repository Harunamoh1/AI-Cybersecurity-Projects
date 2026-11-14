"""Runtime configuration, read from ``SENTINEL_*`` environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(f"SENTINEL_{name}", default)


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", str(_BACKEND_ROOT / "models"))))
    samples_dir: Path = field(default_factory=lambda: Path(_env("SAMPLES_DIR", str(_BACKEND_ROOT / "samples"))))
    database_path: str = field(default_factory=lambda: _env("DB_PATH", str(_BACKEND_ROOT / "sentinel.db")))
    cors_origins: list[str] = field(default_factory=lambda: _list(_env("CORS_ORIGINS", "http://localhost:3000")))
    # Optional shared secret. When set, every /api/v1 route requires the
    # X-API-Key header; the Next.js server injects it, browsers never see it.
    api_key: str = field(default_factory=lambda: _env("API_KEY", ""))
    max_upload_mb: float = field(default_factory=lambda: float(_env("MAX_UPLOAD_MB", "25")))
    max_rows: int = field(default_factory=lambda: int(_env("MAX_ROWS", "100000")))
    max_columns: int = field(default_factory=lambda: int(_env("MAX_COLUMNS", "300")))
    max_batch: int = field(default_factory=lambda: int(_env("MAX_BATCH", "1000")))
    max_explain: int = field(default_factory=lambda: int(_env("MAX_EXPLAIN", "25")))
    retention_analyses: int = field(default_factory=lambda: int(_env("RETENTION_ANALYSES", "200")))
    rate_limit_per_minute: int = field(default_factory=lambda: int(_env("RATE_LIMIT", "60")))
    inference_threads: int = field(default_factory=lambda: int(_env("THREADS", "2")))
    verify_model_checksum: bool = field(default_factory=lambda: _env("VERIFY_CHECKSUM", "true").lower() != "false")

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def get_settings() -> Settings:
    return Settings()
