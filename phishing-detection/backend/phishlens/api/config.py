"""Runtime configuration (12-factor: everything from environment variables)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("PHISHLENS_MODEL_DIR", _BACKEND_ROOT / "models")))
    database_path: str = field(
        default_factory=lambda: os.getenv("PHISHLENS_DB_PATH", str(_BACKEND_ROOT / "phishlens.db"))
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _list(os.getenv("PHISHLENS_CORS_ORIGINS", "http://localhost:3000"))
    )
    max_url_length: int = field(default_factory=lambda: int(os.getenv("PHISHLENS_MAX_URL_LENGTH", "2048")))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("PHISHLENS_MAX_BATCH", "100")))
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv("PHISHLENS_RATE_LIMIT", "120")))
    verify_model_checksum: bool = field(
        default_factory=lambda: os.getenv("PHISHLENS_VERIFY_CHECKSUM", "true").lower() != "false"
    )


def get_settings() -> Settings:
    return Settings()
