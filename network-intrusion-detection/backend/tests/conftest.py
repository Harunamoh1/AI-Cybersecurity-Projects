from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

MODEL_DIR = BACKEND_ROOT / "models"
SAMPLES_DIR = BACKEND_ROOT / "samples"
HAS_MODEL = (MODEL_DIR / "sentinel_xgb.ubj").exists()
requires_model = pytest.mark.skipif(not HAS_MODEL, reason="trained model not present; run python -m sentinelai.training.train")


@pytest.fixture(scope="session")
def detector():
    if not HAS_MODEL:
        pytest.skip("trained model not present")
    from sentinelai.inference.detector import Detector

    return Detector.load(MODEL_DIR)


@pytest.fixture(scope="session")
def mixed_csv() -> bytes:
    return (SAMPLES_DIR / "mixed_traffic.csv").read_bytes()


@pytest.fixture(scope="session")
def mixed_df():
    import pandas as pd

    return pd.read_csv(SAMPLES_DIR / "mixed_traffic.csv", dtype=str)


def make_settings(tmp_path=None, **overrides):
    from sentinelai.api.config import Settings

    base = dict(database_path=":memory:", rate_limit_per_minute=0, cors_origins=["http://localhost:3000"],
                model_dir=MODEL_DIR, samples_dir=SAMPLES_DIR, api_key="")
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def client(detector):
    from fastapi.testclient import TestClient

    from sentinelai.api.main import create_app

    with TestClient(create_app(settings=make_settings(), detector=detector)) as c:
        yield c
