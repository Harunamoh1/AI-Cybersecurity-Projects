from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

MODEL_DIR = BACKEND_ROOT / "models"
HAS_MODEL = (MODEL_DIR / "phishlens_model.joblib").exists()
requires_model = pytest.mark.skipif(not HAS_MODEL, reason="trained model not present; run python -m phishlens.ml.train")


@pytest.fixture(scope="session")
def model():
    if not HAS_MODEL:
        pytest.skip("trained model not present")
    from phishlens.ml.model import PhishingModel

    return PhishingModel.load(MODEL_DIR)


@pytest.fixture()
def client(model):
    from fastapi.testclient import TestClient

    from phishlens.api.config import Settings
    from phishlens.api.main import create_app

    settings = Settings(database_path=":memory:", rate_limit_per_minute=0, cors_origins=["http://localhost:3000"])
    app = create_app(settings=settings, model=model)
    with TestClient(app) as c:
        yield c
