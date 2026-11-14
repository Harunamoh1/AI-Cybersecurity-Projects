"""Tests for the trained model artifact, risk scoring and explanations."""

from __future__ import annotations

import shutil

import pytest

from phishlens.features import InvalidURLError
from phishlens.ml.model import ModelIntegrityError, PhishingModel, risk_level
from tests.conftest import MODEL_DIR, requires_model


@pytest.mark.parametrize(
    "p,expected", [(0.0, "low"), (0.19, "low"), (0.2, "medium"), (0.39, "medium"), (0.4, "high"), (0.69, "high"), (0.7, "critical"), (1.0, "critical")]
)
def test_risk_bands_anchor_on_threshold(p, expected):
    assert risk_level(p, threshold=0.4) == expected


@requires_model
def test_tampered_model_is_refused(tmp_path):
    for f in MODEL_DIR.iterdir():
        if f.is_file():
            shutil.copy(f, tmp_path / f.name)
    with (tmp_path / "phishlens_model.joblib").open("ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ModelIntegrityError):
        PhishingModel.load(tmp_path)


@requires_model
def test_missing_checksum_is_refused(tmp_path):
    shutil.copy(MODEL_DIR / "phishlens_model.joblib", tmp_path / "phishlens_model.joblib")
    with pytest.raises(ModelIntegrityError):
        PhishingModel.load(tmp_path)


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run `python -m phishlens.ml.train`"):
        PhishingModel.load(tmp_path)


def test_analyze_structure(model):
    r = model.analyze("http://paypal.com.account-verify.secure-login.xyz/webscr?cmd=_login")
    assert 0 <= r["probability"] <= 1
    assert r["risk_score"] == pytest.approx(round(r["probability"] * 100, 1))
    assert r["verdict"] == ("phishing" if r["probability"] >= r["threshold"] else "legitimate")
    assert r["risk_level"] == risk_level(r["probability"], r["threshold"])
    assert r["components"]["registered_domain"] == "secure-login.xyz"
    assert r["contributions"], "a high-risk URL should have explanatory contributions"
    impacts = [abs(c["impact"]) for c in r["contributions"]]
    assert impacts == sorted(impacts, reverse=True)


def test_analyze_rejects_invalid(model):
    with pytest.raises(InvalidURLError):
        model.analyze("javascript:alert(document.cookie)")


def test_predictions_are_deterministic(model):
    u = "https://secure-chase-online.com/auth/verify"
    assert model.predict_proba([u])[0] == model.predict_proba([u])[0]


# Behavioural checks on clear-cut examples. These are sanity checks of the
# trained model (not part of the reported metrics, which come from the held-out
# test split). Borderline URLs are deliberately excluded.
CLEAR_PHISHING = [
    "http://paypal.com.account-verify.secure-login.xyz/webscr?cmd=_login",
    "http://appleid.apple.com-verify-account.support/signin/",
    "https://microsoft-office365-login.firebaseapp.com/owa/",
    "http://192.168.4.20/bank/login.php?session=update",
    "http://login-microsoftonline.com.ru-secure.top/owa/auth/verify",
]
CLEAR_LEGITIMATE = [
    "https://en.wikipedia.org/wiki/Phishing",
    "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do",
    "https://www.bbc.co.uk/news/technology",
    "https://github.com/torvalds/linux",
    "https://docs.python.org/3/library/urllib.parse.html",
]


@pytest.mark.parametrize("url", CLEAR_PHISHING)
def test_clear_phishing_is_flagged(model, url):
    assert model.analyze(url)["verdict"] == "phishing"


@pytest.mark.parametrize("url", CLEAR_LEGITIMATE)
def test_clear_legitimate_is_not_flagged(model, url):
    assert model.analyze(url)["verdict"] == "legitimate"


def test_phishing_scores_above_legitimate_on_average(model):
    phish = model.predict_proba(CLEAR_PHISHING).mean()
    legit = model.predict_proba(CLEAR_LEGITIMATE).mean()
    assert phish - legit > 0.5
