"""Model loading, scoring and local explanations used at inference time."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from phishlens.features import (
    ALL_FEATURES,
    FEATURE_DESCRIPTIONS,
    ParsedURL,
    extract_features,
    highlights,
    parse_url,
)

log = logging.getLogger(__name__)

RISK_LEVELS = ("low", "medium", "high", "critical")


class ModelIntegrityError(RuntimeError):
    """The model file on disk does not match its recorded checksum."""


@dataclass
class Contribution:
    feature: str
    description: str
    value: float | str
    reference: float | str
    impact: float  # change in log-odds of phishing vs. the legitimate reference


def risk_level(probability: float, threshold: float) -> str:
    """Map a probability to a risk band anchored on the tuned decision threshold.

    low       p < 0.5 * t
    medium    0.5 * t <= p < t
    high      t <= p < t + (1 - t) / 2
    critical  p >= t + (1 - t) / 2
    """
    if probability < 0.5 * threshold:
        return "low"
    if probability < threshold:
        return "medium"
    if probability < threshold + (1 - threshold) / 2:
        return "high"
    return "critical"


class PhishingModel:
    """Thin wrapper around the persisted scikit-learn pipeline."""

    def __init__(self, artifact: dict[str, Any], metrics: dict[str, Any] | None = None):
        self.pipeline = artifact["pipeline"]
        self.model_name: str = artifact["model_name"]
        self.threshold: float = float(artifact["threshold"])
        self.reference: dict[str, Any] = artifact["reference_values"]
        self.version: str = artifact.get("version", "unknown")
        self.trained_at: str = artifact.get("trained_at", "unknown")
        self.feature_names: list[str] = list(artifact["feature_names"])
        if self.feature_names != list(ALL_FEATURES):
            raise ValueError("Model was trained with a different feature set than this code version")
        self.metrics = metrics or {}

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, model_dir: Path, verify_checksum: bool = True) -> PhishingModel:
        model_path = model_dir / "phishlens_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python -m phishlens.ml.train` first."
            )
        checksum_path = model_dir / "phishlens_model.sha256"
        if verify_checksum:
            # joblib/pickle files execute code on load: refuse anything that is not
            # byte-identical to the artifact the training pipeline produced.
            if not checksum_path.exists():
                raise ModelIntegrityError(f"Missing checksum file {checksum_path}")
            expected = checksum_path.read_text().split()[0]
            actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if expected != actual:
                raise ModelIntegrityError("Model checksum mismatch; refusing to load")
        artifact = joblib.load(model_path)  # noqa: S301 - integrity verified above
        metrics_path = model_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        log.info("Loaded model %s (v%s)", artifact["model_name"], artifact.get("version"))
        return cls(artifact, metrics)

    # --------------------------------------------------------------- predict
    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame.from_records(rows, columns=self.feature_names)

    def predict_proba(self, urls: list[str]) -> np.ndarray:
        rows = [extract_features(u) for u in urls]
        return self.pipeline.predict_proba(self._frame(rows))[:, 1]

    def _log_odds(self, frame: pd.DataFrame) -> np.ndarray:
        if hasattr(self.pipeline, "decision_function"):
            return np.asarray(self.pipeline.decision_function(frame), dtype=float)
        p = np.clip(self.pipeline.predict_proba(frame)[:, 1], 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def explain(self, features: dict[str, Any], top_k: int = 8) -> list[Contribution]:
        """Occlusion-style local explanation in log-odds space.

        For each feature we replace the URL's value with the median value seen
        on *legitimate* training URLs and re-score. The drop in log-odds is how
        strongly that feature pushed this URL towards (positive) or away from
        (negative) a phishing verdict. Log-odds are used rather than probability
        because probabilities saturate near 0 and 1, which would make every
        contribution of a very confident prediction look like zero.
        One batched call scores all perturbations.
        """
        base = self._log_odds(self._frame([features]))[0]
        perturbed, names = [], []
        for name in self.feature_names:
            ref = self.reference[name]
            if features[name] == ref:
                continue
            row = dict(features)
            row[name] = ref
            perturbed.append(row)
            names.append(name)
        if not perturbed:
            return []
        scores = self._log_odds(self._frame(perturbed))
        contribs = [
            Contribution(
                feature=n,
                description=FEATURE_DESCRIPTIONS.get(n, n),
                value=features[n],
                reference=self.reference[n],
                impact=round(float(base - s), 4),
            )
            for n, s in zip(names, scores)
        ]
        contribs = [c for c in contribs if abs(c.impact) >= 0.05]
        contribs.sort(key=lambda c: abs(c.impact), reverse=True)
        return contribs[:top_k]

    def analyze(self, url: str) -> dict[str, Any]:
        """Full analysis of a single URL. Raises InvalidURLError for bad input."""
        t0 = time.perf_counter()
        parsed: ParsedURL = parse_url(url)
        feats = extract_features(parsed)
        proba = float(self.pipeline.predict_proba(self._frame([feats]))[:, 1][0])
        contribs = self.explain(feats)
        level = risk_level(proba, self.threshold)
        return {
            "url": parsed.original,
            "canonical_url": parsed.canonical,
            "verdict": "phishing" if proba >= self.threshold else "legitimate",
            "risk_score": round(proba * 100, 1),
            "risk_level": level,
            "probability": round(proba, 6),
            "threshold": round(self.threshold, 4),
            "components": {
                "scheme": parsed.scheme or None,
                "host": parsed.host,
                "subdomain": parsed.subdomain,
                "domain": parsed.domain,
                "suffix": parsed.suffix,
                "registered_domain": parsed.registered_domain,
                "hosting_platform": parsed.private_suffix or None,
                "port": parsed.port,
                "path": parsed.path,
                "query": parsed.query,
                "is_ip": parsed.is_ip,
            },
            "contributions": [c.__dict__ for c in contribs],
            "indicators": highlights(parsed),
            "features": feats,
            "model": {"name": self.model_name, "version": self.version},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
