"""Runtime detector: load the deployed model and score / explain flows.

Artifact format (``models/``), deliberately pickle-free so that loading a model
can never execute code:

* ``sentinel_xgb.ubj``  — XGBoost booster in its native binary JSON format
* ``model_meta.json``   — classes, preprocessing parameters, calibration
  temperature, training reference ranges
* ``SHA256SUMS``        — checksums verified before anything is loaded
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import softmax

from sentinelai.features import ColumnReport, Preprocessor, SchemaError, canonicalize
from sentinelai.schema import BENIGN, CLASS_INFO, FEATURE_BY_NAME, SEVERITY_ORDER, try_map_label

log = logging.getLogger(__name__)

BOOSTER_FILE = "sentinel_xgb.ubj"
META_FILE = "model_meta.json"
CHECKSUM_FILE = "SHA256SUMS"
LOW_CONFIDENCE = 0.6


class ModelIntegrityError(RuntimeError):
    """A model file does not match its recorded checksum."""


def severity_for(cls: str, confidence: float) -> str:
    """Class base severity, downgraded one level when the model is unsure."""
    base = CLASS_INFO[cls]["severity"]
    if base == "none":
        return "none"
    if confidence < LOW_CONFIDENCE:
        return SEVERITY_ORDER[max(1, SEVERITY_ORDER.index(base) - 1)]
    return base


def confidence_band(confidence: float) -> str:
    if confidence >= 0.9:
        return "high"
    if confidence >= LOW_CONFIDENCE:
        return "medium"
    return "low"


@dataclass
class PreparedInput:
    X: pd.DataFrame  # preprocessed model input, one row per valid flow
    row_index: np.ndarray  # position of each valid flow in the original input
    raw: pd.DataFrame  # canonical, numeric, pre-imputation values of the valid flows
    imputed_cells: np.ndarray  # per valid flow: how many feature cells were missing/invalid
    rejected_rows: list[int]  # input rows with too few usable values
    columns: ColumnReport
    labels: pd.Series | None  # ground-truth taxonomy labels if the input carried a Label column


@dataclass
class Predictions:
    classes: list[str]
    proba: np.ndarray  # calibrated probabilities, shape (n, n_classes)
    pred: np.ndarray  # predicted class index
    confidence: np.ndarray
    attack_probability: np.ndarray  # 1 - P(BENIGN)

    def labels(self) -> list[str]:
        return [self.classes[i] for i in self.pred]

    def severity(self) -> list[str]:
        return [severity_for(self.classes[i], float(c)) for i, c in zip(self.pred, self.confidence)]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Detector:
    def __init__(self, booster: xgb.Booster, meta: dict[str, Any]):
        self.booster = booster
        self.meta = meta
        self.classes: list[str] = list(meta["classes"])
        self.benign_index = self.classes.index(BENIGN)
        self.preprocessor = Preprocessor.from_dict(meta["preprocessor"])
        self.features = self.preprocessor.features
        self.temperature = float(meta.get("temperature", 1.0))
        self.version: str = meta.get("version", "unknown")
        self.reference: dict[str, list[float]] = meta.get("reference_ranges", {})
        self._explainer = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: Path, verify_checksum: bool = True) -> Detector:
        model_dir = Path(model_dir)
        booster_bytes = (model_dir / BOOSTER_FILE).read_bytes()
        meta_bytes = (model_dir / META_FILE).read_bytes()
        if verify_checksum:
            expected = {}
            for line in (model_dir / CHECKSUM_FILE).read_text().splitlines():
                if line.strip():
                    digest, name = line.split(maxsplit=1)
                    expected[name.strip().lstrip("*")] = digest
            for name, data in ((BOOSTER_FILE, booster_bytes), (META_FILE, meta_bytes)):
                if expected.get(name) != sha256_bytes(data):
                    raise ModelIntegrityError(f"Checksum mismatch for {name}")
        booster = xgb.Booster()
        booster.load_model(bytearray(booster_bytes))
        booster.set_param({"nthread": 1})
        meta = json.loads(meta_bytes)
        log.info("Loaded model %s (%d features, T=%.3f)", meta.get("version"), len(meta["preprocessor"]["features"]),
                 meta.get("temperature", 1.0))
        return cls(booster, meta)

    # ------------------------------------------------------------- preparation
    def prepare(self, df: pd.DataFrame, max_invalid_fraction: float = 0.5) -> PreparedInput:
        """Canonicalise raw CICFlowMeter columns and preprocess them for the model.

        Raises :class:`SchemaError` if required features are missing. Rows where
        more than ``max_invalid_fraction`` of the features are missing or
        invalid are rejected instead of being silently imputed.
        """
        canon, labels, report = canonicalize(df)
        missing = [f for f in self.features if f not in canon.columns]
        if missing:
            shown = ", ".join(FEATURE_BY_NAME[m].cic2017 for m in missing[:8])
            more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            raise SchemaError(f"Missing {len(missing)} required flow feature column(s): {shown}{more}")
        invalid = self.preprocessor.invalid_mask(canon)
        n_invalid = invalid.sum(axis=1).to_numpy()
        ok = n_invalid <= max_invalid_fraction * len(self.features)
        rows = np.flatnonzero(ok)
        valid = canon.iloc[rows]
        X = self.preprocessor.transform(valid)
        raw_vals = valid[self.features].apply(pd.to_numeric, errors="coerce")
        mapped = None
        if labels is not None:
            mapped = labels.iloc[rows].map(try_map_label).reset_index(drop=True)
        return PreparedInput(
            X=X.reset_index(drop=True).astype("float32"),
            row_index=rows,
            raw=raw_vals.reset_index(drop=True),
            imputed_cells=n_invalid[rows],
            rejected_rows=np.flatnonzero(~ok).tolist(),
            columns=report,
            labels=mapped,
        )

    # --------------------------------------------------------------- inference
    def _dmatrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X[self.features].to_numpy(dtype=np.float32), feature_names=self.features)

    def logits(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._dmatrix(X), output_margin=True)

    def predict(self, X: pd.DataFrame) -> Predictions:
        z = self.logits(X)
        proba = softmax(z / self.temperature, axis=1)
        pred = proba.argmax(axis=1)
        return Predictions(
            classes=self.classes,
            proba=proba,
            pred=pred,
            confidence=proba.max(axis=1),
            attack_probability=1.0 - proba[:, self.benign_index],
        )

    # ---------------------------------------------------------- explainability
    @property
    def explainer(self):
        if self._explainer is None:
            import shap  # imported lazily: heavy import, only needed for explanations

            self._explainer = shap.TreeExplainer(self.booster)
        return self._explainer

    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Exact TreeSHAP via the SHAP library.

        Returns ``(values, base)`` with values of shape (n, n_features, n_classes)
        in raw logit units, and base of shape (n_classes,).
        """
        vals = np.asarray(self.explainer.shap_values(X[self.features].to_numpy(dtype=np.float32)))
        if vals.ndim == 3 and vals.shape[0] == len(self.classes) and vals.shape[1] == len(X):
            vals = np.transpose(vals, (1, 2, 0))  # older SHAP: list per class
        base = np.asarray(self.explainer.expected_value, dtype=float).reshape(-1)
        return vals, base

    def fast_contributions(self, X: pd.DataFrame) -> np.ndarray:
        """XGBoost's built-in TreeSHAP (same algorithm, C++ batch implementation).

        Shape (n, n_classes, n_features + 1); last column is the bias. Used for
        bulk "top reasons" on large uploads; tests check it equals SHAP output.
        """
        return self.booster.predict(self._dmatrix(X), pred_contribs=True)

    def top_reasons(self, X: pd.DataFrame, pred: np.ndarray, k: int = 3) -> list[list[dict[str, Any]]]:
        """Top-k features pushing each flow toward its predicted class."""
        contrib = self.fast_contributions(X)[:, :, :-1]
        out = []
        values = X[self.features].to_numpy()
        for i, c in enumerate(pred):
            row = contrib[i, c] / self.temperature
            top = np.argsort(-row)[:k]
            out.append([{"feature": self.features[j], "value": float(values[i, j]), "impact": round(float(row[j]), 4)}
                        for j in top if row[j] > 0])
        return out

    def explain(self, X: pd.DataFrame, raw: pd.DataFrame | None = None, top_k: int = 12) -> list[dict[str, Any]]:
        """Local explanation of each flow, from the SHAP library.

        For the predicted class: contributions to its (temperature-scaled) logit.
        ``attack_evidence``: contributions *against* the BENIGN logit, i.e. what
        made the flow look malicious regardless of the attack type.
        """
        preds = self.predict(X)
        vals, base = self.shap_values(X)
        vals = vals / self.temperature
        base = base / self.temperature
        out = []
        Xv = X[self.features].to_numpy()
        for i in range(len(X)):
            c = int(preds.pred[i])
            sv = vals[i, :, c]
            order = np.argsort(-np.abs(sv))[:top_k]
            ev = -vals[i, :, self.benign_index]
            ev_order = np.argsort(-ev)[:top_k]

            def item(j: int, v: float, i: int = i) -> dict[str, Any]:
                f = FEATURE_BY_NAME[self.features[j]]
                rv = None if raw is None else raw.iloc[i][self.features[j]]
                ref = self.reference.get(self.features[j])
                return {
                    "feature": f.name,
                    "label": f.cic2017,
                    "description": f.description,
                    "unit": f.unit,
                    "value": float(Xv[i, j]),
                    "raw_value": None if rv is None or pd.isna(rv) or np.isinf(rv) else float(rv),
                    "benign_median": None if ref is None else ref[2],
                    "impact": round(float(v), 5),
                }

            out.append({
                "predicted_class": self.classes[c],
                "confidence": float(preds.confidence[i]),
                "attack_probability": float(preds.attack_probability[i]),
                "probabilities": {k: float(p) for k, p in zip(self.classes, preds.proba[i])},
                "base_value": float(base[c]),
                "logit": float(base[c] + sv.sum()),
                "contributions": [item(j, sv[j]) for j in order],
                "attack_evidence": [item(j, ev[j]) for j in ev_order if ev[j] > 0],
                "other_features_impact": round(float(sv.sum() - sv[order].sum()), 5),
            })
        return out

    # --------------------------------------------------------------- drift
    def distribution_shift(self, X: pd.DataFrame, top: int = 5) -> dict[str, Any]:
        """Fraction of feature values outside the training [p0.1, p99.9] range.

        A high rate means the traffic looks unlike CIC-IDS2017 (different network,
        different CICFlowMeter version) and predictions deserve less trust.
        """
        if not self.reference or len(X) == 0:
            return {"out_of_range_rate": 0.0, "features": []}
        rates = {}
        for f in self.features:
            lo, hi = self.reference[f][0], self.reference[f][1]
            col = X[f].to_numpy()
            rates[f] = float(((col < lo) | (col > hi)).mean())
        overall = float(np.mean(list(rates.values())))
        worst = sorted(rates.items(), key=lambda kv: -kv[1])[:top]
        peak = worst[0][1] if worst else 0.0
        # By design ~0.2% of in-distribution values fall outside [p0.1, p99.9].
        # A single feature far outside its training range is already a warning
        # sign (CSE-CIC-IDS2018 traffic: mean 1%, worst feature 16%).
        if overall > 0.05 or peak > 0.10:
            level = "high"
        elif overall > 0.01 or peak > 0.02:
            level = "moderate"
        else:
            level = "low"
        return {
            "out_of_range_rate": round(overall, 5),
            "max_feature_rate": round(peak, 5),
            "level": level,
            "features": [{"feature": k, "rate": round(v, 4)} for k, v in worst if v > 0],
        }


def validate_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise SchemaError("The file contains no flow records")
