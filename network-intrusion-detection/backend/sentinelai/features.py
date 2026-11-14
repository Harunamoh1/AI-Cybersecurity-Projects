"""Column canonicalisation, feature selection and preprocessing.

The same code path is used for the training data and for CSV files uploaded
to the API, so a flow is transformed identically at training and inference
time. Everything that is *learned* from data (selected features, imputation
medians) is fitted on the training split only and serialised as plain JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sentinelai.schema import (
    FEATURE_NAMES,
    IDENTIFIER_COLUMNS,
    LABEL_KEYS,
    POLICY_EXCLUDED,
    SENTINEL_FEATURES,
    canonical_name,
    normalize_key,
)


class SchemaError(ValueError):
    """The input does not contain the flow features the model needs."""


@dataclass
class ColumnReport:
    mapped: dict[str, str] = field(default_factory=dict)  # raw -> canonical
    identifiers_dropped: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    label_column: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mapped": len(self.mapped),
            "identifiers_dropped": self.identifiers_dropped,
            "unexpected_columns": self.unexpected,
            "label_column": self.label_column,
        }


def canonicalize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, ColumnReport]:
    """Rename recognised CICFlowMeter columns to canonical names.

    Returns ``(features, labels_or_None, report)``. Identifier columns (IPs,
    ports, flow id, timestamp) are dropped so they can never reach the model or
    the history store; unknown columns are reported and ignored.
    """
    report = ColumnReport()
    positions: list[int] = []
    names: list[str] = []
    label = None
    # Positional iteration: duplicated header names must not select two columns.
    for pos, col in enumerate(df.columns):
        key = normalize_key(col)
        name = canonical_name(col)
        if name == "__duplicate__":
            continue
        if name is not None:
            if name in names:  # first occurrence wins
                continue
            positions.append(pos)
            names.append(name)
            report.mapped[str(col)] = name
        elif key in LABEL_KEYS:
            if label is None:
                label = df.iloc[:, pos]
                report.label_column = str(col)
        elif key in IDENTIFIER_COLUMNS:
            report.identifiers_dropped.append(str(col))
        else:
            report.unexpected.append(str(col))
    out = df.iloc[:, positions].copy()
    out.columns = names
    return out, label, report


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to float64; unparsable cells become NaN."""
    return df.apply(pd.to_numeric, errors="coerce").astype("float64")


def sanitize(X: pd.DataFrame) -> pd.DataFrame:
    """Replace values CICFlowMeter produces in error by NaN.

    * ±inf in rate features (zero-duration flows divide by zero).
    * Negative values, except the documented ``-1`` "not observed" sentinel of
      the initial TCP window features. Negative durations/IATs are clock
      artefacts of the original capture.
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if col in SENTINEL_FEATURES:
            X.loc[X[col] < -1, col] = np.nan
        else:
            X.loc[X[col] < 0, col] = np.nan
    return X


# --------------------------------------------------------------------------- selection


@dataclass
class SelectionReport:
    selected: list[str]
    dropped: dict[str, str]  # feature -> reason


def select_features(X_train: pd.DataFrame, corr_threshold: float = 0.99, sample: int = 300_000,
                    seed: int = 42) -> SelectionReport:
    """Unsupervised, training-split-only feature selection.

    1. policy exclusions (``destination_port``)
    2. constant columns
    3. exact duplicates of an earlier column
    4. near-duplicates: |Spearman rho| >= ``corr_threshold`` with an earlier kept
       column. Removing them does not cost accuracy for tree models but stops
       SHAP from splitting credit between redundant copies of the same signal.
    """
    dropped: dict[str, str] = {}
    cols = [c for c in FEATURE_NAMES if c in X_train.columns]
    for c in list(cols):
        if c in POLICY_EXCLUDED:
            dropped[c] = f"policy: {POLICY_EXCLUDED[c]}"
    cols = [c for c in cols if c not in dropped]

    Xs = X_train[cols]
    if len(Xs) > sample:
        Xs = Xs.sample(sample, random_state=seed)
    for c in cols:
        if Xs[c].nunique(dropna=True) <= 1 and X_train[c].nunique(dropna=True) <= 1:
            dropped[c] = "constant in training data"
    cols = [c for c in cols if c not in dropped]

    kept: list[str] = []
    for c in cols:
        twin = next((k for k in kept if Xs[c].equals(Xs[k]) and X_train[c].equals(X_train[k])), None)
        if twin:
            dropped[c] = f"identical to {twin}"
        else:
            kept.append(c)

    ranks = Xs[kept].rank().fillna(0.0)
    corr = np.corrcoef(ranks.to_numpy(dtype="float64"), rowvar=False)
    final: list[str] = []
    idx = {c: i for i, c in enumerate(kept)}
    for c in kept:
        twin = next((k for k in final if abs(corr[idx[c], idx[k]]) >= corr_threshold), None)
        if twin:
            dropped[c] = f"|Spearman rho| >= {corr_threshold} with {twin} ({corr[idx[c], idx[twin]]:.4f})"
        else:
            final.append(c)
    return SelectionReport(selected=final, dropped=dropped)


# --------------------------------------------------------------------------- preprocessing


@dataclass
class Preprocessor:
    """Select features, sanitise and median-impute. Fitted on training data only."""

    features: list[str]
    medians: dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        clean = sanitize(X[self.features])
        med = clean.median()
        self.medians = {c: float(0.0 if np.isnan(v) else v) for c, v in med.items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise SchemaError(f"Missing required feature columns: {', '.join(missing)}")
        sub = X[self.features]
        if all(pd.api.types.is_numeric_dtype(sub[c]) for c in self.features):
            sub = sub.astype("float64")
        else:
            sub = to_numeric(sub)
        return sanitize(sub).fillna(self.medians)

    def invalid_mask(self, X: pd.DataFrame) -> pd.DataFrame:
        """Boolean frame: which cells are missing/invalid before imputation."""
        return sanitize(to_numeric(X[self.features])).isna()

    def to_dict(self) -> dict[str, Any]:
        return {"features": self.features, "medians": self.medians}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Preprocessor:
        return cls(features=list(d["features"]), medians={k: float(v) for k, v in d["medians"].items()})


def signed_log1p(X: np.ndarray) -> np.ndarray:
    """Variance-stabilising transform for heavy-tailed counters (linear models)."""
    return np.sign(X) * np.log1p(np.abs(X))
