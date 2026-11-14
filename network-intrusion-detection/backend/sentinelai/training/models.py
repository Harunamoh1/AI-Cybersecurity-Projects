"""Candidate models. Hyperparameters are fixed a priori (not tuned on the test set);
XGBoost uses the validation split for early stopping only."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sentinelai.features import signed_log1p

SUPERVISED = ("random_forest", "xgboost", "decision_tree", "logistic_regression")
DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
    "isolation_forest": "Isolation Forest",
}


def class_weights(y: np.ndarray, n_classes: int, power: float = 0.5, cap: float = 100.0) -> np.ndarray:
    """Per-class weights (n_max / n_c) ** power, capped.

    ``power=0.5`` is a compromise: full inverse-frequency weighting (power=1)
    buys a little minority recall at the price of many more false alarms on
    benign traffic, which matters in an IDS where benign flows dominate.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = (counts.max() / np.maximum(counts, 1)) ** power
    return np.minimum(w, cap)


def build(name: str, seed: int, n_jobs: int, n_classes: int) -> Any:
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=1,
                                      n_jobs=n_jobs, random_state=seed)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=1000, learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=1.0, tree_method="hist", max_bin=256, objective="multi:softprob",
            num_class=n_classes, eval_metric="mlogloss", early_stopping_rounds=30,
            n_jobs=n_jobs, random_state=seed,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=25, min_samples_leaf=2, random_state=seed)
    if name == "logistic_regression":
        return make_pipeline(
            FunctionTransformer(signed_log1p),
            StandardScaler(),
            LogisticRegression(max_iter=400, C=1.0, random_state=seed),
        )
    if name == "isolation_forest":
        # Unsupervised: fitted on benign training flows only.
        return make_pipeline(
            FunctionTransformer(signed_log1p),
            IsolationForest(n_estimators=200, max_samples=1024, random_state=seed, n_jobs=n_jobs),
        )
    raise ValueError(name)
