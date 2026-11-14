"""Evaluation metrics. Everything reported in ``models/metrics.json`` comes from here."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (robust for tiny n)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _r(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


def downsample_curve(x: np.ndarray, y: np.ndarray, points: int = 200) -> list[list[float]]:
    """Keep a curve's shape with at most ``points`` vertices (log-spaced on x for ROC)."""
    if len(x) <= points:
        return [[_r(a, 5), _r(b, 5)] for a, b in zip(x, y)]
    grid = np.unique(np.concatenate([[0.0], np.logspace(-6, 0, points - 1)]))
    yi = np.interp(grid, x, y)
    return [[_r(a, 6), _r(b, 5)] for a, b in zip(grid, yi)]


def multiclass_report(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None,
                      classes: list[str]) -> dict[str, Any]:
    """Accuracy, macro/weighted P/R/F1, per-class metrics with Wilson CIs, ROC-AUC, confusion matrix."""
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {}
    for i, c in enumerate(classes):
        tp = int(((y_true == i) & (y_pred == i)).sum())
        lo, hi = wilson_interval(tp, int(s[i]))
        per_class[c] = {
            "precision": _r(p[i]), "recall": _r(r[i]), "f1": _r(f[i]), "support": int(s[i]),
            "recall_ci95": [_r(lo, 4), _r(hi, 4)],
        }
        if proba is not None and 0 < s[i] < len(y_true):
            per_class[c]["roc_auc"] = _r(roc_auc_score(y_true == i, proba[:, i]))
    out: dict[str, Any] = {
        "accuracy": _r(accuracy_score(y_true, y_pred)),
        "macro_precision": _r(p.mean()),
        "macro_recall": _r(r.mean()),
        "macro_f1": _r(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": _r(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if proba is not None:
        present = [i for i in labels if 0 < (y_true == i).sum() < len(y_true)]
        out["roc_auc_macro_ovr"] = _r(np.mean([roc_auc_score(y_true == i, proba[:, i]) for i in present]))
    return out


def binary_report(is_attack: np.ndarray, flagged: np.ndarray, score: np.ndarray | None) -> dict[str, Any]:
    """Attack-vs-benign view: what a SOC analyst sees as alerts."""
    is_attack = is_attack.astype(bool)
    flagged = flagged.astype(bool)
    tp = int((is_attack & flagged).sum())
    fp = int((~is_attack & flagged).sum())
    fn = int((is_attack & ~flagged).sum())
    tn = int((~is_attack & ~flagged).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out: dict[str, Any] = {
        "precision": _r(prec),
        "recall": _r(rec),
        "f1": _r(2 * prec * rec / (prec + rec) if prec + rec else 0.0),
        "false_positive_rate": _r(fp / (fp + tn) if fp + tn else 0.0, 7),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if score is not None and 0 < is_attack.sum() < len(is_attack):
        out["roc_auc"] = _r(roc_auc_score(is_attack, score))
    return out


def binary_roc(is_attack: np.ndarray, score: np.ndarray, points: int = 200) -> list[list[float]]:
    fpr, tpr, _ = roc_curve(is_attack, score)
    return downsample_curve(fpr, tpr, points)


def class_roc_curves(y_true: np.ndarray, proba: np.ndarray, classes: list[str], points: int = 150) -> dict[str, Any]:
    curves = {}
    for i, c in enumerate(classes):
        pos = y_true == i
        if 0 < pos.sum() < len(y_true):
            fpr, tpr, _ = roc_curve(pos, proba[:, i])
            curves[c] = downsample_curve(fpr, tpr, points)
    return curves


def calibration_report(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> dict[str, Any]:
    """Top-label expected calibration error (ECE) and reliability-diagram bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    ece, table = 0.0, []
    n = len(confidence)
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(confidence[m].mean()), float(correct[m].mean())
        ece += m.sum() / n * abs(conf - acc)
        table.append({"bin": [_r(edges[b], 3), _r(edges[b + 1], 3)], "confidence": _r(conf, 4),
                      "accuracy": _r(acc, 4), "count": int(m.sum())})
    return {"ece": _r(ece), "bins": table}


def nll(proba: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0))))
