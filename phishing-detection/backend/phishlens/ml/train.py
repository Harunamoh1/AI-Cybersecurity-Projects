"""Reproducible training + evaluation pipeline.

Usage::

    python -m phishlens.ml.train --data-dir data --out-dir models

Steps
  1. Download datasets (SHA-256 verified) and load URL + label only.
  2. Canonicalise, drop unparseable / conflicting / duplicate URLs.
  3. Split 64/16/20 train/val/test, stratified by label+source and grouped by
     registrable domain so no domain spans two splits.
  4. Fit candidate models on train only; compare on validation.
  5. Pick the best explainable (feature-based) model by validation ROC-AUC and
     tune its decision threshold on validation (max F1).
  6. Evaluate every candidate exactly once on the untouched test split.
  7. Run robustness experiments (per-source metrics, cross-dataset transfer,
     sample-weighting ablation) and permutation feature importance.
  8. Persist model, metrics JSON, figures and the artifact checksum.

Every number written to ``metrics.json`` is computed here from real
predictions; nothing is hand-entered.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from phishlens import __version__
from phishlens.features import ALL_FEATURES, CATEGORICAL_FEATURES, FEATURE_NAMES
from phishlens.ml.data import (
    DATASETS,
    RANDOM_STATE,
    clean,
    download_datasets,
    featurize,
    grouped_split,
    load_raw,
    sha256_file,
    source_balanced_weights,
)

log = logging.getLogger("phishlens.train")

MODEL_FILENAME = "phishlens_model.joblib"
METRICS_FILENAME = "metrics.json"
CHECKSUM_FILENAME = "phishlens_model.sha256"


# --------------------------------------------------------------------------- #
# Candidate models
# --------------------------------------------------------------------------- #
def _feature_preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            # TargetEncoder uses internal cross-fitting during fit_transform, so a
            # row's own label never leaks into its encoded TLD value.
            ("tld", TargetEncoder(target_type="binary", cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)), list(CATEGORICAL_FEATURES)),
            ("num", numeric, list(FEATURE_NAMES)),
        ],
        verbose_feature_names_out=False,
    )


def build_candidates() -> dict[str, dict]:
    """name -> {pipeline, input: 'features'|'text', explainable: bool}."""
    return {
        "logistic_regression": {
            "pipeline": Pipeline(
                [
                    ("pre", _feature_preprocessor(scale=True)),
                    ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=RANDOM_STATE)),
                ]
            ),
            "input": "features",
            "explainable": True,
        },
        "random_forest": {
            "pipeline": Pipeline(
                [
                    ("pre", _feature_preprocessor(scale=False)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            min_samples_leaf=2,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "input": "features",
            "explainable": True,
        },
        "hist_gradient_boosting": {
            "pipeline": Pipeline(
                [
                    ("pre", _feature_preprocessor(scale=False)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=400,
                            learning_rate=0.08,
                            max_leaf_nodes=63,
                            min_samples_leaf=40,
                            l2_regularization=1.0,
                            early_stopping=False,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "input": "features",
            "explainable": True,
        },
        "char_ngram_logreg": {
            # Reference baseline: character 3-5-grams of the canonical URL. Strong
            # but opaque, so it is reported for comparison and not deployed.
            "pipeline": Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer="char", ngram_range=(3, 5), min_df=3, max_features=300_000,
                            sublinear_tf=True, dtype=np.float32,
                        ),
                    ),
                    ("clf", LogisticRegression(max_iter=3000, C=4.0, solver="liblinear", random_state=RANDOM_STATE)),
                ]
            ),
            "input": "text",
            "explainable": False,
        },
    }


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, proba)) if both_classes else None,
        "brier": float(brier_score_loss(y_true, proba)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[idx])


def _downsample_curve(x: np.ndarray, y: np.ndarray, points: int = 120) -> list[dict]:
    if len(x) <= points:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.linspace(0, len(x) - 1, points).astype(int))
    return [{"x": round(float(x[i]), 5), "y": round(float(y[i]), 5)} for i in idx]


def _predict(entry: dict, X_feat: pd.DataFrame, X_text: pd.Series) -> np.ndarray:
    X = X_feat if entry["input"] == "features" else X_text
    return entry["pipeline"].predict_proba(X)[:, 1]


def _fit(entry: dict, X_feat, X_text, y, weights) -> float:
    X = X_feat if entry["input"] == "features" else X_text
    t0 = time.perf_counter()
    entry["pipeline"].fit(X, y, clf__sample_weight=weights)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save_figures(out_dir: Path, report: dict, y_test, test_probas: dict, final_name: str, importances) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Confusion matrix of the deployed model
    cm = report["models"][final_name]["test"]["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}\n({v / mat[i].sum():.1%})", ha="center", va="center",
                color="white" if v > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {final_name} (test)")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for name, proba in test_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC {report['models'][name]['test']['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test split)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

    # Permutation importance
    top = importances[:15][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.barh([t["feature"] for t in top], [t["importance"] for t in top],
            xerr=[t["std"] for t in top], color="#2f6f8f")
    ax.set_xlabel("Mean drop in ROC-AUC when permuted (validation)")
    ax.set_title(f"Permutation importance - {final_name}")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(data_dir: Path, out_dir: Path, skip_figures: bool = False) -> dict:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = download_datasets(data_dir)
    raw = load_raw(paths)
    df, cleaning = clean(raw)
    log.info("Cleaning report: %s", cleaning)

    tr, va, te = grouped_split(df)
    train, val, test = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    overlap = (set(train["group"]) & set(test["group"])) | (set(train["group"]) & set(val["group"])) | (
        set(val["group"]) & set(test["group"])
    )
    assert not overlap, f"Domain leakage across splits: {list(overlap)[:5]}"

    log.info("Featurising %d URLs", len(df))
    Xtr, Xva, Xte = featurize(train["url"]), featurize(val["url"]), featurize(test["url"])
    Ttr, Tva, Tte = train["canonical"], val["canonical"], test["canonical"]
    ytr, yva, yte = train["label"].to_numpy(), val["label"].to_numpy(), test["label"].to_numpy()
    wtr = source_balanced_weights(train)

    def split_summary(d: pd.DataFrame) -> dict:
        return {
            "rows": len(d),
            "phishing": int(d["label"].sum()),
            "legitimate": int((d["label"] == 0).sum()),
            "domains": int(d["group"].nunique()),
            "by_source": {s: int(n) for s, n in d["source"].value_counts().items()},
        }

    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "random_state": RANDOM_STATE,
        "datasets": [
            {"name": s.name, "url": s.url, "sha256": s.sha256, "license": s.license, "citation": s.citation}
            for s in DATASETS
        ],
        "cleaning": cleaning,
        "splits": {"train": split_summary(train), "validation": split_summary(val), "test": split_summary(test)},
        "split_strategy": "StratifiedGroupKFold by registrable domain (PSL-aware), stratified on label+source",
        "sample_weighting": "equal total weight per (source, label) cell",
        "models": {},
    }

    # ---- fit + validate every candidate -------------------------------------
    candidates = build_candidates()
    val_probas, test_probas = {}, {}
    for name, entry in candidates.items():
        log.info("Training %s", name)
        fit_seconds = _fit(entry, Xtr, Ttr, ytr, wtr)
        val_probas[name] = _predict(entry, Xva, Tva)
        report["models"][name] = {
            "explainable": entry["explainable"],
            "input": entry["input"],
            "fit_seconds": round(fit_seconds, 2),
            "validation": compute_metrics(yva, val_probas[name]),
        }
        log.info("  val ROC-AUC=%.4f F1=%.4f", report["models"][name]["validation"]["roc_auc"],
                 report["models"][name]["validation"]["f1"])

    # ---- model selection on validation only ---------------------------------
    explainable = [n for n, e in candidates.items() if e["explainable"]]
    final_name = max(explainable, key=lambda n: report["models"][n]["validation"]["roc_auc"])
    threshold = best_f1_threshold(yva, val_probas[final_name])
    report["selection"] = {
        "criterion": "highest validation ROC-AUC among explainable feature-based models",
        "selected_model": final_name,
        "decision_threshold": round(threshold, 4),
        "threshold_criterion": "maximum F1 on the validation split",
    }
    log.info("Selected %s, threshold %.4f", final_name, threshold)

    # ---- single evaluation on the held-out test set -------------------------
    for name, entry in candidates.items():
        test_probas[name] = _predict(entry, Xte, Tte)
        report["models"][name]["test"] = compute_metrics(yte, test_probas[name])
    report["models"][final_name]["test_at_tuned_threshold"] = compute_metrics(yte, test_probas[final_name], threshold)

    p_final = test_probas[final_name]
    fpr, tpr, _ = roc_curve(yte, p_final)
    prec, rec, _ = precision_recall_curve(yte, p_final)
    report["curves"] = {
        "roc": _downsample_curve(fpr, tpr),
        "pr": _downsample_curve(rec[::-1], prec[::-1]),
    }
    hist, edges = np.histogram(p_final[yte == 0], bins=20, range=(0, 1))
    hist_p, _ = np.histogram(p_final[yte == 1], bins=20, range=(0, 1))
    report["score_distribution"] = [
        {"bin": round(float(edges[i]), 2), "legitimate": int(hist[i]), "phishing": int(hist_p[i])} for i in range(20)
    ]

    # per-source breakdown on test
    report["per_source_test"] = {}
    for src in test["source"].unique():
        m = (test["source"] == src).to_numpy()
        report["per_source_test"][src] = compute_metrics(yte[m], p_final[m], threshold)

    # ---- robustness experiments (train split only, evaluated on test) -------
    experiments = {}
    # (a) no sample weighting (threshold re-tuned on validation for a fair comparison)
    abl = build_candidates()[final_name]
    _fit(abl, Xtr, Ttr, ytr, None)
    p_abl = _predict(abl, Xte, Tte)
    thr_abl = best_f1_threshold(yva, _predict(abl, Xva, Tva))
    experiments["no_sample_weighting"] = {
        "description": f"{final_name} trained on the same split without source-balanced weights",
        "overall": compute_metrics(yte, p_abl, thr_abl),
        "per_source": {
            src: compute_metrics(yte[(test["source"] == src).to_numpy()], p_abl[(test["source"] == src).to_numpy()], thr_abl)
            for src in test["source"].unique()
        },
    }
    # (b) without the Common Crawl legitimate deep-URL corpus
    m_nocc = (train["source"] != "commoncrawl_legit").to_numpy()
    nocc = build_candidates()[final_name]
    _fit(nocc, Xtr[m_nocc], Ttr[m_nocc], ytr[m_nocc], source_balanced_weights(train[m_nocc]))
    p_nocc = _predict(nocc, Xte, Tte)
    thr_nocc = best_f1_threshold(yva[(val["source"] != "commoncrawl_legit").to_numpy()],
                                 _predict(nocc, Xva[(val["source"] != "commoncrawl_legit").to_numpy()],
                                          Tva[(val["source"] != "commoncrawl_legit").to_numpy()]))
    experiments["without_commoncrawl_corpus"] = {
        "description": f"{final_name} trained without the legitimate deep-URL corpus "
                       "(threshold re-tuned on validation without it), evaluated on the full test split",
        "overall": compute_metrics(yte, p_nocc, thr_nocc),
        "per_source": {
            src: compute_metrics(yte[(test["source"] == src).to_numpy()], p_nocc[(test["source"] == src).to_numpy()], thr_nocc)
            for src in test["source"].unique()
        },
    }
    # (c) cross-dataset transfer: train on one source, test on the other's test rows
    for train_src, test_src in (("phiusiil", "hannousse"), ("hannousse", "phiusiil")):
        mtr = (train["source"] == train_src).to_numpy()
        mte = (test["source"] == test_src).to_numpy()
        cross = build_candidates()[final_name]
        _fit(cross, Xtr[mtr], Ttr[mtr], ytr[mtr], None)
        p_cross = _predict(cross, Xte[mte], Tte[mte])
        experiments[f"train_{train_src}_test_{test_src}"] = {
            "description": f"{final_name} trained only on {train_src} (train split), "
                           f"evaluated on {test_src} (test split)",
            "overall": compute_metrics(yte[mte], p_cross),
        }
    report["experiments"] = experiments

    # ---- permutation importance (validation subsample) -----------------------
    rng = np.random.default_rng(RANDOM_STATE)
    sub = rng.choice(len(Xva), size=min(15_000, len(Xva)), replace=False)
    pi = permutation_importance(
        candidates[final_name]["pipeline"], Xva.iloc[sub], yva[sub], scoring="roc_auc",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importances = sorted(
        (
            {"feature": f, "importance": round(float(m), 5), "std": round(float(s), 5)}
            for f, m, s in zip(ALL_FEATURES, pi.importances_mean, pi.importances_std)
        ),
        key=lambda d: d["importance"],
        reverse=True,
    )
    report["feature_importance"] = importances

    # ---- reference profile for local explanations ---------------------------
    legit = Xtr[ytr == 0]
    reference = {f: float(legit[f].median()) for f in FEATURE_NAMES}
    reference["tld"] = str(legit["tld"].mode().iloc[0])

    artifact = {
        "pipeline": candidates[final_name]["pipeline"],
        "model_name": final_name,
        "feature_names": list(ALL_FEATURES),
        "threshold": float(threshold),
        "reference_values": reference,
        "version": __version__,
        "trained_at": report["generated_at"],
        "sklearn_version": sklearn.__version__,
    }
    model_path = out_dir / MODEL_FILENAME
    joblib.dump(artifact, model_path, compress=3)
    digest = sha256_file(model_path)
    (out_dir / CHECKSUM_FILENAME).write_text(f"{digest}  {MODEL_FILENAME}\n")
    report["artifact"] = {"file": MODEL_FILENAME, "sha256": digest, "bytes": model_path.stat().st_size}
    report["total_seconds"] = round(time.perf_counter() - t_start, 1)

    (out_dir / METRICS_FILENAME).write_text(json.dumps(report, indent=2))
    if not skip_figures:
        _save_figures(out_dir, report, yte, test_probas, final_name, importances)
    log.info("Done in %.1fs. Artifacts in %s", report["total_seconds"], out_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the PhishLens URL classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(args.data_dir, args.out_dir, args.skip_figures)
    sel = report["selection"]["selected_model"]
    print(json.dumps({"selected": sel, "test": report["models"][sel]["test_at_tuned_threshold"]}, indent=2))


if __name__ == "__main__":
    main()
