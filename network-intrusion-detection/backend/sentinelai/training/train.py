"""End-to-end, reproducible training and evaluation pipeline.

    python -m sentinelai.training.train            # full run (≈1 h on 4 cores)
    python -m sentinelai.training.train --help

Steps
  1. download + verify CIC-IDS2017, clean, de-duplicate (cached as parquet)
  2. primary protocol: stratified 70/15/15 split; train five models; evaluate on test
  3. calibrate the deployed model (temperature scaling on validation)
  4. global SHAP analysis of the deployed model
  5. robustness protocol: chronological split, all models retrained
  6. cross-dataset experiments with CSE-CIC-IDS2018 (optional, ``--cross-dataset``)
  7. write the pickle-free artifact, metrics.json and figures

The test split is used only for final reporting; model selection, early
stopping, calibration and thresholds use the validation split.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle  # noqa: S403 - only to *measure* sklearn model size, never to load
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.special import softmax

from sentinelai import __version__
from sentinelai.data.loading import clean, load_cicids2017
from sentinelai.data.sources import (
    BACKEND_ROOT,
    CICIDS2017_ARCHIVE,
    CICIDS2017_CITATION,
    RAW_DIR,
    cicids2017_csvs,
)
from sentinelai.data.splitting import stratified_split, temporal_split
from sentinelai.features import Preprocessor, select_features, signed_log1p
from sentinelai.inference.detector import BOOSTER_FILE, CHECKSUM_FILE, META_FILE, Detector, sha256_bytes
from sentinelai.schema import BENIGN, CLASS_INFO, CLASSES, FEATURE_BY_NAME
from sentinelai.training import evaluate as ev
from sentinelai.training.calibration import fit_temperature
from sentinelai.training.models import DISPLAY_NAMES, SUPERVISED, build, class_weights

log = logging.getLogger("sentinelai.train")

SEED = 42
PROCESSED = BACKEND_ROOT / "data" / "processed"
MODEL_DIR = BACKEND_ROOT / "models"
DEPLOYED = "xgboost"
BENIGN_IDX = CLASSES.index(BENIGN)


# ----------------------------------------------------------------------------- data


def load_clean_2017(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = PROCESSED / "cicids2017_clean.parquet"
    stats_file = PROCESSED / "cicids2017_clean_stats.json"
    if cache.exists() and stats_file.exists():
        log.info("Using cached cleaned data %s", cache)
        return pd.read_parquet(cache), json.loads(stats_file.read_text())
    raw = load_cicids2017(cicids2017_csvs(raw_dir))
    fd = clean(raw)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fd.frame.to_parquet(cache, index=False)
    stats_file.write_text(json.dumps(fd.stats, indent=2, default=int))
    return fd.frame, fd.stats


def encode(labels: pd.Series) -> np.ndarray:
    return labels.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)


# ------------------------------------------------------------------------- protocol


def model_size_bytes(model: Any) -> int:
    if isinstance(model, xgb.XGBClassifier):
        return len(model.get_booster().save_raw("ubj"))
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def _predict(model: Any, X: np.ndarray) -> tuple[np.ndarray, float]:
    t = time.perf_counter()
    proba = model.predict_proba(X)
    return proba, time.perf_counter() - t


def run_protocol(frame: pd.DataFrame, split: np.ndarray, models: tuple[str, ...], n_jobs: int,
                 keep_models: bool = False) -> dict[str, Any]:
    """Fit feature selection/preprocessing on train, train + evaluate every model."""
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    sel = select_features(frame.loc[tr], seed=SEED)
    pre = Preprocessor(sel.selected).fit(frame.loc[tr])
    X = {k: pre.transform(frame.loc[m]).to_numpy(dtype=np.float32) for k, m in (("train", tr), ("val", va), ("test", te))}
    y = {k: encode(frame.loc[m, "label"]) for k, m in (("train", tr), ("val", va), ("test", te))}
    cw = class_weights(y["train"], len(CLASSES))
    sw = cw[y["train"]]
    out: dict[str, Any] = {
        "selection": sel, "preprocessor": pre, "X": X, "y": y, "class_weights": cw,
        "split_counts": {k: {c: int((y[k] == i).sum()) for i, c in enumerate(CLASSES)} for k in y},
        "results": {}, "models": {},
    }
    for name in models:
        log.info("[%s] training on %d flows x %d features", name, len(y["train"]), X["train"].shape[1])
        model = build(name, SEED, n_jobs, len(CLASSES))
        t0 = time.perf_counter()
        if name == "xgboost":
            model.fit(X["train"], y["train"], sample_weight=sw, eval_set=[(X["val"], y["val"])],
                      sample_weight_eval_set=[cw[y["val"]]], verbose=False)
        elif name == "logistic_regression":
            model.fit(X["train"], y["train"], logisticregression__sample_weight=sw)
        else:
            model.fit(X["train"], y["train"], sample_weight=sw)
        fit_s = time.perf_counter() - t0
        proba_val, _ = _predict(model, X["val"])
        proba, pred_s = _predict(model, X["test"])
        pred = proba.argmax(1)
        res = {
            "display_name": DISPLAY_NAMES[name],
            "train_seconds": round(fit_s, 1),
            "inference_us_per_flow": round(pred_s / len(pred) * 1e6, 3),
            "model_size_bytes": model_size_bytes(model),
            "validation": {"macro_f1": ev.multiclass_report(y["val"], proba_val.argmax(1), None, list(CLASSES))["macro_f1"]},
            "test": ev.multiclass_report(y["test"], pred, proba, list(CLASSES)),
            "binary_test": ev.binary_report(y["test"] != BENIGN_IDX, pred != BENIGN_IDX, 1 - proba[:, BENIGN_IDX]),
            "binary_roc": ev.binary_roc(y["test"] != BENIGN_IDX, 1 - proba[:, BENIGN_IDX]),
        }
        if name == "xgboost":
            res["best_iteration"] = int(model.best_iteration)
            res["n_trees"] = int((model.best_iteration + 1) * len(CLASSES))
        log.info("[%s] %.0fs  val macro-F1 %.4f  test macro-F1 %.4f  acc %.4f", name, fit_s,
                 res["validation"]["macro_f1"], res["test"]["macro_f1"], res["test"]["accuracy"])
        out["results"][name] = res
        out["models"][name] = model
        out[f"proba_test_{name}"] = proba
        if not keep_models and name != DEPLOYED:
            out["models"][name] = None
    out["results"]["isolation_forest"], out["iforest_scores"] = run_isolation_forest(X, y, n_jobs)
    return out


def run_isolation_forest(X: dict[str, np.ndarray], y: dict[str, np.ndarray], n_jobs: int) -> tuple[dict[str, Any], Any]:
    """Unsupervised baseline: fit on benign training flows only.

    The alert threshold is the 99th percentile of anomaly scores on *benign
    validation* flows (a 1% false-alarm budget); no attack labels are used to
    fit or tune it.
    """
    model = build("isolation_forest", SEED, n_jobs, len(CLASSES))
    benign = X["train"][y["train"] == BENIGN_IDX]
    t0 = time.perf_counter()
    model.fit(benign)
    fit_s = time.perf_counter() - t0
    score_val = -model.score_samples(X["val"])
    thr = float(np.quantile(score_val[y["val"] == BENIGN_IDX], 0.99))
    t0 = time.perf_counter()
    score = -model.score_samples(X["test"])
    pred_s = time.perf_counter() - t0
    flagged = score > thr
    is_attack = y["test"] != BENIGN_IDX
    # Context only: the threshold maximising F1 on validation (this one *does* use labels).
    from sklearn.metrics import precision_recall_curve

    pr, rc, th = precision_recall_curve(y["val"] != BENIGN_IDX, score_val)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    thr_f1 = float(th[np.argmax(f1[:-1])])
    per_class = {}
    for i, c in enumerate(CLASSES):
        m = y["test"] == i
        if m.any() and c != BENIGN:
            k = int(flagged[m].sum())
            lo, hi = ev.wilson_interval(k, int(m.sum()))
            per_class[c] = {"detection_rate": round(k / m.sum(), 6), "support": int(m.sum()), "ci95": [round(lo, 4), round(hi, 4)]}
    res = {
        "display_name": DISPLAY_NAMES["isolation_forest"],
        "kind": "anomaly_detection",
        "train_seconds": round(fit_s, 1),
        "trained_on": f"{len(benign)} benign training flows (labels of attacks unused)",
        "inference_us_per_flow": round(pred_s / len(score) * 1e6, 3),
        "model_size_bytes": model_size_bytes(model),
        "threshold": {"rule": "99th percentile of benign validation scores", "value": round(thr, 6)},
        "binary_test": ev.binary_report(is_attack, flagged, score),
        "binary_test_f1_threshold": {"threshold": round(thr_f1, 6),
                                     **ev.binary_report(is_attack, score > thr_f1, None)},
        "binary_roc": ev.binary_roc(is_attack, score),
        "per_class_detection": per_class,
    }
    log.info("[isolation_forest] ROC-AUC %.4f recall %.4f FPR %.4f", res["binary_test"]["roc_auc"],
             res["binary_test"]["recall"], res["binary_test"]["false_positive_rate"])
    return res, (model, thr)


# ---------------------------------------------------------------------- global SHAP


def global_shap(det: Detector, X_test: pd.DataFrame, y_test: np.ndarray, per_class: int = 250,
                beeswarm_points: int = 700, top: int = 12) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    idx = []
    for i in range(len(CLASSES)):
        cand = np.flatnonzero(y_test == i)
        idx.extend(rng.choice(cand, min(per_class, len(cand)), replace=False).tolist())
    idx = np.array(sorted(idx))
    Xs, ys = X_test.iloc[idx].reset_index(drop=True), y_test[idx]
    t0 = time.perf_counter()
    vals, base = det.shap_values(Xs)
    shap_s = time.perf_counter() - t0
    vals = vals / det.temperature
    base = base / det.temperature
    pred = softmax(base + vals.sum(1), axis=1).argmax(1)
    feats = det.features
    own = vals[np.arange(len(ys)), :, pred]  # predicted-class contributions
    overall = np.abs(own).mean(0)
    evidence = -vals[:, :, BENIGN_IDX]  # pushes away from BENIGN = attack evidence
    order = np.argsort(-overall)

    per_cls: dict[str, list[dict[str, Any]]] = {}
    matrix = []
    for i, c in enumerate(CLASSES):
        m = ys == i
        imp = np.abs(vals[m, :, i]).mean(0) if m.any() else np.zeros(len(feats))
        matrix.append(imp)
        per_cls[c] = [{"feature": feats[j], "mean_abs_shap": round(float(imp[j]), 5)} for j in np.argsort(-imp)[:top]]

    ev_order = np.argsort(-np.abs(evidence).mean(0))[:top]
    pts = rng.choice(len(ys), min(beeswarm_points, len(ys)), replace=False)
    ref = det.reference
    beeswarm = []
    for j in ev_order:
        f = feats[j]
        lo, hi = ref[f][0], ref[f][1]
        xv = Xs[f].to_numpy()
        scaled = np.clip((signed_log1p(xv) - signed_log1p(np.array(lo))) /
                         max(float(signed_log1p(np.array(hi)) - signed_log1p(np.array(lo))), 1e-9), 0, 1)
        beeswarm.append({
            "feature": f,
            "mean_abs": round(float(np.abs(evidence[:, j]).mean()), 5),
            "points": [[round(float(evidence[p, j]), 4), round(float(scaled[p]), 3), int(ys[p])] for p in pts],
        })

    top_feats = [feats[j] for j in order[:8]]
    profiles = {}
    for i, c in enumerate(CLASSES):
        m = y_test == i
        profiles[c] = {f: float(np.median(X_test.loc[m, f])) for f in top_feats}

    return {
        "sample": {"size": int(len(ys)), "per_class_cap": per_class, "seconds": round(shap_s, 2)},
        "units": "contribution to the temperature-scaled softmax logit",
        "importance": [{"feature": feats[j], "label": FEATURE_BY_NAME[feats[j]].cic2017,
                        "mean_abs_shap": round(float(overall[j]), 5)} for j in order],
        "per_class": per_cls,
        "class_feature_matrix": {"classes": list(CLASSES), "features": [feats[j] for j in order[:top]],
                                 "values": [[round(float(matrix[i][j]), 5) for j in order[:top]] for i in range(len(CLASSES))]},
        "beeswarm": beeswarm,
        "class_profiles": {"features": top_feats, "medians": profiles},
    }


# --------------------------------------------------------------------------- figures


def save_figures(metrics: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    dep = metrics["models"][DEPLOYED]["test"]
    cm = np.array(dep["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "#111")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("XGBoost — test confusion matrix (row-normalised colour)")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    for r in metrics["models"].values():
        pts = np.array(r["binary_roc"])
        auc = r["binary_test"].get("roc_auc", float("nan"))
        ax.plot(np.maximum(pts[:, 0], 1e-6), pts[:, 1], label=f"{r['display_name']} (AUC {auc:.4f})")
    ax.set_xscale("log"); ax.set_xlim(1e-5, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate (log)"); ax.set_ylabel("True positive rate")
    ax.set_title("Attack vs benign ROC — test split"); ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "roc_curves.png"); plt.close(fig)

    imp = metrics["explainability"]["importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh([d["label"] for d in imp], [d["mean_abs_shap"] for d in imp], color="#3b82f6")
    ax.set_xlabel("mean |SHAP| (predicted-class logit)"); ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout(); fig.savefig(out / "shap_importance.png"); plt.close(fig)


# ------------------------------------------------------------------------------ main


def summarise_protocol(res: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, r in res["results"].items():
        if "test" in r:
            out[name] = {
                "display_name": r["display_name"],
                "accuracy": r["test"]["accuracy"],
                "macro_f1": r["test"]["macro_f1"],
                "roc_auc_macro_ovr": r["test"].get("roc_auc_macro_ovr"),
                "per_class_recall": {c: v["recall"] for c, v in r["test"]["per_class"].items()},
                "per_class_f1": {c: v["f1"] for c, v in r["test"]["per_class"].items()},
                "binary": r["binary_test"],
            }
        else:
            out[name] = {"display_name": r["display_name"], "binary": r["binary_test"],
                         "per_class_detection": r["per_class_detection"]}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-temporal", action="store_true", help="skip the chronological-split protocol")
    ap.add_argument("--cross-dataset", action="store_true", help="run CSE-CIC-IDS2018 experiments (~2.3 GB download)")
    ap.add_argument("--models", default=",".join(SUPERVISED), help="comma-separated supervised models")
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    ap.add_argument("--sample", type=int, default=0, help="smoke run: at most N flows per class")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    started = time.perf_counter()
    models = tuple(m for m in args.models.split(",") if m)
    if DEPLOYED not in models:
        raise SystemExit("xgboost must be among the trained models (it is the deployed model)")

    frame, clean_stats = load_clean_2017()
    if args.sample:
        keep = frame.sample(frac=1.0, random_state=SEED).groupby(frame["label"].fillna("_"), sort=False).head(args.sample)
        frame = frame.loc[np.sort(keep.index)].reset_index(drop=True)
    probe = frame[frame["label"].isna()].reset_index(drop=True)  # Heartbleed: never trained on
    frame = frame[frame["label"].notna()].reset_index(drop=True)

    # ---------------------------------------------------------------- primary
    split = stratified_split(frame["label"], seed=SEED)
    primary = run_protocol(frame, split, models, args.jobs)
    sel, pre = primary["selection"], primary["preprocessor"]
    xgb_model = primary["models"][DEPLOYED]
    # Keep only the trees up to the early-stopping optimum: XGBClassifier.predict_proba
    # does this implicitly, a raw Booster would otherwise use every trained tree.
    booster = xgb_model.get_booster()[: xgb_model.best_iteration + 1]
    booster.set_param({"nthread": args.jobs})

    # Selection rule (validation only): best macro-F1; the deployed model must
    # also be serialisable without pickle and explainable with exact TreeSHAP.
    val_rank = sorted(((r["validation"]["macro_f1"], n) for n, r in primary["results"].items() if "validation" in r), reverse=True)

    # ------------------------------------------------------------ calibration
    X_val = pd.DataFrame(primary["X"]["val"], columns=sel.selected)
    X_test = pd.DataFrame(primary["X"]["test"], columns=sel.selected)
    y_val, y_test = primary["y"]["val"], primary["y"]["test"]
    dval = xgb.DMatrix(X_val.to_numpy(), feature_names=sel.selected)
    temperature = fit_temperature(booster.predict(dval, output_margin=True), y_val)
    proba_raw = primary[f"proba_test_{DEPLOYED}"]

    train_X = pd.DataFrame(primary["X"]["train"], columns=sel.selected)
    benign_tr = train_X[primary["y"]["train"] == BENIGN_IDX]
    reference = {f: [float(train_X[f].quantile(0.001)), float(train_X[f].quantile(0.999)), float(benign_tr[f].median())]
                 for f in sel.selected}
    meta = {
        "name": "SentinelAI XGBoost",
        "version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "classes": list(CLASSES),
        "class_info": CLASS_INFO,
        "preprocessor": pre.to_dict(),
        "temperature": temperature,
        "reference_ranges": reference,
        "training_data": "CIC-IDS2017 (stratified 70% train split after de-duplication)",
    }
    det = Detector(booster, meta)
    cal = det.predict(X_test)
    assert np.array_equal(cal.pred, proba_raw.argmax(1)), "temperature scaling must not change predictions"
    correct = cal.pred == y_test
    calibration = {
        "method": "temperature scaling fitted on the validation split (minimise NLL)",
        "temperature": round(temperature, 5),
        "test_nll_before": round(ev.nll(proba_raw, y_test), 6),
        "test_nll_after": round(ev.nll(cal.proba, y_test), 6),
        "before": ev.calibration_report(proba_raw.max(1), correct),
        "after": ev.calibration_report(cal.confidence, correct),
    }
    log.info("calibration: T=%.3f ECE %.5f -> %.5f", temperature, calibration["before"]["ece"], calibration["after"]["ece"])

    # unseen attack probe: Heartbleed was never in training
    probe_info: dict[str, Any] = {"note": "Heartbleed flows were excluded from training (11 raw flows)."}
    if len(probe):
        Xp = pre.transform(probe)
        pp = det.predict(Xp)
        if_model, if_thr = primary["iforest_scores"]
        if_flag = (-if_model.score_samples(Xp.to_numpy(dtype=np.float32))) > if_thr
        probe_info.update({
            "flows": int(len(probe)),
            "xgboost_predictions": pd.Series(pp.labels()).value_counts().to_dict(),
            "xgboost_flagged_as_attack": int((pp.pred != BENIGN_IDX).sum()),
            "isolation_forest_flagged": int(if_flag.sum()),
        })

    explain = global_shap(det, X_test, y_test)
    dep_res = primary["results"][DEPLOYED]
    dep_res["class_roc"] = ev.class_roc_curves(y_test, cal.proba, list(CLASSES))

    metrics: dict[str, Any] = {
        "project": "SentinelAI",
        "version": __version__,
        "generated_at": meta["trained_at"],
        "deployed_model": DEPLOYED,
        "selection": {
            "rule": ("Highest validation macro-F1. The deployed model must also be serialisable without pickle "
                     "and support exact TreeSHAP; the test split is used only for reporting."),
            "validation_ranking": [{"model": n, "macro_f1": f} for f, n in val_rank],
        },
        "dataset": {
            "name": "CIC-IDS2017",
            "citation": CICIDS2017_CITATION,
            "archive": {"url": CICIDS2017_ARCHIVE.url, "sha256": CICIDS2017_ARCHIVE.sha256},
            "cleaning": clean_stats,
            "split_protocol": "stratified random 70/15/15 after de-duplication (seed 42)",
            "splits": primary["split_counts"],
            "class_weights": {c: round(float(w), 3) for c, w in zip(CLASSES, primary["class_weights"])},
        },
        "features": {
            "selected": [{"name": f, "label": FEATURE_BY_NAME[f].cic2017, "label_2018": FEATURE_BY_NAME[f].cic2018,
                          "description": FEATURE_BY_NAME[f].description, "unit": FEATURE_BY_NAME[f].unit} for f in sel.selected],
            "dropped": sel.dropped,
        },
        "classes": [{"name": c, **CLASS_INFO[c]} for c in CLASSES],
        "models": primary["results"],
        "calibration": calibration,
        "unseen_attack_probe": probe_info,
        "explainability": explain,
        "experiments": {},
    }

    # ---------------------------------------------------------------- temporal
    if not args.skip_temporal:
        log.info("=== temporal protocol ===")
        tsplit = temporal_split(frame)
        temporal = run_protocol(frame, tsplit, models, args.jobs)
        metrics["experiments"]["temporal_split"] = {
            "description": ("Chronological split inside every (capture day, class) block: earliest 70% train, "
                            "next 15% validation, latest 15% test. Feature selection and preprocessing refitted."),
            "splits": temporal["split_counts"],
            "results": summarise_protocol(temporal),
        }
        del temporal

    # ----------------------------------------------------------- cross-dataset
    if args.cross_dataset:
        from sentinelai.training.crossdataset import run_cross_dataset

        metrics["experiments"]["cross_dataset"] = run_cross_dataset(frame, split, det, args.jobs)

    metrics["environment"] = {
        "python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgb.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
        "seed": SEED, "jobs": args.jobs, "total_runtime_minutes": round((time.perf_counter() - started) / 60, 1),
    }

    # ---------------------------------------------------------------- artifact
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    booster.set_param({"nthread": 1})
    booster_bytes = bytes(booster.save_raw("ubj"))
    meta["booster_sha256"] = sha256_bytes(booster_bytes)
    # model_size_bytes above is the model as trained (incl. trees past the early-stopping optimum).
    metrics["models"][DEPLOYED]["deployed_size_bytes"] = len(booster_bytes)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    meta_bytes = json.dumps(meta, indent=1).encode()
    (out / BOOSTER_FILE).write_bytes(booster_bytes)
    (out / META_FILE).write_bytes(meta_bytes)
    (out / CHECKSUM_FILE).write_text(f"{sha256_bytes(booster_bytes)}  {BOOSTER_FILE}\n{sha256_bytes(meta_bytes)}  {META_FILE}\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    save_figures(metrics, out / "figures")
    # Persist the test-split row ids so sample files can be drawn from unseen flows.
    if not args.sample:
        np.save(PROCESSED / "primary_split.npy", split.astype("U5"))
    log.info("done in %.1f min -> %s", (time.perf_counter() - started) / 60, out)


if __name__ == "__main__":
    main()
