"""Cross-dataset generalisation experiments with CSE-CIC-IDS2018.

Question: does a model trained on CIC-IDS2017 detect the same attack *types*
generated a year later on a different network (AWS testbed, newer
CICFlowMeter), and does adding 2018 data to training help?

A  2017 model (the deployed one)   -> all sampled 2018 flows
B  2018-only model                  -> 2018 test (in-domain) and 2017 test (reverse)
C  joint 2017 + 2018 model          -> 2017 test and 2018 test
D  ablation: 2017 model without the TCP-stack features (initial window sizes,
   minimum segment size), which describe the attacking *hosts* rather than
   the attack behaviour -> 2017 test and 2018 test

2018 flows are sampled (at most ``CAP`` per day x label), de-duplicated and
split chronologically within each (day, class) block. PortScan does not occur
in the 2018 files used. Macro averages are over classes present in each test set.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from sentinelai.data.loading import clean, load_cicids2018
from sentinelai.data.sources import BACKEND_ROOT, CICIDS2018_CITATION, cicids2018_csvs
from sentinelai.data.splitting import temporal_split
from sentinelai.features import Preprocessor, select_features
from sentinelai.inference.detector import Detector
from sentinelai.schema import BENIGN, CLASSES
from sentinelai.training import evaluate as ev
from sentinelai.training.models import build, class_weights

log = logging.getLogger("sentinelai.crossdataset")
CAP = 40_000
SEED = 42
BENIGN_IDX = CLASSES.index(BENIGN)
PROCESSED = BACKEND_ROOT / "data" / "processed"


def _encode(s: pd.Series) -> np.ndarray:
    return s.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)


def _report(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = proba.argmax(1)
    rep = ev.multiclass_report(y, pred, proba, list(CLASSES))
    present = [c for c, v in rep["per_class"].items() if v["support"] > 0]
    return {
        "accuracy": rep["accuracy"],
        "macro_f1_present_classes": round(float(np.mean([rep["per_class"][c]["f1"] for c in present])), 6),
        "per_class": {c: {k: rep["per_class"][c][k] for k in ("precision", "recall", "f1", "support")} for c in present},
        "binary": ev.binary_report(y != BENIGN_IDX, pred != BENIGN_IDX, 1 - proba[:, BENIGN_IDX]),
        "confusion_matrix": rep["confusion_matrix"],
    }


HOST_STACK_FEATURES = ("init_win_bytes_fwd", "init_win_bytes_bwd", "min_seg_size_fwd")


def _fit_xgb(train: pd.DataFrame, val: pd.DataFrame, jobs: int, exclude: tuple[str, ...] = ()) -> tuple[Any, Preprocessor]:
    """Same hyperparameters as the deployed model, via the native API (the
    scikit-learn wrapper rejects training sets with a missing class, and 2018
    has no PortScan)."""
    sel = select_features(train.drop(columns=list(exclude)), seed=SEED)
    pre = Preprocessor(sel.selected).fit(train)
    Xtr, Xva = pre.transform(train).to_numpy(np.float32), pre.transform(val).to_numpy(np.float32)
    ytr, yva = _encode(train["label"]), _encode(val["label"])
    cw = class_weights(ytr, len(CLASSES))
    ref = build("xgboost", SEED, jobs, len(CLASSES))
    params = {k: v for k, v in ref.get_xgb_params().items() if v is not None and k not in ("n_estimators",)}
    params.update({"num_class": len(CLASSES), "nthread": jobs, "seed": SEED})
    params.pop("n_jobs", None), params.pop("random_state", None)
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=cw[ytr])
    dva = xgb.DMatrix(Xva, label=yva, weight=cw[yva])
    booster = xgb.train(params, dtr, num_boost_round=ref.n_estimators, evals=[(dva, "val")],
                        early_stopping_rounds=ref.early_stopping_rounds, verbose_eval=False)
    return booster[: booster.best_iteration + 1], pre


def _proba(model: Any, pre: Preprocessor, df: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(pre.transform(df).to_numpy(np.float32)))


def _load_2018() -> pd.DataFrame:
    cache = PROCESSED / "cicids2018_sample_clean.parquet"
    stats = PROCESSED / "cicids2018_sample_clean_stats.json"
    if cache.exists() and stats.exists():
        return pd.read_parquet(cache), json.loads(stats.read_text())
    fd = clean(load_cicids2018(cicids2018_csvs(), per_class_cap=CAP, seed=SEED))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fd.frame.to_parquet(cache, index=False)
    stats.write_text(json.dumps(fd.stats, default=int))
    return fd.frame, fd.stats


def run_cross_dataset(frame17: pd.DataFrame, split17: np.ndarray, det: Detector, jobs: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    frame18, stats18 = _load_2018()
    d18 = frame18[frame18["label"].notna()].reset_index(drop=True)
    s18 = temporal_split(d18)
    tr18, va18, te18 = (d18[s18 == k] for k in ("train", "val", "test"))
    tr17, va17, te17 = (frame17[split17 == k] for k in ("train", "val", "test"))
    y18, y17 = _encode(te18["label"]), _encode(te17["label"])
    log.info("2018: %d flows after cleaning (%s)", len(d18), d18["label"].value_counts().to_dict())

    # A: deployed 2017 model on 2018 (all sampled flows, and on the 2018 test split for comparability)
    X18_all = det.preprocessor.transform(d18)
    pa = det.predict(X18_all)
    exp_a_all = _report(_encode(d18["label"]), pa.proba)
    exp_a_all["distribution_shift"] = det.distribution_shift(X18_all)
    exp_a_test = _report(y18, det.predict(det.preprocessor.transform(te18)).proba)
    exp_17_test = _report(y17, det.predict(det.preprocessor.transform(te17)).proba)

    # B: 2018-only
    m18, p18 = _fit_xgb(tr18, va18, jobs)
    b_in = _report(y18, _proba(m18, p18, te18))
    b_rev = _report(y17, _proba(m18, p18, te17))

    # C: joint
    mj, pj = _fit_xgb(pd.concat([tr17, tr18]), pd.concat([va17, va18]), jobs)
    c18 = _report(y18, _proba(mj, pj, te18))
    c17 = _report(y17, _proba(mj, pj, te17))

    # D: ablation without host TCP-stack features
    ma, pa_ = _fit_xgb(tr17, va17, jobs, exclude=HOST_STACK_FEATURES)
    abl17 = _report(y17, _proba(ma, pa_, te17))
    abl18 = _report(y18, _proba(ma, pa_, te18))

    def head(r: dict[str, Any]) -> dict[str, Any]:
        return {"macro_f1": r["macro_f1_present_classes"], "attack_recall": r["binary"]["recall"],
                "false_positive_rate": r["binary"]["false_positive_rate"],
                "per_class_recall": {c: v["recall"] for c, v in r["per_class"].items()}}

    log.info("cross-dataset done in %.0fs", time.perf_counter() - t0)
    return {
        "description": __doc__.strip().split("\n\n")[0],
        "dataset": {"name": "CSE-CIC-IDS2018", "citation": CICIDS2018_CITATION,
                    "sampling": f"at most {CAP} flows per (day, label), seed {SEED}, then de-duplicated",
                    "cleaning": {k: v for k, v in stats18.items() if k != "excluded_labels"},
                    "splits": {k: d18[s18 == k]["label"].value_counts().to_dict() for k in ("train", "val", "test")}},
        "summary": {
            "test_2017": {"trained_2017": head(exp_17_test), "trained_2018": head(b_rev), "trained_joint": head(c17),
                          "trained_2017_no_host_stack": head(abl17)},
            "test_2018": {"trained_2017": head(exp_a_test), "trained_2018": head(b_in), "trained_joint": head(c18),
                          "trained_2017_no_host_stack": head(abl18)},
        },
        "ablation_excluded_features": list(HOST_STACK_FEATURES),
        "A_2017_model_on_all_2018": exp_a_all,
        "seconds": round(time.perf_counter() - t0, 1),
    }
