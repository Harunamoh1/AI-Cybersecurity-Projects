"""Turn a table of flows into stored predictions and an analysis summary."""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from sentinelai.inference.detector import Detector, confidence_band
from sentinelai.schema import BENIGN, CLASSES, SEVERITY_ORDER

log = logging.getLogger("sentinelai.api")


def analyse_frame(det: Detector, df: pd.DataFrame) -> dict[str, Any]:
    """Run the detector over a raw flow table (any supported column naming).

    Returns ``{"flows", "features", "summary", "rejected_rows", "duration_ms"}``.
    Raises :class:`sentinelai.features.SchemaError` on unusable input.
    """
    t0 = time.perf_counter()
    prep = det.prepare(df)
    n = len(prep.X)
    if n == 0:
        from sentinelai.features import SchemaError

        raise SchemaError("No row contained enough valid numeric flow features")
    preds = det.predict(prep.X)
    labels = preds.labels()
    severity = preds.severity()
    threat_idx = np.flatnonzero(preds.pred != det.benign_index)
    reasons: dict[int, list[dict[str, Any]]] = {}
    if len(threat_idx):
        sub = prep.X.iloc[threat_idx]
        for i, r in zip(threat_idx, det.top_reasons(sub, preds.pred[threat_idx], k=3)):
            reasons[int(i)] = r

    true = prep.labels
    flows = []
    for i in range(n):
        flows.append({
            "index": int(prep.row_index[i]),
            "predicted_class": labels[i],
            "confidence": round(float(preds.confidence[i]), 6),
            "attack_probability": round(float(preds.attack_probability[i]), 6),
            "severity": severity[i],
            "true_label": None if true is None or pd.isna(true.iloc[i]) else str(true.iloc[i]),
            "imputed_features": int(prep.imputed_cells[i]),
            "reasons": reasons.get(i, []),
        })

    counts = Counter(labels)
    sev_counts = Counter(s for s, lab in zip(severity, labels) if lab != BENIGN)
    summary: dict[str, Any] = {
        "class_counts": {c: counts.get(c, 0) for c in CLASSES},
        "severity_counts": {s: sev_counts.get(s, 0) for s in SEVERITY_ORDER[1:]},
        "confidence_bands": dict(Counter(confidence_band(float(c)) for c in preds.confidence)),
        "mean_confidence": round(float(preds.confidence.mean()), 6),
        "threats": int(len(threat_idx)),
        "flows_with_imputed_values": int((prep.imputed_cells > 0).sum()),
        "columns": prep.columns.as_dict(),
        "distribution_shift": det.distribution_shift(prep.X),
    }
    if true is not None and true.notna().any():
        m = true.notna().to_numpy()
        t = true[m].to_numpy()
        p = np.array(labels)[m]
        attack_t, attack_p = t != BENIGN, p != BENIGN
        tp = int((attack_t & attack_p).sum())
        summary["ground_truth"] = {
            "labelled_flows": int(m.sum()),
            "unmapped_labels": int((~m).sum()),
            "accuracy": round(float((t == p).mean()), 6),
            "attack_recall": round(tp / attack_t.sum(), 6) if attack_t.any() else None,
            "attack_precision": round(tp / attack_p.sum(), 6) if attack_p.any() else None,
            "false_positive_rate": round(float((~attack_t & attack_p).sum() / max((~attack_t).sum(), 1)), 6),
            "per_class_recall": {c: round(float((p[t == c] == c).mean()), 6) for c in CLASSES if (t == c).any()},
        }
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info("analysed %d flows (%d rejected): %d threats in %.0f ms", n, len(prep.rejected_rows), len(threat_idx), duration_ms)
    return {
        "flows": flows,
        "features": prep.X.to_numpy(dtype=np.float32),
        "summary": summary,
        "rejected_rows": prep.rejected_rows,
        "duration_ms": duration_ms,
    }
