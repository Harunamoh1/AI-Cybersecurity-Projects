"""Train / validation / test splits.

* ``stratified_split`` — the primary protocol: stratified random 70/15/15 on the
  de-duplicated data (the standard protocol in the literature, made honest by
  de-duplicating first).
* ``temporal_split`` — robustness protocol: within every (capture day, class)
  block, the earliest 70% of flows train, the next 15% validate and the latest
  15% test. The model is always evaluated on traffic that happened *after*
  everything it was trained on, which removes the optimism of interleaving
  near-identical consecutive flows of the same attack run across splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = ("train", "val", "test")


def stratified_split(labels: pd.Series, val: float = 0.15, test: float = 0.15, seed: int = 42) -> np.ndarray:
    idx = np.arange(len(labels))
    y = labels.to_numpy()
    train_idx, rest = train_test_split(idx, test_size=val + test, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(rest, test_size=test / (val + test), stratify=y[rest], random_state=seed)
    out = np.empty(len(labels), dtype=object)
    out[train_idx], out[val_idx], out[test_idx] = "train", "val", "test"
    return out


def temporal_split(frame: pd.DataFrame, val: float = 0.15, test: float = 0.15) -> np.ndarray:
    """Chronological split within each (day, label) block. Requires ``day``, ``label``, ``time_key``."""
    out = np.empty(len(frame), dtype=object)
    order = frame.assign(_pos=np.arange(len(frame))).sort_values(["day", "label", "time_key", "_pos"], kind="stable")
    for _, g in order.groupby(["day", "label"], sort=False):
        n = len(g)
        n_test = max(1, int(round(n * test))) if n >= 3 else 0
        n_val = max(1, int(round(n * val))) if n >= 3 else 0
        n_train = n - n_val - n_test
        pos = g["_pos"].to_numpy()
        out[pos[:n_train]] = "train"
        out[pos[n_train:n_train + n_val]] = "val"
        out[pos[n_train + n_val:]] = "test"
    return out
