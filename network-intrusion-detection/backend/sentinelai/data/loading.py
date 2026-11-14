"""Load, verify and clean the CIC-IDS2017 / CSE-CIC-IDS2018 flow exports."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinelai.data.sources import EXPECTED_CICIDS2017_COUNTS
from sentinelai.features import canonicalize, to_numeric
from sentinelai.schema import FEATURE_NAMES, map_label

log = logging.getLogger(__name__)


@dataclass
class FlowData:
    """Cleaned flows with canonical feature columns plus bookkeeping columns.

    ``frame`` columns: canonical features, ``raw_label``, ``label`` (taxonomy
    class or NA for excluded labels), ``day`` (source file stem) and
    ``time_key`` (minutes since midnight, used by the temporal split).
    """

    frame: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)


def _clean_raw_label(s: pd.Series) -> pd.Series:
    # "Web Attack � Brute Force" -> "Web Attack Brute Force"
    return s.astype(str).str.replace(r"[^\x20-\x7e]+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


_TS_2017 = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s+(\d{1,2}):(\d{2})(?::(\d{2}))?")


def parse_time_2017(ts: pd.Series) -> pd.Series:
    """Minutes since midnight from CIC-IDS2017 timestamps.

    The export uses a 12-hour clock without AM/PM (``5/7/2017 2:43``). Capture
    ran during working hours (~08:40-17:10), so hours 1-7 are afternoon.
    """
    parts = ts.astype(str).str.extract(_TS_2017)
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce")
    sec = pd.to_numeric(parts[2], errors="coerce").fillna(0)
    hour = hour.where(hour >= 8, hour + 12)
    return hour * 60 + minute + sec / 60


def parse_time_2018(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    return t.dt.hour * 60 + t.dt.minute + t.dt.second / 60


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", dtype=str, engine="pyarrow")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _assemble(df: pd.DataFrame, day: str, time_key: pd.Series) -> pd.DataFrame:
    feats, labels, _ = canonicalize(df)
    if labels is None:
        raise ValueError(f"{day}: no Label column")
    out = to_numeric(feats).reindex(columns=list(FEATURE_NAMES))
    out["raw_label"] = _clean_raw_label(labels).to_numpy()
    out["day"] = day
    out["time_key"] = time_key.to_numpy()
    return out


def load_cicids2017(csvs: list[Path], verify_counts: bool = True) -> pd.DataFrame:
    frames = []
    for path in csvs:
        df = _read_csv(path)
        df = df[df["Label"].notna() & (df["Label"].astype(str).str.strip() != "")]  # trailing empty rows
        frames.append(_assemble(df, path.name.split(".")[0], parse_time_2017(df["Timestamp"])))
        log.info("loaded %s: %d rows", path.name, len(df))
    data = pd.concat(frames, ignore_index=True)
    if verify_counts:
        got = data["raw_label"].value_counts().to_dict()
        if got != EXPECTED_CICIDS2017_COUNTS:
            raise ValueError(f"CIC-IDS2017 label counts differ from the published ones: {got}")
    return data


def load_cicids2018(csvs: list[Path], per_class_cap: int, seed: int = 42) -> pd.DataFrame:
    """Load the 2018 day files, sampling at most ``per_class_cap`` flows per (day, label)."""
    frames = []
    rng = np.random.default_rng(seed)
    for path in csvs:
        df = _read_csv(path)
        df = df[df["Label"].notna() & (df["Label"] != "Label")]  # repeated header rows
        parts = []
        for _, g in df.groupby("Label", sort=True):
            if len(g) > per_class_cap:
                g = g.iloc[np.sort(rng.choice(len(g), per_class_cap, replace=False))]
            parts.append(g)
        df = pd.concat(parts)
        frames.append(_assemble(df, path.name.split("_")[0], parse_time_2018(df["Timestamp"])))
        log.info("loaded %s: %d sampled rows", path.name, len(df))
    return pd.concat(frames, ignore_index=True)


def clean(data: pd.DataFrame) -> FlowData:
    """Map labels, then remove duplicate and contradictory flows.

    De-duplication happens *before* any split: identical feature vectors in
    train and test would otherwise inflate every metric. Feature vectors that
    occur with more than one label cannot be classified correctly by any model
    and are removed entirely.
    """
    stats: dict[str, Any] = {"raw_rows": len(data)}
    data = data.copy()
    data["label"] = data["raw_label"].map(map_label)
    excluded = data["label"].isna()
    stats["excluded_labels"] = data.loc[excluded, "raw_label"].value_counts().to_dict()

    feats = list(FEATURE_NAMES)
    vals = data[feats]
    stats["cells_nan"] = int(vals.isna().sum().sum())
    stats["cells_inf"] = int(np.isinf(vals.to_numpy()).sum())

    h = pd.util.hash_pandas_object(vals, index=False)
    lab = data["label"].fillna("__excluded__")
    key = pd.DataFrame({"h": h.to_numpy(), "y": lab.to_numpy()})
    n_labels = key.drop_duplicates().groupby("h")["y"].nunique()
    conflicting = key["h"].isin(n_labels.index[n_labels > 1]).to_numpy()
    stats["conflicting_rows_removed"] = int(conflicting.sum())
    stats["conflicting_vectors"] = int((n_labels > 1).sum())

    dup = key.duplicated().to_numpy() & ~conflicting
    stats["duplicate_rows_removed"] = int(dup.sum())
    data = data.loc[~dup & ~conflicting].reset_index(drop=True)
    stats["clean_rows"] = len(data)
    stats["class_counts"] = data["label"].value_counts().to_dict()
    return FlowData(frame=data, stats=stats)
