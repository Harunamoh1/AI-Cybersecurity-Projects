"""Build the demo CSV files in ``samples/`` from held-out test flows.

    python scripts/make_samples.py      # after python -m sentinelai.training.train

Every CIC-IDS2017 row comes from the *test* split of the primary protocol, so
uploading a sample never shows the model a flow it was trained on. Files are
written in the original CICFlowMeter layout (all 78 feature columns plus
``Label``), exactly like a real export. One file uses the CSE-CIC-IDS2018
naming convention and 2018 traffic to demonstrate cross-dataset behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentinelai.data.splitting import temporal_split  # noqa: E402
from sentinelai.schema import FEATURES  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "samples"
SEED = 7
INT_LIKE = {f.name for f in FEATURES if not any(k in f.name for k in ("mean", "std", "var", "per_s", "ratio", "avg", "bulk_rate"))}


def export(df: pd.DataFrame, path: Path, naming: str) -> None:
    out = pd.DataFrame()
    for f in FEATURES:
        col = df[f.name]
        name = f.cic2017 if naming == "2017" else f.cic2018
        if f.name in INT_LIKE and col.dropna().apply(lambda v: np.isinf(v) or float(v).is_integer()).all():
            out[name] = col.map(lambda v: "" if pd.isna(v) else ("Infinity" if np.isinf(v) else str(int(v))))
        else:
            out[name] = col.map(lambda v: "" if pd.isna(v) else ("Infinity" if v == np.inf else repr(float(v))))
    out["Label"] = df["raw_label"].to_numpy()
    out = out.sample(frac=1.0, random_state=SEED)  # interleave classes like real traffic
    out.to_csv(path, index=False)


def take(df: pd.DataFrame, mask: pd.Series, n: int) -> pd.DataFrame:
    sub = df[mask]
    return sub.sample(min(n, len(sub)), random_state=SEED)


def main() -> None:
    frame = pd.read_parquet(PROCESSED / "cicids2017_clean.parquet")
    frame = frame[frame["label"].notna()].reset_index(drop=True)
    split = np.load(PROCESSED / "primary_split.npy")
    if len(split) != len(frame):
        raise SystemExit("split file does not match the cleaned data; re-run training")
    test = frame[split == "test"]
    OUT.mkdir(exist_ok=True)
    manifest: dict[str, dict] = {}

    parts = [take(test, test["label"] == "BENIGN", 3000)]
    for c in ("DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration"):
        parts.append(take(test, test["label"] == c, 150))
    mixed = pd.concat(parts)
    export(mixed, OUT / "mixed_traffic.csv", "2017")
    manifest["mixed_traffic.csv"] = {"title": "Mixed traffic (all attack classes)",
                                     "description": "3,000 benign flows plus up to 150 flows of every attack class, CIC-IDS2017 test split."}

    fri = test["day"].str.startswith("Friday-WorkingHours-Afternoon-DDos")
    ddos = pd.concat([take(test, fri & (test["label"] == "DDoS"), 1500), take(test, fri & (test["label"] == "BENIGN"), 700)])
    export(ddos, OUT / "ddos_friday_afternoon.csv", "2017")
    manifest["ddos_friday_afternoon.csv"] = {"title": "DDoS — Friday afternoon",
                                             "description": "LOIC HTTP flood mixed with concurrent benign traffic."}

    thu = test["day"].str.startswith("Thursday-WorkingHours-Morning-WebAttacks")
    web = pd.concat([take(test, thu & (test["label"] == "WebAttack"), 400), take(test, thu & (test["label"] == "BENIGN"), 1200)])
    export(web, OUT / "web_attacks_thursday.csv", "2017")
    manifest["web_attacks_thursday.csv"] = {"title": "Web attacks — Thursday morning",
                                            "description": "Brute force, XSS and SQL injection against a web application."}

    mon = test["day"].str.startswith("Monday")
    export(take(test, mon, 2000), OUT / "benign_monday.csv", "2017")
    manifest["benign_monday.csv"] = {"title": "Benign baseline — Monday",
                                     "description": "Attack-free working-day traffic; any alert here is a false positive."}

    f18 = PROCESSED / "cicids2018_sample_clean.parquet"
    if f18.exists():
        d18 = pd.read_parquet(f18)
        d18 = d18[d18["label"].notna()].reset_index(drop=True)
        t18 = d18[temporal_split(d18) == "test"]
        parts = [take(t18, t18["label"] == "BENIGN", 1000)]
        for c in ("DoS", "DDoS", "BruteForce", "WebAttack", "Bot", "Infiltration"):
            parts.append(take(t18, t18["label"] == c, 150))
        export(pd.concat(parts), OUT / "cicids2018_cross_dataset.csv", "2018")
        manifest["cicids2018_cross_dataset.csv"] = {
            "title": "CSE-CIC-IDS2018 (different network)",
            "description": ("2018 flows in the 2018 column naming. Expect a distribution-shift warning and much lower "
                            "accuracy: the model was trained on 2017 traffic only.")}

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for p in sorted(OUT.glob("*.csv")):
        print(f"{p.name}: {sum(1 for _ in p.open()) - 1} rows, {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
