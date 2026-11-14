"""Dataset acquisition, cleaning and leakage-safe splitting.

Datasets (both CC BY 4.0, downloaded at build time, never committed):

* **PhiUSIIL Phishing URL Dataset** - Prasad & Chandra (2024), UCI ML Repository
  id 967, DOI 10.24432/C5C051. 235,795 URLs.
* **Web page phishing detection dataset** - Hannousse & Yahiouche (2021),
  Mendeley Data, DOI 10.17632/c2gw7fy2j4.3. 11,430 URLs.

Plus one derived corpus built by ``scripts/build_legit_corpus.py`` and committed
to the repository (``datasets/legit_deep_urls.csv``):

* **Legitimate deep URLs** - 16k crawled page URLs (with real paths) from
  popular registrable domains: pinned Tranco list ``Y897G`` x Common Crawl
  ``CC-MAIN-2026-39`` index. Corrects the "legitimate == bare home page" bias
  of the two public datasets.

Only the raw URL string and the label are used. PhiUSIIL also ships ~50
pre-computed columns, some derived from the fetched HTML and some (e.g.
``URLSimilarityIndex``) computed with knowledge of the legitimate set; those are
deliberately ignored because they are not available at inference time and leak
label information.
"""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from phishlens.features import ALL_FEATURES, InvalidURLError, extract_features, parse_url

log = logging.getLogger(__name__)

RANDOM_STATE = 42
BACKEND_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    sha256: str
    filename: str
    license: str
    citation: str
    local_path: Path | None = None  # committed file instead of a download


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="phiusiil",
        url="https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
        sha256="0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a",
        filename="phiusiil.zip",
        license="CC BY 4.0",
        citation=(
            "Prasad, A. & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered "
            "phishing URL detection framework based on similarity index and incremental learning. "
            "Computers & Security, 136. UCI ML Repository, https://doi.org/10.24432/C5C051"
        ),
    ),
    DatasetSpec(
        name="hannousse",
        url=(
            "https://data.mendeley.com/public-files/datasets/c2gw7fy2j4/files/"
            "575316f4-ee1d-453e-a04f-7b950915b61b/file_downloaded"
        ),
        sha256="21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c",
        filename="hannousse_dataset_B_05_2020.csv",
        license="CC BY 4.0",
        citation=(
            "Hannousse, A. & Yahiouche, S. (2021). Towards benchmark datasets for machine "
            "learning based website phishing detection: An experimental study. Engineering "
            "Applications of Artificial Intelligence, 104. Mendeley Data, "
            "https://doi.org/10.17632/c2gw7fy2j4.3"
        ),
    ),
    DatasetSpec(
        name="commoncrawl_legit",
        url="https://tranco-list.eu/list/Y897G + https://data.commoncrawl.org/cc-index/collections/CC-MAIN-2026-39/",
        sha256="138bca3fac000c4c5dca13d2381f086d38af92d82740b0536418d81440055084",
        filename="legit_deep_urls.csv",
        license="Derived from Tranco (free, cite Le Pochat et al. NDSS 2019) and Common Crawl (Terms of Use)",
        citation=(
            "Built by scripts/build_legit_corpus.py from the Tranco list Y897G "
            "(Le Pochat et al., 'Tranco: A Research-Oriented Top Sites Ranking Hardened Against "
            "Manipulation', NDSS 2019) and the Common Crawl CC-MAIN-2026-39 URL index."
        ),
        local_path=BACKEND_ROOT / "datasets" / "legit_deep_urls.csv",
    ),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_datasets(data_dir: Path, force: bool = False) -> dict[str, Path]:
    """Download every dataset into ``data_dir`` and verify its SHA-256."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for spec in DATASETS:
        dest = spec.local_path or data_dir / spec.filename
        if spec.local_path is None and (force or not dest.exists()):
            log.info("Downloading %s from %s", spec.name, spec.url)
            req = urllib.request.Request(spec.url, headers={"User-Agent": "phishlens-dataset-fetch/1.0"})  # noqa: S310
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed https URLs
                dest.write_bytes(resp.read())
        digest = sha256_file(dest)
        if digest != spec.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {spec.name}: expected {spec.sha256}, got {digest}. "
                "The upstream file changed or the download is corrupt."
            )
        paths[spec.name] = dest
    return paths


def load_raw(paths: dict[str, Path]) -> pd.DataFrame:
    """Return a frame with columns ``url``, ``label`` (1 = phishing), ``source``."""
    with zipfile.ZipFile(paths["phiusiil"]) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(member) as fh:
            phi = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), usecols=["URL", "label"])
    # PhiUSIIL convention: label 1 = legitimate, 0 = phishing.
    phi = pd.DataFrame({"url": phi["URL"], "label": 1 - phi["label"].astype(int), "source": "phiusiil"})

    han = pd.read_csv(paths["hannousse"], usecols=["url", "status"])
    han = pd.DataFrame(
        {"url": han["url"], "label": (han["status"] == "phishing").astype(int), "source": "hannousse"}
    )
    cc = pd.read_csv(paths["commoncrawl_legit"], usecols=["url"])
    cc = pd.DataFrame({"url": cc["url"], "label": 0, "source": "commoncrawl_legit"})
    return pd.concat([phi, han, cc], ignore_index=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse, canonicalise and de-duplicate URLs.

    Duplicates are removed on the *canonical* URL so that trivial variants
    (``http://`` vs ``https://``, ``www.``, trailing ``/``) cannot appear on
    both sides of the train/test split. Canonical URLs that occur with
    contradicting labels are dropped entirely.
    """
    report: dict[str, int] = {"raw_rows": len(df)}
    df = df.dropna(subset=["url"]).copy()
    df["url"] = df["url"].astype(str).str.strip()

    canon, groups, ok = [], [], []
    for u in df["url"]:
        try:
            p = parse_url(u)
            canon.append(p.canonical)
            groups.append(p.group_domain)
            ok.append(True)
        except InvalidURLError:
            canon.append(None)
            groups.append(None)
            ok.append(False)
    df["canonical"] = canon
    df["group"] = groups
    report["unparseable"] = int((~np.array(ok)).sum())
    df = df[np.array(ok)]

    label_nunique = df.groupby("canonical")["label"].transform("nunique")
    report["conflicting_label_rows"] = int((label_nunique > 1).sum())
    df = df[label_nunique == 1]

    before = len(df)
    df = df.drop_duplicates(subset=["canonical"], keep="first")
    report["duplicate_rows"] = before - len(df)
    report["clean_rows"] = len(df)
    report["clean_phishing"] = int(df["label"].sum())
    report["clean_legitimate"] = int((df["label"] == 0).sum())
    return df.reset_index(drop=True), report


def featurize(urls: pd.Series | list[str]) -> pd.DataFrame:
    """Vectorise URLs into the feature frame consumed by the model pipeline."""
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame.from_records(rows, columns=list(ALL_FEATURES))


def grouped_split(
    df: pd.DataFrame, test_size_folds: int = 5, val_folds: int = 5, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / validation / test indices, stratified by label and grouped by domain.

    All URLs that share a registrable domain (PSL-aware, so each
    ``*.firebaseapp.com`` tenant is its own group) land in the same split. This
    prevents the model from being scored on domains it memorised in training,
    which would inflate every metric.

    Proportions: test = 1/test_size_folds (20%), validation = 1/val_folds of the
    remainder (16%), train = the rest (64%).
    """
    strat = df["label"].astype(str) + "_" + df["source"]
    outer = StratifiedGroupKFold(n_splits=test_size_folds, shuffle=True, random_state=random_state)
    trainval_idx, test_idx = next(outer.split(df, strat, groups=df["group"]))

    tv = df.iloc[trainval_idx]
    inner = StratifiedGroupKFold(n_splits=val_folds, shuffle=True, random_state=random_state)
    tr_rel, val_rel = next(inner.split(tv, strat.iloc[trainval_idx], groups=tv["group"]))
    return trainval_idx[tr_rel], trainval_idx[val_rel], test_idx


def source_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """Sample weights that give every (source, label) cell equal total weight.

    PhiUSIIL contributes ~130k legitimate URLs that are all bare home pages,
    while the Hannousse and Common Crawl sets contribute ~22k legitimate URLs
    with realistic paths. Without re-weighting the much larger PhiUSIIL cells
    dominate and the model learns "has a path => phishing".
    """
    cell = df["source"] + "_" + df["label"].astype(str)
    counts = cell.map(cell.value_counts())
    w = len(df) / (cell.nunique() * counts)
    return w.to_numpy(dtype=float)
