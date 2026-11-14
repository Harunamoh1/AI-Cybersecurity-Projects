"""Dataset sources, pinned by SHA-256, and a checksum-verifying downloader.

Raw data is downloaded on demand into ``backend/data/raw`` and never committed.

* **CIC-IDS2017** — Sharafaldin, Habibi Lashkari & Ghorbani (2018), Canadian
  Institute for Cybersecurity, University of New Brunswick. The official UNB
  download is now behind a registration form, so the pipeline fetches the
  original ``GeneratedLabelledFlows.zip`` archive from a public Hugging Face
  mirror pinned to an immutable commit. Its authenticity is verified twice:
  by the SHA-256 below and by checking that the per-label row counts equal the
  counts published for the dataset (see ``EXPECTED_CICIDS2017_COUNTS``).
* **CSE-CIC-IDS2018** — CSE & CIC (2018), public AWS Open Data bucket
  ``s3://cse-cic-ids2018``. Used only for the cross-dataset generalisation
  experiment; seven daily files whose attacks overlap the CIC-IDS2017 taxonomy.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = BACKEND_ROOT / "data" / "raw"

CICIDS2017_CITATION = (
    "Iman Sharafaldin, Arash Habibi Lashkari and Ali A. Ghorbani, \"Toward Generating a New Intrusion "
    "Detection Dataset and Intrusion Traffic Characterization\", ICISSP 2018, pp. 108-116."
)
CICIDS2018_CITATION = (
    "Communications Security Establishment (CSE) & Canadian Institute for Cybersecurity (CIC), "
    "CSE-CIC-IDS2018 on AWS, https://registry.opendata.aws/cse-cic-ids2018/"
)


@dataclass(frozen=True)
class RemoteFile:
    url: str
    sha256: str
    filename: str
    size: int


_HF = "https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/811c007f08d693a8a8b4226c197c138391806ce6"
CICIDS2017_ARCHIVE = RemoteFile(
    url=f"{_HF}/csvs/GeneratedLabelledFlows.zip",
    sha256="7bdbef286f8893f31c6db12105fa097fa5c2dcc6733179037a08129d150ea27a",
    filename="GeneratedLabelledFlows.zip",
    size=283_876_488,
)

# Per-label row counts of the eight CIC-IDS2017 CSVs as published by CIC and
# widely reproduced in the literature (2,830,743 flows in total).
EXPECTED_CICIDS2017_COUNTS: dict[str, int] = {
    "BENIGN": 2_273_097,
    "DoS Hulk": 231_073,
    "PortScan": 158_930,
    "DDoS": 128_027,
    "DoS GoldenEye": 10_293,
    "FTP-Patator": 7_938,
    "SSH-Patator": 5_897,
    "DoS slowloris": 5_796,
    "DoS Slowhttptest": 5_499,
    "Bot": 1_966,
    "Web Attack Brute Force": 1_507,
    "Web Attack XSS": 652,
    "Infiltration": 36,
    "Web Attack Sql Injection": 21,
    "Heartbleed": 11,
}

_S3 = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
CICIDS2018_FILES: tuple[RemoteFile, ...] = tuple(
    RemoteFile(url=f"{_S3}/{name}", sha256=sha, filename=name, size=size)
    for name, sha, size in (
        ("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
         "acff8bc61376ee031d80878ee6099e0b1a87a1bd711d8068298421418c9f8147", 358_223_333),
        ("Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
         "fa2947a8256d81ee9103ae16139d62d0e17aa23e696ee80d9e76fb51c01c9c4b", 375_945_899),
        ("Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
         "1a4919faa0c49c7af97230b0c2d076eba23ee6dd81103a3801d51ac316355d8b", 333_723_605),
        ("Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
         "a5f4a1c2689e0aa6566c03a58466de9c407c0be0cbd3cc69306544026611be04", 328_893_673),
        ("Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
         "d0a7f5059d9823b6e9b392b759e306481a3502d190dea7a1b5502ae079ea069b", 382_840_456),
        ("Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
         "f15e2a12304446058a0186c8ad67de2bd15735a9ba5c70c9a1f4c4242ab06771", 209_249_758),
        ("Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
         "d96f38e7496aba83475031e6fb8c6fdf1abf6aa1b71325a917798f3c7de93de1", 352_368_373),
    )
)


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(remote: RemoteFile, dest_dir: Path) -> Path:
    """Download ``remote`` into ``dest_dir`` (if absent) and verify its SHA-256."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / remote.filename
    if not path.exists():
        if not remote.url.startswith("https://"):
            raise ValueError("Only HTTPS sources are allowed")
        log.info("Downloading %s (%.0f MB)", remote.filename, remote.size / 1e6)
        tmp = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(remote.url, timeout=120) as resp, tmp.open("wb") as out:  # noqa: S310 - https only
            shutil.copyfileobj(resp, out, length=1 << 20)
        tmp.rename(path)
    digest = sha256_file(path)
    if digest != remote.sha256:
        raise ChecksumError(f"{path.name}: expected sha256 {remote.sha256}, got {digest}")
    return path


def extract_csvs(archive: Path, out: Path) -> list[Path]:
    """Extract only the CSV members of ``archive`` into ``out``, flattened.

    Member paths are reduced to their basename, so entries such as
    ``../../etc/x.csv`` (zip-slip) cannot escape ``out``.
    """
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename.replace("\\", "/")).name
            if member.is_dir() or not name.lower().endswith(".csv") or name.startswith("."):
                continue
            with zf.open(member) as src, (out / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return sorted(out.glob("*.csv"))


def cicids2017_csvs(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return the eight verified CIC-IDS2017 day CSVs, downloading/extracting as needed."""
    base = raw_dir / "cicids2017"
    archive = fetch(CICIDS2017_ARCHIVE, base)
    out = base / "flows"
    csvs = sorted(out.glob("*.csv"))
    if len(csvs) != 8:
        csvs = extract_csvs(archive, out)
    if len(csvs) != 8:
        raise RuntimeError(f"Expected 8 CIC-IDS2017 CSVs, found {len(csvs)}")
    return csvs


def cicids2018_csvs(raw_dir: Path = RAW_DIR) -> list[Path]:
    base = raw_dir / "cicids2018"
    return [fetch(f, base) for f in CICIDS2018_FILES]
