import zipfile

import numpy as np
import pandas as pd
import pytest

from sentinelai.data import sources
from sentinelai.data.loading import _clean_raw_label, clean, parse_time_2017, parse_time_2018
from sentinelai.data.splitting import stratified_split, temporal_split
from sentinelai.schema import FEATURE_NAMES


def _frame(rows):
    """rows: list of (feature value, raw label)."""
    df = pd.DataFrame(0.0, index=range(len(rows)), columns=list(FEATURE_NAMES))
    df["flow_duration"] = [r[0] for r in rows]
    df["raw_label"] = [r[1] for r in rows]
    df["day"] = "d"
    df["time_key"] = np.arange(len(rows), dtype=float)
    return df


def test_clean_removes_duplicates_and_conflicts_before_split():
    data = _frame([
        (1, "BENIGN"), (1, "BENIGN"), (1, "BENIGN"),  # duplicates -> keep one
        (2, "DoS Hulk"), (2, "BENIGN"),  # same vector, two labels -> remove both
        (3, "PortScan"),
        (4, "Heartbleed"),  # excluded label: kept with label NA (unseen-attack probe)
    ])
    fd = clean(data)
    assert fd.stats["duplicate_rows_removed"] == 2
    assert fd.stats["conflicting_rows_removed"] == 2
    assert fd.stats["conflicting_vectors"] == 1
    assert fd.frame["flow_duration"].tolist() == [1, 3, 4]
    assert fd.frame["label"].tolist()[:2] == ["BENIGN", "PortScan"]
    assert pd.isna(fd.frame["label"].iloc[2])
    assert fd.stats["excluded_labels"] == {"Heartbleed": 1}


def test_clean_counts_nan_and_inf():
    data = _frame([(1, "BENIGN"), (2, "BENIGN")])
    data.loc[0, "flow_bytes_per_s"] = np.inf
    data.loc[1, "flow_packets_per_s"] = np.nan
    stats = clean(data).stats
    assert stats["cells_inf"] == 1 and stats["cells_nan"] == 1


def test_stratified_split_is_disjoint_stratified_and_reproducible():
    y = pd.Series(["BENIGN"] * 700 + ["DoS"] * 200 + ["Bot"] * 100)
    s1, s2 = stratified_split(y, seed=1), stratified_split(y, seed=1)
    assert (s1 == s2).all()
    assert set(s1) == {"train", "val", "test"}
    assert (s1 == "train").sum() == 700
    for cls, n in (("BENIGN", 700), ("DoS", 200), ("Bot", 100)):
        assert abs((s1[y == cls] == "test").sum() - 0.15 * n) <= 1


def test_temporal_split_tests_on_later_traffic():
    frame = pd.DataFrame({
        "day": ["mon"] * 100 + ["tue"] * 40,
        "label": ["BENIGN"] * 60 + ["DoS"] * 40 + ["DoS"] * 40,
        "time_key": np.r_[np.random.default_rng(0).permutation(100), np.arange(40)].astype(float),
    })
    split = temporal_split(frame)
    for (_, _), g in frame.assign(s=split).groupby(["day", "label"]):
        assert g.loc[g.s == "train", "time_key"].max() <= g.loc[g.s == "val", "time_key"].min()
        assert g.loc[g.s == "val", "time_key"].max() <= g.loc[g.s == "test", "time_key"].min()
        assert (g.s == "test").sum() == round(len(g) * 0.15)


def test_parse_time_2017_resolves_missing_am_pm():
    ts = pd.Series(["5/7/2017 8:42", "5/7/2017 12:59", "5/7/2017 1:00", "03/07/2017 08:55:58", "garbage"])
    out = parse_time_2017(ts)
    assert out.iloc[:3].tolist() == [8 * 60 + 42, 12 * 60 + 59, 13 * 60]  # 1:00 is afternoon
    assert out.iloc[3] == pytest.approx(8 * 60 + 55 + 58 / 60)
    assert np.isnan(out.iloc[4])


def test_parse_time_2018():
    assert parse_time_2018(pd.Series(["14/02/2018 13:05:30"])).iloc[0] == pytest.approx(13 * 60 + 5.5)


def test_clean_raw_label_strips_mojibake():
    assert _clean_raw_label(pd.Series(["Web Attack ï¿½ Brute Force"])).iloc[0] == "Web Attack Brute Force"


def test_fetch_rejects_checksum_mismatch(tmp_path):
    (tmp_path / "f.csv").write_text("a,b\n1,2\n")
    remote = sources.RemoteFile(url="https://example.invalid/f.csv", sha256="0" * 64, filename="f.csv", size=8)
    with pytest.raises(sources.ChecksumError):
        sources.fetch(remote, tmp_path)


def test_fetch_refuses_non_https(tmp_path):
    remote = sources.RemoteFile(url="http://example.invalid/f.csv", sha256="0" * 64, filename="g.csv", size=8)
    with pytest.raises(ValueError, match="HTTPS"):
        sources.fetch(remote, tmp_path)


def test_extract_csvs_blocks_zip_slip(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("TrafficLabelling /Monday.csv", "x\n")
        zf.writestr("../../escape.csv", "evil\n")
        zf.writestr("dir/notes.txt", "ignored")
    out = tmp_path / "out"
    files = sources.extract_csvs(archive, out)
    assert sorted(p.name for p in files) == ["Monday.csv", "escape.csv"]
    assert all(p.parent == out for p in files)
    assert not (tmp_path / "escape.csv").exists() and not (tmp_path.parent / "escape.csv").exists()


def test_pinned_sources_are_https_with_sha256():
    for f in (sources.CICIDS2017_ARCHIVE, *sources.CICIDS2018_FILES):
        assert f.url.startswith("https://") and len(f.sha256) == 64
    assert sum(sources.EXPECTED_CICIDS2017_COUNTS.values()) == 2_830_743
