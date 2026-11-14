"""Tests for cleaning, leakage-safe splitting and weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phishlens.ml.data import clean, featurize, grouped_split, source_balanced_weights


def _synthetic(n_domains: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n_domains):
        label = i % 2
        source = "a" if i % 3 else "b"
        for j in range(4):
            rows.append({"url": f"https://site{i}.com/page{j}", "label": label, "source": source})
    return pd.DataFrame(rows)


def test_clean_removes_canonical_duplicates():
    df = pd.DataFrame(
        {
            "url": ["https://www.a.com", "http://a.com/", "a.com", "https://b.com/x"],
            "label": [0, 0, 0, 1],
            "source": ["s"] * 4,
        }
    )
    out, report = clean(df)
    assert len(out) == 2
    assert report["duplicate_rows"] == 2


def test_clean_drops_conflicting_labels_and_garbage():
    df = pd.DataFrame(
        {
            "url": ["https://c.com/x", "http://www.c.com/x", "javascript:alert(1)", None, "https://ok.com"],
            "label": [0, 1, 1, 0, 0],
            "source": ["s"] * 5,
        }
    )
    out, report = clean(df)
    assert list(out["canonical"]) == ["ok.com"]
    assert report["conflicting_label_rows"] == 2
    assert report["unparseable"] == 1


def test_grouped_split_has_no_domain_overlap_and_covers_everything():
    df, _ = clean(_synthetic())
    tr, va, te = grouped_split(df)
    groups = [set(df.iloc[idx]["group"]) for idx in (tr, va, te)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert sorted(np.concatenate([tr, va, te]).tolist()) == list(range(len(df)))
    # stratification keeps both classes in every split
    for idx in (tr, va, te):
        assert df.iloc[idx]["label"].nunique() == 2


def test_grouped_split_is_reproducible():
    df, _ = clean(_synthetic())
    a = grouped_split(df)
    b = grouped_split(df)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_source_balanced_weights_equalise_cells():
    df = pd.DataFrame({"source": ["a"] * 90 + ["b"] * 10, "label": [0] * 60 + [1] * 30 + [0] * 5 + [1] * 5})
    w = source_balanced_weights(df)
    cell = df["source"] + df["label"].astype(str)
    totals = pd.Series(w).groupby(cell.values).sum()
    assert totals.max() == pytest.approx(totals.min())
    assert w.sum() == pytest.approx(len(df))


def test_featurize_shape():
    X = featurize(["https://a.com", "http://1.2.3.4/x"])
    assert X.shape == (2, 36)
