import numpy as np
import pandas as pd
import pytest

from sentinelai.features import Preprocessor, SchemaError, canonicalize, sanitize, select_features, signed_log1p


def test_canonicalize_maps_drops_and_reports():
    df = pd.DataFrame([[1, 2, 3, 4, "10.0.0.1", "x", "BENIGN"]],
                      columns=[" Flow Duration", "Tot Fwd Pkts", "Fwd Header Length", "Fwd Header Length.1",
                               "Source IP", "Mystery", "Label"])
    feats, labels, report = canonicalize(df)
    assert list(feats.columns) == ["flow_duration", "total_fwd_packets", "fwd_header_bytes"]
    assert labels.tolist() == ["BENIGN"]
    assert report.identifiers_dropped == ["Source IP"]
    assert report.unexpected == ["Mystery"]
    assert report.label_column == "Label"


def test_canonicalize_duplicate_header_names_keeps_first():
    df = pd.DataFrame([[1, 99]], columns=["Flow Duration", "Flow Duration"])
    feats, _, _ = canonicalize(df)
    assert feats.shape == (1, 1) and feats.iloc[0, 0] == 1


def test_canonicalize_same_feature_under_two_namings_keeps_first():
    df = pd.DataFrame([[5, 7]], columns=["Total Fwd Packets", "Tot Fwd Pkts"])
    feats, _, _ = canonicalize(df)
    assert feats.to_dict("records") == [{"total_fwd_packets": 5}]


def test_sanitize_inf_negative_and_sentinel():
    X = pd.DataFrame({
        "flow_bytes_per_s": [np.inf, -np.inf, 5.0],
        "flow_duration": [-3.0, 0.0, 10.0],
        "init_win_bytes_fwd": [-1.0, -2.0, 8192.0],  # -1 = "not observed" sentinel
    })
    out = sanitize(X)
    assert out["flow_bytes_per_s"].isna().tolist() == [True, True, False]
    assert out["flow_duration"].isna().tolist() == [True, False, False]
    assert out["init_win_bytes_fwd"].tolist()[0] == -1.0
    assert np.isnan(out["init_win_bytes_fwd"].tolist()[1])


def test_preprocessor_fit_transform_roundtrip():
    train = pd.DataFrame({"flow_duration": [1.0, 3.0, np.inf, 5.0], "total_fwd_packets": [1, 2, 3, 4]})
    pre = Preprocessor(["flow_duration", "total_fwd_packets"]).fit(train)
    assert pre.medians["flow_duration"] == 3.0
    test = pd.DataFrame({"flow_duration": ["abc", "-1", "7"], "total_fwd_packets": ["1", None, "2"], "extra": 1})
    out = pre.transform(test)
    assert out["flow_duration"].tolist() == [3.0, 3.0, 7.0]  # unparsable and negative -> median
    assert out["total_fwd_packets"].tolist() == [1.0, 2.5, 2.0]
    assert list(out.columns) == ["flow_duration", "total_fwd_packets"]
    again = Preprocessor.from_dict(pre.to_dict())
    pd.testing.assert_frame_equal(again.transform(test), out)


def test_preprocessor_missing_column():
    pre = Preprocessor(["flow_duration", "total_fwd_packets"], {"flow_duration": 0.0, "total_fwd_packets": 0.0})
    with pytest.raises(SchemaError, match="total_fwd_packets"):
        pre.transform(pd.DataFrame({"flow_duration": [1.0]}))


def test_invalid_mask_counts_bad_cells():
    pre = Preprocessor(["flow_duration", "flow_bytes_per_s"], {"flow_duration": 0.0, "flow_bytes_per_s": 0.0})
    mask = pre.invalid_mask(pd.DataFrame({"flow_duration": ["1", "x"], "flow_bytes_per_s": ["Infinity", "2"]}))
    assert mask.sum(axis=1).tolist() == [1, 1]


def test_select_features_reasons():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1000, 500).astype(float)
    X = pd.DataFrame({
        "destination_port": rng.integers(0, 65535, 500),
        "flow_duration": a,
        "total_fwd_packets": a,  # identical
        "total_bwd_packets": a * 2 + 1,  # monotone -> |rho| = 1
        "fwd_psh_flags": np.zeros(500),  # constant
        "bwd_pkt_len_max": rng.normal(size=500),
    })
    rep = select_features(X)
    assert rep.selected == ["flow_duration", "bwd_pkt_len_max"]
    assert rep.dropped["destination_port"].startswith("policy")
    assert rep.dropped["total_fwd_packets"] == "identical to flow_duration"
    assert "Spearman" in rep.dropped["total_bwd_packets"]
    assert rep.dropped["fwd_psh_flags"] == "constant in training data"


def test_signed_log1p():
    np.testing.assert_allclose(signed_log1p(np.array([-np.e + 1, 0.0, np.e - 1])), [-1.0, 0.0, 1.0])
