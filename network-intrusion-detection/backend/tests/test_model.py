"""Tests of the deployed model artifact and the inference/explainability code."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from sentinelai.features import SchemaError
from sentinelai.inference.detector import Detector, ModelIntegrityError, confidence_band, severity_for
from sentinelai.schema import CLASSES, FEATURE_BY_NAME, map_label
from tests.conftest import MODEL_DIR, SAMPLES_DIR, requires_model

pytestmark = requires_model


def test_checksums_verified_and_tampering_detected(tmp_path):
    for f in ("sentinel_xgb.ubj", "model_meta.json", "SHA256SUMS"):
        shutil.copy(MODEL_DIR / f, tmp_path / f)
    Detector.load(tmp_path)  # intact copy loads
    meta = json.loads((tmp_path / "model_meta.json").read_text())
    meta["temperature"] = 0.01
    (tmp_path / "model_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ModelIntegrityError):
        Detector.load(tmp_path)


def test_artifact_is_pickle_free():
    assert not list(MODEL_DIR.glob("*.pkl")) and not list(MODEL_DIR.glob("*.joblib"))
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    assert meta["classes"] == list(CLASSES)
    assert "destination_port" not in meta["preprocessor"]["features"]


def test_predictions_on_held_out_sample(detector, mixed_df):
    prep = detector.prepare(mixed_df)
    preds = detector.predict(prep.X)
    truth = mixed_df["Label"].map(map_label).iloc[prep.row_index].to_numpy()
    acc = (np.array(preds.labels()) == truth).mean()
    assert acc > 0.95, acc
    np.testing.assert_allclose(preds.proba.sum(1), 1.0, atol=1e-5)
    assert ((preds.confidence >= 1 / len(CLASSES)) & (preds.confidence <= 1)).all()
    np.testing.assert_allclose(preds.attack_probability, 1 - preds.proba[:, 0], atol=1e-6)


def test_ground_truth_labels_are_extracted(detector, mixed_df):
    prep = detector.prepare(mixed_df)
    assert prep.labels is not None and set(prep.labels.dropna()) <= set(CLASSES)


def test_temperature_scaling_preserves_argmax(detector, mixed_df):
    X = detector.prepare(mixed_df.head(500)).X
    z = detector.logits(X)
    assert (softmax(z, axis=1).argmax(1) == detector.predict(X).pred).all()


def test_predictions_are_deterministic(detector, mixed_df):
    X = detector.prepare(mixed_df.head(300)).X
    np.testing.assert_array_equal(detector.predict(X).proba, detector.predict(X).proba)


def test_cic2018_column_names_are_accepted(detector, mixed_df):
    renamed = mixed_df.head(200).rename(columns=lambda c: next(
        (f.cic2018 for f in FEATURE_BY_NAME.values() if f.cic2017 == c.strip()), c))
    a = detector.predict(detector.prepare(mixed_df.head(200)).X).pred
    b = detector.predict(detector.prepare(renamed).X).pred
    np.testing.assert_array_equal(a, b)


def test_missing_required_columns(detector, mixed_df):
    with pytest.raises(SchemaError, match="Missing"):
        detector.prepare(mixed_df.drop(columns=["Flow Duration", "Bwd Packet Length Max"]))


def test_extra_columns_are_ignored(detector, mixed_df):
    df = mixed_df.head(50).assign(Unexpected="x", **{"Source IP": "10.0.0.1"})
    prep = detector.prepare(df)
    assert "Unexpected" in prep.columns.unexpected and "Source IP" in prep.columns.identifiers_dropped
    assert len(prep.X) == 50


def test_rows_with_mostly_invalid_values_are_rejected(detector, mixed_df):
    df = mixed_df.head(10).copy()
    df.iloc[3, :-1] = "garbage"
    df.iloc[5, 2] = "NaN"  # one bad cell: imputed, not rejected
    prep = detector.prepare(df)
    assert prep.rejected_rows == [3]
    assert list(prep.row_index) == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    assert prep.imputed_cells[list(prep.row_index).index(5)] >= 1


def test_shap_values_are_additive_and_match_xgboost(detector, mixed_df):
    X = detector.prepare(mixed_df.sample(60, random_state=1)).X
    vals, base = detector.shap_values(X)
    assert vals.shape == (60, len(detector.features), len(CLASSES))
    np.testing.assert_allclose(base + vals.sum(1), detector.logits(X), atol=1e-3)
    native = np.transpose(detector.fast_contributions(X)[:, :, :-1], (0, 2, 1))
    np.testing.assert_allclose(vals, native, atol=1e-4)


def test_local_explanation_structure(detector, mixed_df):
    attack = mixed_df[mixed_df["Label"] == "DDoS"].head(1)
    X = detector.prepare(attack).X
    e = detector.explain(X)[0]
    assert e["predicted_class"] == "DDoS"
    assert e["contributions"] and e["attack_evidence"]
    impacts = [abs(c["impact"]) for c in e["contributions"]]
    assert impacts == sorted(impacts, reverse=True)
    assert all(c["impact"] > 0 for c in e["attack_evidence"])
    # the explanation reconstructs the calibrated logit of the predicted class
    total = e["base_value"] + sum(c["impact"] for c in e["contributions"]) + e["other_features_impact"]
    assert total == pytest.approx(e["logit"], abs=1e-3)
    assert sum(e["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_top_reasons_support_the_prediction(detector, mixed_df):
    X = detector.prepare(mixed_df.head(200)).X
    preds = detector.predict(X)
    reasons = detector.top_reasons(X, preds.pred, k=3)
    assert len(reasons) == 200
    assert all(0 < len(r) <= 3 and all(x["impact"] > 0 for x in r) for r in reasons if r)


def test_distribution_shift_low_for_2017_high_for_2018(detector, mixed_df):
    low = detector.distribution_shift(detector.prepare(mixed_df).X)
    assert low["level"] in ("low", "moderate")
    path = SAMPLES_DIR / "cicids2018_cross_dataset.csv"
    if path.exists():
        high = detector.distribution_shift(detector.prepare(pd.read_csv(path, dtype=str)).X)
        assert high["level"] == "high" and high["max_feature_rate"] > low["max_feature_rate"]


@pytest.mark.parametrize("cls,conf,expected", [
    ("BENIGN", 0.99, "none"), ("DDoS", 0.95, "critical"), ("DDoS", 0.5, "high"),
    ("PortScan", 0.9, "medium"), ("PortScan", 0.4, "low"), ("DoS", 0.7, "high"),
])
def test_severity(cls, conf, expected):
    assert severity_for(cls, conf) == expected


def test_confidence_band():
    assert [confidence_band(c) for c in (0.95, 0.7, 0.3)] == ["high", "medium", "low"]
