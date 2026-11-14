import numpy as np
import pytest
from scipy.special import softmax

from sentinelai.training import evaluate as ev
from sentinelai.training.calibration import fit_temperature
from sentinelai.training.models import class_weights


def test_wilson_interval():
    lo, hi = ev.wilson_interval(5, 5)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.5655, abs=1e-3)  # tiny n -> wide interval
    lo, hi = ev.wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 1.0)


def test_binary_report_counts():
    truth = np.array([1, 1, 0, 0, 0])
    flagged = np.array([1, 0, 1, 0, 0])
    r = ev.binary_report(truth, flagged, np.array([0.9, 0.2, 0.8, 0.1, 0.3]))
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 2)
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["false_positive_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert 0 <= r["roc_auc"] <= 1


def test_multiclass_report():
    y = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 1, 1, 1, 2])
    proba = np.eye(3)[pred] * 0.9 + 0.1 / 3
    r = ev.multiclass_report(y, pred, proba, ["a", "b", "c"])
    assert r["accuracy"] == 0.8
    assert r["per_class"]["a"]["recall"] == 0.5 and r["per_class"]["b"]["precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert r["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]
    assert "roc_auc_macro_ovr" in r


def test_calibration_report_perfect_and_overconfident():
    conf = np.full(1000, 0.8)
    perfect = np.r_[np.ones(800), np.zeros(200)].astype(bool)
    assert ev.calibration_report(conf, perfect)["ece"] == pytest.approx(0.0)
    assert ev.calibration_report(np.full(1000, 0.99), perfect)["ece"] == pytest.approx(0.19, abs=1e-6)


def test_temperature_recovers_true_value():
    rng = np.random.default_rng(0)
    logits = rng.normal(scale=3, size=(20000, 4))
    y = np.array([rng.choice(4, p=p) for p in softmax(logits / 2.0, axis=1)])
    assert fit_temperature(logits, y) == pytest.approx(2.0, rel=0.08)


def test_class_weights_sqrt_and_cap():
    y = np.array([0] * 10000 + [1] * 100 + [2] * 1)
    w = class_weights(y, 3, power=0.5, cap=50)
    assert w[0] == 1.0 and w[1] == pytest.approx(10.0) and w[2] == 50.0


def test_downsample_curve_keeps_endpoints():
    x = np.linspace(0, 1, 5000)
    pts = ev.downsample_curve(x, np.sqrt(x), points=100)
    assert len(pts) <= 100 and pts[0][0] == 0.0 and pts[-1][0] == 1.0
