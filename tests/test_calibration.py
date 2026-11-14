import numpy as np
import pytest

from remark.calibration import alert_metrics, choose_threshold, fit_temperature
from remark.metrics import expected_calibration_error, negative_log_likelihood, softmax


def test_ece_matches_hand_computed_example():
    # Top-class confidences 0.9, 0.9, 0.6, 0.6 -> bins (0.8,0.9] and (0.5,0.6].
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.4, 0.6], [0.4, 0.6]])
    y = np.array([0, 1, 1, 1])
    # bin 0.9: acc 0.5, conf 0.9 -> |0.5-0.9| * 2/4 = 0.2
    # bin 0.6: acc 1.0, conf 0.6 -> |1.0-0.6| * 2/4 = 0.2
    assert expected_calibration_error(probs, y, n_bins=10) == pytest.approx(0.4)


def test_ece_is_zero_when_perfectly_calibrated():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, np.array([0, 1])) == pytest.approx(0.0)


def test_temperature_scaling_lowers_nll_on_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=400)
    logits = rng.normal(size=(400, 3))
    logits[np.arange(400), y] += 1.0
    logits *= 8.0  # far too confident for a ~60%-accurate classifier
    before = negative_log_likelihood(softmax(logits), y)
    t = fit_temperature(logits, y)
    after = negative_log_likelihood(softmax(logits / t), y)
    assert t > 1.0 and after < before


def test_threshold_meets_recall_target_and_is_highest_such():
    rng = np.random.default_rng(1)
    risk = rng.uniform(size=500)
    is_risky = rng.uniform(size=500) < 0.3
    t = choose_threshold(risk, is_risky, target_recall=0.90)
    assert alert_metrics(risk, is_risky, t)["recall"] >= 0.90
    higher = np.min(risk[is_risky & (risk > t)])  # next risky score above t
    assert alert_metrics(risk, is_risky, higher)["recall"] < 0.90


def test_threshold_requires_risky_rows():
    with pytest.raises(ValueError):
        choose_threshold(np.array([0.1, 0.2]), np.array([False, False]))
