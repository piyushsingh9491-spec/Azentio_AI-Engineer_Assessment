"""Temperature scaling and alert-threshold selection.

Both are fitted on the validation split only. Callers must never pass test data.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar

from remark.metrics import negative_log_likelihood, softmax


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Single scalar T > 0 minimising NLL of softmax(logits / T); searched on log T."""
    def nll(log_t: float) -> float:
        return negative_log_likelihood(softmax(logits / math.exp(log_t)), y_true)

    result = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return float(math.exp(result.x))


def choose_threshold(risk_scores: np.ndarray, is_risky: np.ndarray, target_recall: float = 0.90) -> float:
    """Highest threshold t such that flagging `risk >= t` catches >= target_recall of risky rows.

    The highest such t minimises alerts (false positives) subject to the recall floor.
    """
    risky_scores = np.sort(risk_scores[is_risky.astype(bool)])[::-1]
    if len(risky_scores) == 0:
        raise ValueError("need at least one risky row to choose a threshold")
    k = math.ceil(target_recall * len(risky_scores))
    return float(risky_scores[k - 1])


def alert_metrics(risk_scores: np.ndarray, is_risky: np.ndarray, threshold: float) -> dict:
    alerts = risk_scores >= threshold
    is_risky = is_risky.astype(bool)
    true_alerts = int((alerts & is_risky).sum())
    return {
        "threshold": threshold,
        "recall": true_alerts / max(1, int(is_risky.sum())),
        "precision": true_alerts / max(1, int(alerts.sum())),
        "alerts_per_1000": 1000 * float(alerts.mean()),
    }
