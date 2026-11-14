"""Drift check against the training-time reference stored next to each model.

Two signals:
  * PSI of the predicted-label distribution vs the training predictions;
  * out-of-vocabulary rate of word tokens (purely numeric tokens ignored, since
    order ids and amounts are always new).
"""

from __future__ import annotations

import numpy as np

from remark.labels import LABELS
from remark.text import word_tokens

PSI_THRESHOLD = 0.2      # industry rule of thumb: > 0.2 = significant shift
OOV_THRESHOLD = 0.25  # in-dist val 0.10, test (30% new templates) 0.15, drift batch 0.41
PSI_EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = PSI_EPS) -> float:
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def oov_rate(texts: list[str], vocab: set[str]) -> float:
    tokens = [t for text in texts for t in word_tokens(text) if not t.isdigit()]
    return sum(t not in vocab for t in tokens) / max(1, len(tokens))


def drift_report(predicted: np.ndarray, texts: list[str], reference: dict) -> dict:
    counts = np.bincount(predicted, minlength=len(LABELS))
    dist = counts / max(1, counts.sum())
    psi_value = psi(np.array(reference["label_dist"]), dist)
    oov = oov_rate(texts, set(reference["vocab"]))
    return {
        "n": len(texts),
        "psi": psi_value,
        "oov_rate": oov,
        "predicted_label_dist": dict(zip(LABELS, dist.round(4).tolist())),
        "thresholds": {"psi": PSI_THRESHOLD, "oov_rate": OOV_THRESHOLD},
        "drift": bool(psi_value > PSI_THRESHOLD or oov > OOV_THRESHOLD),
    }
