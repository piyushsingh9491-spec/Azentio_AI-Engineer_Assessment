"""Classification and calibration metrics (pure numpy / sklearn)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from remark.labels import LABELS


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=range(len(LABELS)), average="macro",
                          zero_division=0))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(LABELS)), zero_division=0)
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(y_true, y_pred, labels=range(len(LABELS))).tolist(),
        },
    }


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """ECE on the top-class probability with equal-width bins over [0, 1].

    Bins are (lo, hi] except the first, which also includes 0.
    """
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0))))
