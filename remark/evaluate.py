"""Shared evaluation + report writing for every model.

Leakage boundary: `test` is only ever passed to `evaluate_on_test`; nothing in
this module fits anything on it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from remark.calibration import alert_metrics, choose_threshold, fit_temperature
from remark.data import holdout_mask, label_ids
from remark.labels import BENIGN_ID, LABELS
from remark.metrics import (
    classification_report, expected_calibration_error, macro_f1, negative_log_likelihood, softmax,
)
from remark.models import Predictor
from remark.text import word_tokens

REFERENCE_FILE = "reference.json"
TARGET_RECALL = 0.90
ECE_BINS = 10


def data_fingerprint(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:8]


def evaluate_on_test(predictor: Predictor, test: pd.DataFrame, fit_splits: list[pd.DataFrame]) -> dict:
    """Metrics on the full test set plus held-out-template vs in-distribution rows."""
    y = label_ids(test)
    pred = predictor.predict_proba(test["narration"].tolist()).argmax(axis=1)
    mask = holdout_mask(test, fit_splits)
    report = classification_report(y, pred)
    report["holdout_templates"] = {"rows": int(mask.sum()), "macro_f1": macro_f1(y[mask], pred[mask])}
    report["in_distribution"] = {"rows": int((~mask).sum()),
                                 "macro_f1": macro_f1(y[~mask], pred[~mask])}
    return report


def calibrate_on_val(predictor: Predictor, val: pd.DataFrame) -> None:
    """Fit temperature, then the alert threshold on calibrated val risk; store both in meta."""
    y_val = label_ids(val)
    logits = predictor.raw_logits(val["narration"].tolist())
    temperature = fit_temperature(logits, y_val)
    risk = 1.0 - softmax(logits / temperature)[:, BENIGN_ID]
    predictor.meta["temperature"] = temperature
    predictor.meta["threshold"] = choose_threshold(risk, y_val != BENIGN_ID, TARGET_RECALL)
    predictor.meta["threshold_rule"] = f">= {TARGET_RECALL:.0%} recall of risky val rows"


def calibration_on_test(predictor: Predictor, test: pd.DataFrame) -> dict:
    """ECE/NLL before and after temperature scaling, and alerting at the val-chosen threshold."""
    y = label_ids(test)
    logits = predictor.raw_logits(test["narration"].tolist())
    before, after = softmax(logits), softmax(logits / predictor.temperature)
    return {
        "temperature": predictor.temperature,
        "ece_bins": ECE_BINS,
        "ece_before": expected_calibration_error(before, y, ECE_BINS),
        "ece_after": expected_calibration_error(after, y, ECE_BINS),
        "nll_before": negative_log_likelihood(before, y),
        "nll_after": negative_log_likelihood(after, y),
        "alerting": alert_metrics(1.0 - after[:, BENIGN_ID], y != BENIGN_ID, predictor.threshold),
    }


def build_reference(predictor: Predictor, train: pd.DataFrame) -> dict:
    """Training-time reference for drift: predicted-label distribution and vocabulary."""
    texts = train["narration"].tolist()
    pred = predictor.predict_proba(texts).argmax(axis=1)
    counts = np.bincount(pred, minlength=len(LABELS))
    vocab = Counter(tok for text in texts for tok in word_tokens(text))
    return {"label_dist": (counts / counts.sum()).tolist(), "vocab": sorted(vocab)}


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
