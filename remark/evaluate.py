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

from remark.data import holdout_mask, label_ids
from remark.labels import LABELS
from remark.metrics import classification_report, macro_f1
from remark.models import Predictor
from remark.text import word_tokens

REFERENCE_FILE = "reference.json"


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
