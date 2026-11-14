"""Train the TF-IDF + logistic-regression baseline.

    python -m remark.train_baseline [--data data/] [--models models/] [--reports reports/]

Fits on train.csv; temperature and alert threshold on val.csv; test.csv is
only evaluated. Writes models/baseline/ and reports/baseline.json.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from remark.data import label_ids, load_split
from remark.evaluate import (
    REFERENCE_FILE, build_reference, calibrate_on_val, calibration_on_test, data_fingerprint,
    evaluate_on_test, write_json,
)
from remark.models import BaselinePredictor
from remark.text import normalize, word_tokens

# C=1: val macro-F1 is flat over C in [0.3, 30], ties broken toward stronger regularisation.
PARAMS = {"C": 1.0, "class_weight": "balanced", "char_ngrams": (2, 5), "min_df": 2}


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer=word_tokens, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", preprocessor=normalize,
                                 ngram_range=PARAMS["char_ngrams"], min_df=PARAMS["min_df"],
                                 sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=PARAMS["C"], class_weight=PARAMS["class_weight"], max_iter=3000)
    return Pipeline([("tfidf", features), ("clf", clf)])


def fit_baseline(train: pd.DataFrame) -> BaselinePredictor:
    pipeline = build_pipeline().fit(train["narration"].tolist(), label_ids(train))
    return BaselinePredictor(pipeline)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train the TF-IDF baseline")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    train, val, test = (load_split(args.data, s) for s in ("train", "val", "test"))
    start = time.perf_counter()
    predictor = fit_baseline(train)
    train_seconds = time.perf_counter() - start

    model_dir = args.models / "baseline"
    predictor.meta = {"version": f"baseline-{data_fingerprint(args.data / 'train.csv')}",
                      "params": {k: list(v) if isinstance(v, tuple) else v
                                 for k, v in PARAMS.items()}}
    calibrate_on_val(predictor, val)  # val only: temperature + alert threshold
    predictor.save(model_dir)
    write_json(build_reference(predictor, train), model_dir / REFERENCE_FILE)

    report = {"model": "baseline", "version": predictor.version,
              "train_seconds": round(train_seconds, 2),
              "test": evaluate_on_test(predictor, test, [train, val]),
              "calibration": calibration_on_test(predictor, test)}
    write_json(report, args.reports / "baseline.json")
    t = report["test"]
    print(f"baseline test macro-F1 {t['macro_f1']:.4f} | held-out templates "
          f"{t['holdout_templates']['macro_f1']:.4f} | in-distribution "
          f"{t['in_distribution']['macro_f1']:.4f} | trained in {train_seconds:.1f}s")
    return report


if __name__ == "__main__":
    main()
