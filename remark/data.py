"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from remark.labels import LABEL_TO_ID


def load_split(data_dir: str | Path, split: str) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / f"{split}.csv", dtype={"txn_id": str})


def label_ids(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map(LABEL_TO_ID).to_numpy()


def holdout_mask(test: pd.DataFrame, fit_splits: list[pd.DataFrame]) -> np.ndarray:
    """True for test rows whose template_id never appears in the given fit splits."""
    seen = set().union(*(set(df["template_id"]) for df in fit_splits))
    return ~test["template_id"].isin(seen).to_numpy()
