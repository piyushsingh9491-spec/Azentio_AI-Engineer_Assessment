"""Model-agnostic predictor interface.

Every model exposes raw logits; calibration (temperature), the alert threshold
and the drift reference live in meta.json next to the weights, so the API,
calibration, active learning and drift code never care which model is loaded.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np

from remark.labels import BENIGN_ID, LABELS
from remark.metrics import softmax

META_FILE = "meta.json"


class Predictor(ABC):
    name: str

    def __init__(self, meta: dict | None = None):
        self.meta: dict = meta or {}

    @property
    def version(self) -> str:
        return self.meta.get("version", f"{self.name}-untrained")

    @property
    def temperature(self) -> float:
        return float(self.meta.get("temperature", 1.0))

    @property
    def threshold(self) -> float | None:
        value = self.meta.get("threshold")
        return None if value is None else float(value)

    @abstractmethod
    def raw_logits(self, texts: list[str]) -> np.ndarray:
        """Uncalibrated scores of shape (n, len(LABELS))."""

    def predict_proba(self, texts: list[str], calibrated: bool = True) -> np.ndarray:
        logits = self.raw_logits(texts)
        return softmax(logits / self.temperature if calibrated else logits)

    @staticmethod
    def risk_score(probs: np.ndarray) -> np.ndarray:
        return 1.0 - probs[:, BENIGN_ID]

    def save_meta(self, model_dir: Path) -> None:
        (model_dir / META_FILE).write_text(json.dumps(self.meta, indent=2, sort_keys=True))


class BaselinePredictor(Predictor):
    name = "baseline"

    def __init__(self, pipeline, meta: dict | None = None):
        super().__init__(meta)
        self.pipeline = pipeline

    def raw_logits(self, texts: list[str]) -> np.ndarray:
        # Log-probabilities act as logits: softmax(log p / T) is temperature scaling.
        probs = np.zeros((len(texts), len(LABELS)))
        probs[:, self.pipeline.classes_] = self.pipeline.predict_proba(texts)
        return np.log(np.clip(probs, 1e-9, 1.0))

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_dir / "model.joblib")
        self.save_meta(model_dir)

    @classmethod
    def load(cls, model_dir: Path) -> "BaselinePredictor":
        meta = json.loads((model_dir / META_FILE).read_text())
        return cls(joblib.load(model_dir / "model.joblib"), meta)


def load_predictor(name: str, models_dir: str | Path) -> Predictor:
    model_dir = Path(models_dir) / name
    if name == "baseline":
        return BaselinePredictor.load(model_dir)
    if name == "finetuned":
        from remark.finetune import FinetunedPredictor  # torch imported only when needed
        return FinetunedPredictor.load(model_dir)
    raise ValueError(f"unknown model {name!r}; expected 'baseline' or 'finetuned'")
