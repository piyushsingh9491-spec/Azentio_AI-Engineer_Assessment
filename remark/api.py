"""FastAPI scoring service.

    uvicorn remark.api:app --port 8000          # REMARK_MODEL=baseline|finetuned

Configuration comes from environment variables (see .env.example).
Narrations are never logged: they can contain names and account handles.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import json

import numpy as np
from fastapi import FastAPI, HTTPException, Query, status

from remark.data import load_split
from remark.drift import drift_report
from remark.evaluate import REFERENCE_FILE
from remark.labels import LABELS
from remark.models import Predictor, load_predictor
from remark.schemas import (
    BatchRequest, BatchResponse, DriftRequest, DriftResponse, HealthResponse, LabelRequest,
    LabelResponse, ReviewItem, ReviewQueueResponse, ScoreRequest, ScoreResponse,
)
from remark.store import LabelStore

logger = logging.getLogger("remark.api")


@dataclass(frozen=True)
class Settings:
    model: str = "baseline"
    models_dir: Path = Path("models")
    data_dir: Path = Path("data")
    labels_db: Path = Path("data/labels.db")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.getenv("REMARK_MODEL", cls.model),
            models_dir=Path(os.getenv("REMARK_MODELS_DIR", str(cls.models_dir))),
            data_dir=Path(os.getenv("REMARK_DATA_DIR", str(cls.data_dir))),
            labels_db=Path(os.getenv("REMARK_LABELS_DB", str(cls.labels_db))),
        )


def score(predictor: Predictor, items: list[ScoreRequest]) -> list[ScoreResponse]:
    probs = predictor.predict_proba([item.narration for item in items])
    risk = predictor.risk_score(probs)
    threshold = predictor.threshold
    return [
        ScoreResponse(
            label=LABELS[int(row.argmax())],
            probabilities={label: round(float(p), 6) for label, p in zip(LABELS, row)},
            risk_score=round(float(r), 6),
            model_version=predictor.version,
            threshold=threshold,
            alert=None if threshold is None else bool(r >= threshold),
        )
        for row, r in zip(probs, risk)
    ]


def entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


class ReviewPool:
    """train.csv with labels dropped, pre-scored once by the loaded model."""

    def __init__(self, predictor: Predictor, data_dir: Path):
        pool = load_split(data_dir, "train")[["txn_id", "narration", "channel"]]
        probs = predictor.predict_proba(pool["narration"].tolist())
        self.frame = pool.assign(uncertainty=entropy(probs),
                                 predicted=[LABELS[i] for i in probs.argmax(axis=1)])
        self.frame = self.frame.sort_values(["uncertainty", "txn_id"], ascending=[False, True])
        self.txn_ids = set(pool["txn_id"])

    def most_uncertain(self, k: int, exclude: set[str]) -> list[ReviewItem]:
        rows = self.frame[~self.frame["txn_id"].isin(exclude)].head(k)
        return [ReviewItem(txn_id=r.txn_id, narration=r.narration, channel=r.channel,
                           predicted_label=r.predicted, uncertainty=round(float(r.uncertainty), 6))
                for r in rows.itertuples()]


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or load_predictor(settings.model, settings.models_dir)
        model_dir = settings.models_dir / app.state.predictor.name
        app.state.reference = json.loads((model_dir / REFERENCE_FILE).read_text())
        app.state.pool = ReviewPool(app.state.predictor, settings.data_dir)
        app.state.store = LabelStore(settings.labels_db)
        logger.info("loaded model %s", app.state.predictor.version)
        yield

    app = FastAPI(title="Remark scoring API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        p: Predictor = app.state.predictor
        return HealthResponse(status="ok", model=p.name, model_version=p.version)

    # Plain `def` endpoints run in FastAPI's threadpool, so CPU-bound inference
    # does not block the event loop.
    @app.post("/v1/score", response_model=ScoreResponse)
    def score_one(request: ScoreRequest) -> ScoreResponse:
        return score(app.state.predictor, [request])[0]

    @app.post("/v1/score/batch", response_model=BatchResponse)
    def score_batch(request: BatchRequest) -> BatchResponse:
        return BatchResponse(results=score(app.state.predictor, request.items))

    @app.get("/v1/review-queue", response_model=ReviewQueueResponse)
    def review_queue(k: int = Query(20, ge=1, le=500)) -> ReviewQueueResponse:
        items = app.state.pool.most_uncertain(k, app.state.store.labelled_txn_ids())
        return ReviewQueueResponse(model_version=app.state.predictor.version, items=items)

    @app.post("/v1/labels", response_model=LabelResponse, status_code=status.HTTP_201_CREATED)
    def add_label(request: LabelRequest) -> LabelResponse:
        if request.txn_id not in app.state.pool.txn_ids:
            raise HTTPException(422,  # same status as validation errors
                                detail=f"unknown txn_id {request.txn_id!r}")
        return LabelResponse(**app.state.store.add(request.txn_id, request.label,
                                                   request.analyst_id))

    @app.post("/v1/drift", response_model=DriftResponse)
    def drift(request: DriftRequest) -> DriftResponse:
        predicted = app.state.predictor.predict_proba(request.narrations).argmax(axis=1)
        return DriftResponse(**drift_report(predicted, request.narrations, app.state.reference))

    return app


app = create_app()
