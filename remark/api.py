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

from fastapi import FastAPI

from remark.labels import LABELS
from remark.models import Predictor, load_predictor
from remark.schemas import (
    BatchRequest, BatchResponse, HealthResponse, ScoreRequest, ScoreResponse,
)

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
    return [
        ScoreResponse(
            label=LABELS[int(row.argmax())],
            probabilities={label: round(float(p), 6) for label, p in zip(LABELS, row)},
            risk_score=round(float(r), 6),
            model_version=predictor.version,
        )
        for row, r in zip(probs, risk)
    ]


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or load_predictor(settings.model, settings.models_dir)
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

    return app


app = create_app()
