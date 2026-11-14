"""Request/response models. Validation failures surface as HTTP 422."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from remark.labels import MAX_NARRATION_LEN

Narration = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                             max_length=MAX_NARRATION_LEN)]
Channel = Literal["UPI", "NEFT", "IMPS"]
MAX_BATCH = 256


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    narration: Narration
    channel: Channel
    amount: float = Field(ge=0, le=1e10)


class ScoreResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    risk_score: float
    model_version: str
    threshold: float | None = Field(description="calibrated risk_score alert threshold (fitted on val)")
    alert: bool | None = Field(description="risk_score >= threshold")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ScoreRequest] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    model_version: str


class ReviewItem(BaseModel):
    txn_id: str
    narration: str
    channel: str
    predicted_label: str
    uncertainty: float = Field(description="entropy of the predicted distribution, in nats")


class ReviewQueueResponse(BaseModel):
    model_version: str
    items: list[ReviewItem]


class LabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    txn_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    label: Literal["BENIGN", "GAMBLING_BETTING", "CRYPTO_P2P", "LOAN_APP",
                   "IMPERSONATION_SCAM", "MULE_TASK_JOB"]
    analyst_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64,
                                                 pattern=r"^[A-Za-z0-9_.\-]+$")]


class LabelResponse(BaseModel):
    id: int
    txn_id: str
    label: str
    analyst_id: str
    created_at: str


class DriftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    narrations: list[Narration] = Field(min_length=1, max_length=10_000)


class DriftResponse(BaseModel):
    n: int
    psi: float
    oov_rate: float
    predicted_label_dist: dict[str, float]
    thresholds: dict[str, float]
    drift: bool
