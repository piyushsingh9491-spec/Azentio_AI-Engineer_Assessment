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
