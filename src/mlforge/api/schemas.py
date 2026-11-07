from pydantic import BaseModel, Field


class PredictRequest(BaseModel):

    features: list[float | None] = Field(min_length=1)


class PredictResponse(BaseModel):
    label: int
    probabilities: list[float]


class BatchPredictRequest(BaseModel):

    instances: list[list[float | None]] = Field(min_length=1)


class BatchPredictResponse(BaseModel):
    count: int
    labels: list[int]
    probabilities: list[list[float]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
    n_features: int | None = None
    n_classes: int | None = None
    checkpoint: str | None = None


class ErrorResponse(BaseModel):
    detail: str