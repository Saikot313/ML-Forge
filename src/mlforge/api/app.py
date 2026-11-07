import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from mlforge.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from mlforge.config import InferenceConfig
from mlforge.exceptions import DataValidationError, MLForgeError
from mlforge.inference.predictor import Predictor
from mlforge.logging_utils import get_logger

logger = get_logger(__name__)
CHECKPOINT_ENV_VAR = "MLFORGE_CHECKPOINT"
DEFAULT_CHECKPOINT = "outputs/mlforge_default/checkpoints/best_model.pt"
HTTP_UNPROCESSABLE = 422
HTTP_TOO_LARGE = 413
HTTP_UNAVAILABLE = 503
HTTP_SERVER_ERROR = 500


def get_predictor(request: Request) -> Predictor:
    """Dependency returning the loaded predictor or a 503 response."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=HTTP_UNAVAILABLE,
            detail="Model is not loaded. Check the checkpoint path and server logs.",
        )
    return predictor


def create_app(
    checkpoint_path: str | Path | None = None,
    inference_config: InferenceConfig | None = None,
    predictor: Predictor | None = None,
) -> FastAPI:
    cfg = inference_config or InferenceConfig()
    path = Path(checkpoint_path or os.environ.get(CHECKPOINT_ENV_VAR, DEFAULT_CHECKPOINT))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor
        app.state.checkpoint = "injected" if predictor is not None else str(path)
        if predictor is None:
            try:
                app.state.predictor = Predictor.from_checkpoint(path, cfg)
            except MLForgeError:
                logger.exception("Could not load model from %s; prediction endpoints return 503.", path)
        yield
        app.state.predictor = None

    app = FastAPI(title="MLForge API", version="0.1.0", lifespan=lifespan)
    app.state.predictor = None
    app.state.lock = threading.Lock()  

    @app.exception_handler(DataValidationError)
    async def validation_handler(request: Request, exc: DataValidationError) -> JSONResponse:
        return JSONResponse(status_code=HTTP_UNPROCESSABLE, content={"detail": str(exc)})

    @app.exception_handler(MLForgeError)
    async def mlforge_handler(request: Request, exc: MLForgeError) -> JSONResponse:
        logger.error("Inference failed: %s", exc)
        return JSONResponse(
            status_code=HTTP_SERVER_ERROR, content={"detail": f"Inference failed: {exc}"}
        )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded: Predictor | None = request.app.state.predictor
        if loaded is None:
            return HealthResponse(
                status="degraded", model_loaded=False, checkpoint=request.app.state.checkpoint
            )
        return HealthResponse(
            status="ok",
            model_loaded=True,
            device=str(loaded.device),
            n_features=loaded.n_features,
            n_classes=loaded.n_classes,
            checkpoint=request.app.state.checkpoint,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={HTTP_UNPROCESSABLE: {"model": ErrorResponse}, HTTP_UNAVAILABLE: {"model": ErrorResponse}},
    )
    def predict(
        body: PredictRequest, request: Request, loaded: Predictor = Depends(get_predictor)
    ) -> PredictResponse:
        with request.app.state.lock:
            result = loaded.predict_one(body.features)
        return PredictResponse(**result)

    @app.post(
        "/predict/batch",
        response_model=BatchPredictResponse,
        responses={
            HTTP_UNPROCESSABLE: {"model": ErrorResponse},
            HTTP_TOO_LARGE: {"model": ErrorResponse},
            HTTP_UNAVAILABLE: {"model": ErrorResponse},
        },
    )
    def predict_batch(
        body: BatchPredictRequest, request: Request, loaded: Predictor = Depends(get_predictor)
    ) -> BatchPredictResponse:
        if len(body.instances) > cfg.max_api_batch_size:
            raise HTTPException(
                status_code=HTTP_TOO_LARGE,
                detail=f"Batch of {len(body.instances)} exceeds the limit of {cfg.max_api_batch_size}.",
            )
        with request.app.state.lock:
            proba = loaded.predict_proba(body.instances)
        return BatchPredictResponse(
            count=int(proba.shape[0]),
            labels=[int(i) for i in proba.argmax(axis=1)],
            probabilities=proba.tolist(),
        )

    return app