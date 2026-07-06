"""FastAPI routes (Phase 15): health, metrics, predict, batch-predict, retrain."""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictionResponse,
    RetrainResponse,
)
from src.config.settings import settings
from src.inference.predictor import InvalidInputError
from src.services.classification_service import get_classification_service

logger = logging.getLogger(__name__)
router = APIRouter()

_retrain_state = {"in_progress": False}


@router.get("/health", response_model=HealthResponse, tags=["monitoring"], summary="Service health check")
def health() -> HealthResponse:
    """Returns whether the configured model backend (`SERVING_MODEL_BACKEND`) loaded successfully.
    Use this for load-balancer / container health checks (see the Docker healthcheck in docker-compose.yml)."""
    try:
        get_classification_service()
        model_loaded = True
    except Exception:
        logger.exception("Health check: predictor failed to load")
        model_loaded = False
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_backend=settings.serving_model_backend,
        model_loaded=model_loaded,
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["monitoring"], summary="In-process serving metrics")
def metrics() -> MetricsResponse:
    """Prediction count, average latency, and class distribution served by this process so far.
    Resets on restart — see src/monitoring/prediction_log.py for why this is intentionally simple for v1."""
    return MetricsResponse(**get_classification_service().get_metrics())


def _predict(text: str, explain: bool) -> PredictionResponse:
    try:
        result = get_classification_service().classify(text, explain=explain)
    except InvalidInputError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PredictionResponse(**result.to_dict())


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Classify a single complaint",
    responses={422: {"description": "Empty, too short (<5 chars), or too long (>10,000 chars) text."}},
)
def predict(request: PredictRequest) -> PredictionResponse:
    """Predicts the fault category for one complaint description, with confidence,
    top-3 alternatives, and (for the baseline backend) a word-level SHAP explanation."""
    return _predict(request.text, request.explain)


@router.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    tags=["inference"],
    summary="Classify up to 500 complaints in one request",
)
def batch_predict(request: BatchPredictRequest) -> BatchPredictionResponse:
    """Same prediction as /predict, applied to a list. Returns one result per input, same order."""
    predictions = [_predict(text, request.explain) for text in request.texts]
    return BatchPredictionResponse(predictions=predictions)


def _run_retrain_job() -> None:
    from src.training.train import train_and_select

    try:
        logger.info("Retrain job started")
        train_and_select()
        get_classification_service.cache_clear()  # next request loads the freshly promoted model
        logger.info("Retrain job finished, predictor cache cleared")
    except Exception:
        logger.exception("Retrain job failed")
    finally:
        _retrain_state["in_progress"] = False


@router.post(
    "/retrain",
    response_model=RetrainResponse,
    tags=["training"],
    summary="Trigger a background baseline retrain",
)
def retrain(background_tasks: BackgroundTasks) -> RetrainResponse:
    """Kicks off a full baseline retrain (data pipeline must already be built
    via `python -m src.data.pipeline`) as a background task. Retraining on
    the full ~188K-row training set takes several minutes, so this endpoint
    returns immediately rather than blocking the request — there is no
    persistent job-tracking store in this v1, only the single in-process
    `in_progress` flag below, so status is only meaningful within the
    lifetime of one running API process.
    """
    if _retrain_state["in_progress"]:
        return RetrainResponse(status="already_running", message="A retrain job is already in progress")

    _retrain_state["in_progress"] = True
    background_tasks.add_task(_run_retrain_job)
    return RetrainResponse(status="started", message="Retrain job started in the background")
