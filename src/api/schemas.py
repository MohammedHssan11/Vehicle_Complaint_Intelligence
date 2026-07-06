"""Pydantic request/response schemas for the FastAPI service."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_EXAMPLE_TEXT = "The brake pedal did not engage properly and the car would not stop in time."


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Raw complaint description text (5-10,000 characters).",
        examples=[_EXAMPLE_TEXT],
    )
    explain: bool = Field(
        True, description="Whether to include word-level SHAP explanation (baseline backend only)."
    )

    model_config = ConfigDict(json_schema_extra={"example": {"text": _EXAMPLE_TEXT, "explain": True}})


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of complaint descriptions, max 500 per request.",
    )
    explain: bool = Field(True, description="Whether to include word-level explanation for each prediction.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    _EXAMPLE_TEXT,
                    "Engine stalls at highway speed without warning.",
                ],
                "explain": False,
            }
        }
    )


class TopKItem(BaseModel):
    label: str = Field(..., description="Fault category name (one of the 27 curated classes + OTHER).")
    confidence: float = Field(..., description="Predicted probability for this category, 0-1.")


class ExplanationTerm(BaseModel):
    term: str = Field(..., description="A word/n-gram present in the input text.")
    contribution: float = Field(
        ..., description="SHAP contribution toward the predicted class — positive supports it, negative argues against it."
    )


class PredictionResponse(BaseModel):
    predicted_label: str = Field(..., description="The highest-confidence fault category.")
    confidence: float = Field(..., description="Confidence (probability) of the predicted label, 0-1.")
    top_k: list[TopKItem] = Field(..., description="Top 3 categories by confidence, descending.")
    explanation: list[ExplanationTerm] | None = Field(
        None, description="Top contributing terms, or null if explain=false or the backend doesn't support it."
    )
    explanation_method: str | None = Field(None, description='"shap", "coefficient" (fallback), or null.')
    model_backend: str = Field(..., description='"baseline" or "transformer" — see SERVING_MODEL_BACKEND.')
    latency_ms: float = Field(..., description="Server-side inference time for this prediction, in milliseconds.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_label": "SERVICE BRAKES",
                "confidence": 0.6087,
                "top_k": [
                    {"label": "SERVICE BRAKES", "confidence": 0.6087},
                    {"label": "ELECTRICAL SYSTEM", "confidence": 0.1296},
                    {"label": "POWER TRAIN", "confidence": 0.0718},
                ],
                "explanation": [
                    {"term": "brake", "contribution": 0.6474},
                    {"term": "not stop", "contribution": 0.2595},
                ],
                "explanation_method": "shap",
                "model_backend": "baseline",
                "latency_ms": 14.7,
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(..., description="One prediction per input text, same order.")


class HealthResponse(BaseModel):
    status: str = Field(..., description='"ok" if the model loaded successfully, "degraded" otherwise.')
    model_backend: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    total_predictions: int = Field(..., description="Predictions served since this API process started.")
    avg_latency_ms: float | None = Field(None, description="Average inference latency across all served predictions.")
    class_distribution: dict[str, int] = Field(..., description="Count of predictions per fault category so far.")
    uptime_seconds: float = Field(..., description="Seconds since this API process started.")


class RetrainResponse(BaseModel):
    status: str = Field(..., description='"started" or "already_running".')
    message: str
