"""FastAPI entrypoint. Run with: `uvicorn app.main:app --host 0.0.0.0 --port 8000`"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.config.settings import settings
from src.inference.predictor import get_predictor

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model backend=%s ...", settings.serving_model_backend)
    predictor = get_predictor()
    # Warm up the explainer (its first call is ~3.5s due to one-time SHAP
    # LinearExplainer setup — see src/explainability/explain.py) so the
    # first real user request isn't the one paying that cost.
    predictor.predict("warm up request for explainer initialization", explain=True)
    logger.info("Model backend ready")
    yield


app = FastAPI(
    title="Vehicle Complaint Classification API",
    description=(
        "Predicts the NHTSA fault-category taxonomy (e.g. `ENGINE`, `SERVICE BRAKES`, `AIR BAGS`) "
        "for a free-text vehicle complaint description.\n\n"
        "Trained on 271,517 real NHTSA complaint records with TF-IDF + Linear SVC (0.625 test macro-F1). "
        "See `docs/API.md` for the full endpoint reference, `docs/model_benchmark.md` for the DistilBERT "
        "comparison, and `docs/Architecture.md` for the system design."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"description": "Health and serving metrics for operators.", "name": "monitoring"},
        {"description": "Complaint classification endpoints.", "name": "inference"},
        {"description": "Model (re)training endpoints.", "name": "training"},
    ],
)
app.include_router(router, prefix="/api/v1")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/")
def root():
    return {"name": app.title, "version": app.version, "docs": "/docs"}
