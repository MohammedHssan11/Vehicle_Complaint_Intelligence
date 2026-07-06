# Multi-stage build. Note: torch + transformers + mlflow + shap make this a
# genuinely large image (~2-3GB) regardless of multi-stage tricks — that's
# the real dependency footprint of the "compare classical ML vs transformer"
# requirement, not something to hide. The multi-stage split only avoids
# baking in build-only tooling (gcc, pip caches) into the final layer.

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Bake in the NLTK corpora the preprocessing pipeline needs so the runtime
# container never depends on network access to nltk.org.
RUN python -m nltk.downloader -d /opt/nltk_data stopwords wordnet omw-1.4


FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nltk_data /opt/nltk_data
ENV PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA="/opt/nltk_data" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

COPY pyproject.toml ./
COPY src/ src/
COPY app/ app/

# data/raw, data/processed, artifacts, models, logs, experiments are volume
# mounts (see docker-compose.yml) — the trained model artifacts are not
# baked into the image so retraining doesn't require a rebuild.
RUN mkdir -p data/raw data/interim data/processed artifacts models logs experiments \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

# Default command runs the API; docker-compose overrides this for the
# streamlit service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
