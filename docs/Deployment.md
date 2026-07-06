# Deployment

## Docker Compose (recommended)

```bash
cp .env.example .env
docker compose up --build
```

Starts two services from one image (`Dockerfile`):

| Service | Port | Command |
|---|---|---|
| `api` | 8000 | `uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| `streamlit` | 8501 | `streamlit run app/streamlit_app/Home.py --server.address 0.0.0.0 --server.port 8501` |

Both mount the same host directories (`data/processed`, `artifacts`, `models`, `logs`, `experiments`) so retraining via the API is immediately visible to the Streamlit app without rebuilding the image — model artifacts are **not** baked into the image, only the code is.

The image is intentionally not small (~2-3GB) — that's the real footprint of `torch` + `transformers` + `mlflow` + `shap`, not something multi-stage tricks hide. The Dockerfile still uses a multi-stage build to avoid baking in build-only tooling (`gcc`, pip caches) into the final layer, and bakes in the NLTK corpora (`stopwords`, `wordnet`, `omw-1.4`) so the runtime container never needs network access to nltk.org.

### Before your first run

The API and Streamlit containers need trained model artifacts already present on the host at `models/production/` (run `python -m src.data.pipeline` and `python -m src.training.train` locally first, or run them once inside a container — see below).

### Health checks

The `api` service has a Docker healthcheck hitting `/api/v1/health` every 30s. Check status with `docker compose ps`.

### Troubleshooting

- **`docker compose build` hangs or fails with a connection/RPC error partway through**: this was observed in development — Docker Desktop's engine can crash mid-build under load. Restart Docker Desktop and re-run `docker compose build`; BuildKit layer caching means a retry after a successful partial build is fast.
- **`env file .env not found`**: `docker-compose.yml` marks `.env` as optional (`required: false`), but if you rely on any non-default settings (e.g. `SERVING_MODEL_BACKEND=transformer`), run `cp .env.example .env` first and edit it.

## Running training inside a container

```bash
docker compose run --rm api python -m src.data.pipeline
docker compose run --rm api python -m src.training.train
```

Since `data/processed`, `artifacts`, and `models` are bind-mounted, the results land on the host exactly as if run locally.

## Environment variables

See `.env.example`. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `SERVING_MODEL_BACKEND` | `baseline` | `baseline` (TF-IDF+LinearSVC, fast, full explainability) or `transformer` (DistilBERT, requires `models/transformer/` to exist) |
| `MLFLOW_TRACKING_URI` | `sqlite:///experiments/mlflow.db` | MLflow 3.x deprecated the plain filesystem backend; SQLite is the lightweight default |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Passed through to uvicorn if you don't use the `command:` override in docker-compose |
| `LOG_LEVEL` | `INFO` | Python logging level for the API process |

## Without Docker

```bash
pip install -r requirements.txt
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
streamlit run app/streamlit_app/Home.py
```

`pip install -e .` matters here: it makes `src` importable regardless of working directory, which Streamlit's multipage script-execution model needs (each `pages/*.py` file runs as an independent script).
