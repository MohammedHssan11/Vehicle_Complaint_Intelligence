# API Reference

FastAPI service at `app/main.py`. All endpoints are versioned under `/api/v1`. Interactive OpenAPI docs are served at `/docs` (Swagger UI) and `/redoc`.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## `GET /api/v1/health`

Returns whether the configured model backend loaded successfully.

```bash
curl http://localhost:8000/api/v1/health
```
```json
{"status": "ok", "model_backend": "baseline", "model_loaded": true}
```

## `GET /api/v1/metrics`

In-process serving metrics (counts, latency, class distribution). Resets on process restart — see `src/monitoring/prediction_log.py` for why this is intentionally simple for v1.

```bash
curl http://localhost:8000/api/v1/metrics
```
```json
{
  "total_predictions": 42,
  "avg_latency_ms": 12.3,
  "class_distribution": {"ENGINE": 10, "SERVICE BRAKES": 8},
  "uptime_seconds": 301.4
}
```

## `POST /api/v1/predict`

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The brake pedal did not engage properly.", "explain": true}'
```

Request:

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | Min 5 chars, max 10,000 chars |
| `explain` | bool | no (default `true`) | Word-level explanation; only implemented for the `baseline` backend |

Response:

```json
{
  "predicted_label": "SERVICE BRAKES",
  "confidence": 0.6087,
  "top_k": [
    {"label": "SERVICE BRAKES", "confidence": 0.6087},
    {"label": "ELECTRICAL SYSTEM", "confidence": 0.1296},
    {"label": "POWER TRAIN", "confidence": 0.0718}
  ],
  "explanation": [
    {"term": "brake", "contribution": 0.6474},
    {"term": "not stop", "contribution": 0.2595}
  ],
  "explanation_method": "shap",
  "model_backend": "baseline",
  "latency_ms": 14.7
}
```

Invalid input (empty/too short/too long text) returns `422` with a `detail` message.

## `POST /api/v1/batch-predict`

Same shape as `/predict`, applied to a list (max 500 items):

```bash
curl -X POST http://localhost:8000/api/v1/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Engine stalls at highway speed.", "Seat belt did not retract."], "explain": false}'
```

```json
{"predictions": [{"predicted_label": "ENGINE", ...}, {"predicted_label": "SEAT BELTS", ...}]}
```

## `POST /api/v1/retrain`

Triggers a full baseline retrain (TF-IDF + LogisticRegression/LinearSVC) on `data/processed/{train,val,test}.parquet` as a background task. Returns immediately.

```bash
curl -X POST http://localhost:8000/api/v1/retrain
```
```json
{"status": "started", "message": "Retrain job started in the background"}
```

A second call while a retrain is in progress returns `{"status": "already_running", ...}` instead of starting a duplicate job. There is no persistent job-tracking store — `in_progress` is a single in-process flag, meaningful only for the lifetime of one running API process. On completion, the newly promoted model is picked up automatically (the predictor cache is cleared server-side) — no restart needed.

Requires `data/processed/` to already exist (`python -m src.data.pipeline`).

## Error handling

- `422 Unprocessable Entity` — invalid request body (pydantic validation) or invalid complaint text (empty/too short/too long)
- `500 Internal Server Error` — unhandled exception; logged server-side with the request path, generic message returned to the client (see `app/main.py`'s exception handler)
