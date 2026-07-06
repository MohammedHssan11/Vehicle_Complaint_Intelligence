"""Minimal production monitoring (Phase 19): counts, latency, and class
distribution for every prediction served, plus a durable JSONL audit log.

Deliberately simple for v1 — in-memory counters reset on process restart
(the JSONL file is the durable record; a real deployment would back
`/metrics` with a time-series store, not process memory). This is enough to
demonstrate the monitoring hook the API exposes without building a
full metrics stack for a portfolio project.
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from src.config.settings import paths


class PredictionLogger:
    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path or (paths.logs_dir / "predictions.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = 0
        self._latency_sum = 0.0
        self._class_counts: Counter = Counter()
        self._start_time = time.time()

    def log(self, text: str, predicted_label: str, confidence: float, model_backend: str, latency_ms: float) -> None:
        with self._lock:
            self._count += 1
            self._latency_sum += latency_ms
            self._class_counts[predicted_label] += 1

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text_preview": text[:200],
            "predicted_label": predicted_label,
            "confidence": confidence,
            "model_backend": model_backend,
            "latency_ms": latency_ms,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def get_metrics(self) -> dict:
        with self._lock:
            avg_latency = (self._latency_sum / self._count) if self._count else None
            return {
                "total_predictions": self._count,
                "avg_latency_ms": round(avg_latency, 3) if avg_latency is not None else None,
                "class_distribution": dict(self._class_counts),
                "uptime_seconds": round(time.time() - self._start_time, 1),
            }


@lru_cache(maxsize=1)
def get_prediction_logger() -> PredictionLogger:
    return PredictionLogger()
