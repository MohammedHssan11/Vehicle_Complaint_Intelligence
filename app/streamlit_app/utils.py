"""Shared helpers for the Streamlit pages — thin wrappers around
`src.services.classification_service` with Streamlit-appropriate caching
(`st.cache_resource` for the loaded model, `st.cache_data` for on-disk
artifacts) so navigating between pages doesn't reload anything."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.settings import paths
from src.services.classification_service import ClassificationService


@st.cache_resource(show_spinner="Loading model...")
def get_service(backend: str | None = None) -> ClassificationService:
    return ClassificationService(backend=backend)


@st.cache_data(show_spinner=False)
def load_production_metadata() -> dict | None:
    path = paths.production_model_dir / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_transformer_metadata() -> dict | None:
    path = paths.models_dir / "transformer" / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def transformer_display_name(metadata: dict) -> str:
    """Human-readable base model name from metadata's `model_type`
    (e.g. "microsoft/deberta-v3-base-finetuned" -> "deberta-v3-base") — the
    transformer backend is swappable (`TRANSFORMER_BASE_MODEL`), so the UI
    can't hardcode "DistilBERT"."""
    model_type = metadata.get("model_type", "transformer")
    name = model_type.removesuffix("-finetuned").rsplit("/", 1)[-1]
    return name


def _latest_baseline_run_dir() -> Path | None:
    if not paths.artifacts_dir.exists():
        return None
    candidates = [
        d for d in paths.artifacts_dir.iterdir() if d.is_dir() and d.name != "transformer_runs"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name)


@st.cache_data(show_spinner=False)
def load_latest_baseline_artifacts() -> dict:
    run_dir = _latest_baseline_run_dir()
    if run_dir is None:
        return {}

    result = {"run_dir": str(run_dir)}
    report_path = run_dir / "classification_report.txt"
    if report_path.exists():
        result["classification_report"] = report_path.read_text()

    cm_path = run_dir / "confusion_matrix.csv"
    if cm_path.exists():
        result["confusion_matrix"] = pd.read_csv(cm_path, index_col=0)

    confused_path = run_dir / "most_confused_pairs.csv"
    if confused_path.exists():
        result["most_confused_pairs"] = pd.read_csv(confused_path)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        result["metrics"] = json.loads(metrics_path.read_text())

    return result
