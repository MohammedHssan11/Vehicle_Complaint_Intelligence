from __future__ import annotations

import streamlit as st

from utils import (
    get_service,
    load_latest_baseline_artifacts,
    load_production_metadata,
    load_transformer_metadata,
    transformer_display_name,
)

st.set_page_config(page_title="Model Metrics", page_icon="📊", layout="wide")
st.title("📊 Model Metrics")

metadata = load_production_metadata()
artifacts = load_latest_baseline_artifacts()

st.subheader("Production model (baseline)")
if metadata:
    test_metrics = metadata["test_metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Macro-F1", f"{test_metrics['macro_f1']:.3f}")
    c2.metric("Weighted-F1", f"{test_metrics['weighted_f1']:.3f}")
    c3.metric("Accuracy", f"{test_metrics['accuracy']:.3f}")
    c4.metric("Macro-Recall", f"{test_metrics['macro_recall']:.3f}")
    if "lenient_accuracy" in test_metrics:
        c5.metric(
            "Lenient accuracy",
            f"{test_metrics['lenient_accuracy']:.3f}",
            help="Credits a prediction if it matches ANY listed component, not just the "
            "first-listed one used for training/scoring — see docs/model_benchmark.md's "
            "\"honest evaluation\" section.",
        )

if artifacts.get("metrics"):
    m = artifacts["metrics"]
    st.caption(
        f"Train time: {m['train_time_seconds']:.1f}s | "
        f"Inference latency: {m['inference_latency_ms_per_sample']:.3f} ms/sample | "
        f"Majority-class baseline accuracy: {m['majority_class_baseline_accuracy']:.3f}"
    )

if "classification_report" in artifacts:
    st.subheader("Per-class classification report (held-out test set)")
    st.code(artifacts["classification_report"], language=None)

if "confusion_matrix" in artifacts:
    st.subheader("Confusion matrix")
    st.dataframe(
        artifacts["confusion_matrix"].style.background_gradient(cmap="Blues", axis=None),
        width='stretch',
    )

if "most_confused_pairs" in artifacts:
    st.subheader("Most confused label pairs")
    st.caption("Where the model's errors cluster — genuine semantic overlap (e.g. ENGINE vs POWER TRAIN), not label-taxonomy artifacts.")
    st.dataframe(artifacts["most_confused_pairs"], width='stretch', hide_index=True)

transformer_metadata = load_transformer_metadata()
if transformer_metadata:
    st.divider()
    transformer_name = transformer_display_name(transformer_metadata)
    st.subheader(f"{transformer_name} benchmark")
    if transformer_metadata.get("is_full_dataset_run"):
        latency_note = ""
        if artifacts.get("metrics") and transformer_metadata.get("inference_latency_ms_per_sample"):
            ratio = (
                transformer_metadata["inference_latency_ms_per_sample"]
                / artifacts["metrics"]["inference_latency_ms_per_sample"]
            )
            latency_note = f"but is ~{ratio:.0f}x slower per prediction and "
        st.success(
            f"Trained on the full {transformer_metadata.get('n_train_examples', 188056):,}-row training "
            f"set on {transformer_metadata.get('device', 'GPU').upper()} — a fair comparison against the "
            f"baseline. It scores higher on accuracy/macro-F1 {latency_note or 'but '}has "
            "no word-level explanation yet (see docs/model_benchmark.md). Still not promoted to "
            "`models/production/` by default — that's a latency/explainability tradeoff, not an accuracy one."
        )
    else:
        st.warning(
            "Trained on a stratified subsample (not the full training set) due to this "
            "environment's CPU-only PyTorch — see docs/model_benchmark.md for the full "
            "caveat and GPU scale-up instructions."
        )
    t_metrics = transformer_metadata["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro-F1", f"{t_metrics['macro_f1']:.3f}")
    c2.metric("Weighted-F1", f"{t_metrics['weighted_f1']:.3f}")
    c3.metric("Accuracy", f"{t_metrics['accuracy']:.3f}")
    if "lenient_accuracy" in t_metrics:
        c4.metric(
            "Lenient accuracy",
            f"{t_metrics['lenient_accuracy']:.3f}",
            help="Credits a prediction if it matches ANY listed component, not just the "
            "first-listed one used for training/scoring — see docs/model_benchmark.md's "
            "\"honest evaluation\" section.",
        )

st.divider()
st.subheader("Live serving metrics (this app process)")
service = get_service("baseline")
live_metrics = service.get_metrics()
c1, c2, c3 = st.columns(3)
c1.metric("Predictions served", live_metrics["total_predictions"])
c2.metric("Avg latency", f"{live_metrics['avg_latency_ms']:.1f} ms" if live_metrics["avg_latency_ms"] else "n/a")
c3.metric("Uptime", f"{live_metrics['uptime_seconds']:.0f}s")
if live_metrics["class_distribution"]:
    st.bar_chart(live_metrics["class_distribution"])
