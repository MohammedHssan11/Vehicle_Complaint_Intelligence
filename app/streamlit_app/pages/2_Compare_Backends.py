from __future__ import annotations

import streamlit as st

from charts import confidence_gauge, top_k_bar_chart
from src.inference.predictor import InvalidInputError
from utils import get_service, load_production_metadata, load_transformer_metadata, transformer_display_name

st.set_page_config(page_title="Compare Backends", page_icon="⚖️", layout="wide")
st.title("⚖️ Compare Backends")
st.caption(
    "Run the same complaint through both model backends side by side — demonstrates the "
    "config-swappable serving architecture (`src/inference/predictor.py`), not just a claim in the docs."
)

baseline_meta = load_production_metadata()
transformer_meta = load_transformer_metadata()

if not transformer_meta:
    st.warning(
        "No transformer model found at `models/transformer/`. Run "
        "`python -m src.training.train_transformer` first to enable this page.",
        icon="⚠️",
    )
    st.stop()

transformer_name = transformer_display_name(transformer_meta)

b1, b2 = st.columns(2)
b1.metric("Baseline test macro-F1", f"{baseline_meta['test_metrics']['macro_f1']:.3f}")
b2.metric(f"{transformer_name} test macro-F1", f"{transformer_meta['test_metrics']['macro_f1']:.3f}")

text = st.text_area(
    "Complaint description",
    height=140,
    placeholder="e.g. The airbag deployed unexpectedly while driving at low speed without any collision.",
)

if st.button("Compare", type="primary", width="stretch"):
    if not text.strip():
        st.warning("Enter a complaint description first.", icon="⚠️")
    else:
        baseline_service = get_service("baseline")
        transformer_service = get_service("transformer")
        try:
            with st.spinner("Classifying with both backends..."):
                baseline_result = baseline_service.classify(text, explain=True)
                transformer_result = transformer_service.classify(text, explain=False)
        except InvalidInputError as e:
            st.error(str(e), icon="🚫")
        else:
            st.divider()
            if baseline_result.predicted_label == transformer_result.predicted_label:
                st.success(f"Both backends agree: **{baseline_result.predicted_label}**", icon="✅")
            else:
                st.info(
                    f"Backends disagree: baseline says **{baseline_result.predicted_label}**, "
                    f"{transformer_name} says **{transformer_result.predicted_label}**",
                    icon="🔀",
                )

            col_baseline, col_transformer = st.columns(2)

            with col_baseline:
                st.subheader("Baseline (TF-IDF + Linear SVC)")
                st.plotly_chart(
                    confidence_gauge(baseline_result.confidence, baseline_result.predicted_label),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.plotly_chart(
                    top_k_bar_chart(baseline_result.top_k), width="stretch", config={"displayModeBar": False}
                )
                st.caption(f"Latency: {baseline_result.latency_ms:.1f} ms · full explainability available")

            with col_transformer:
                st.subheader(f"{transformer_name} (fine-tuned)")
                st.plotly_chart(
                    confidence_gauge(transformer_result.confidence, transformer_result.predicted_label),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.plotly_chart(
                    top_k_bar_chart(transformer_result.top_k), width="stretch", config={"displayModeBar": False}
                )
                st.caption(f"Latency: {transformer_result.latency_ms:.1f} ms · no word-level explanation yet")

            st.caption(
                "See the Model Metrics page and docs/model_benchmark.md for the full accuracy comparison "
                "and why it isn't yet a fair transformer-vs-linear-model comparison."
            )
