from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import confidence_gauge, explanation_bar_chart, top_k_bar_chart
from src.inference.predictor import InvalidInputError
from theme import hero_header, inject_global_css, section_label
from utils import get_service, load_transformer_metadata, transformer_display_name

st.set_page_config(page_title="Single Prediction", page_icon="🔍", layout="wide")
inject_global_css()
hero_header("Single prediction", "Classify a complaint description and see the model's confidence and reasoning.")

transformer_metadata = load_transformer_metadata()
backend_options = ["baseline"]
if transformer_metadata:
    backend_options.append("transformer")

with st.container(border=True):
    col_backend, col_explain = st.columns([2, 1])
    with col_backend:
        transformer_name = transformer_display_name(transformer_metadata) if transformer_metadata else "transformer"
        backend = st.selectbox(
            "Model backend",
            backend_options,
            help="baseline = TF-IDF + Linear SVC (fast, full explainability). "
            f"transformer = fine-tuned {transformer_name} (see Model Metrics page for the honest benchmark comparison).",
        )
    with col_explain:
        st.write("")  # vertical alignment spacer
        explain = st.checkbox(
            "Word-level explanation", value=(backend == "baseline"), disabled=(backend != "baseline")
        )

    text = st.text_area(
        "Complaint description",
        height=140,
        placeholder="e.g. The brake pedal did not engage properly and the car would not stop in time.",
    )
    submitted = st.button("Classify", type="primary", width="stretch")

if submitted:
    if not text.strip():
        st.warning("Enter a complaint description first.", icon="⚠️")
    else:
        service = get_service(backend)
        try:
            with st.spinner("Classifying..."):
                result = service.classify(text, explain=explain)
        except InvalidInputError as e:
            st.error(str(e), icon="🚫")
        else:
            st.divider()
            gauge_col, top3_col = st.columns([1, 1.4])

            with gauge_col:
                st.plotly_chart(
                    confidence_gauge(result.confidence, result.predicted_label),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.caption(f"Model: `{result.model_backend}` · Latency: {result.latency_ms:.1f} ms")

            with top3_col:
                section_label("Top 3 categories")
                st.plotly_chart(
                    top_k_bar_chart(result.top_k), width="stretch", config={"displayModeBar": False}
                )

            if result.explanation:
                st.subheader("What drove this prediction")
                st.caption(
                    f"Attribution method: `{result.explanation_method}` — green bars support the "
                    "prediction, red bars argue against it."
                )
                st.plotly_chart(
                    explanation_bar_chart(result.explanation), width="stretch", config={"displayModeBar": False}
                )
                with st.expander("Raw explanation values"):
                    st.dataframe(pd.DataFrame(result.explanation), width="stretch", hide_index=True)
            elif backend == "transformer":
                st.info(
                    "Word-level explanation is only implemented for the baseline backend "
                    "(see docs/model_benchmark.md).",
                    icon="ℹ️",
                )
