from __future__ import annotations

import pandas as pd
import streamlit as st

from theme import hero_header, inject_global_css, section_label
from utils import get_service

st.set_page_config(page_title="Batch Prediction", page_icon="📄", layout="wide")
inject_global_css()
hero_header("Batch prediction", "Classify many complaints at once — upload a CSV or paste text directly.")

MAX_ROWS = 500

input_mode = st.radio("Input method", ["Upload CSV", "Paste text (one complaint per line)"], horizontal=True)

texts: list[str] = []

if input_mode == "Upload CSV":
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if df.empty:
            st.warning("Uploaded CSV is empty.")
        else:
            text_col = st.selectbox("Column containing complaint text", df.columns.tolist())
            texts = df[text_col].dropna().astype(str).tolist()
else:
    pasted = st.text_area("One complaint per line", height=200)
    texts = [line.strip() for line in pasted.splitlines() if line.strip()]

if texts:
    st.caption(f"{len(texts)} complaint(s) loaded" + (f" — showing first {MAX_ROWS}" if len(texts) > MAX_ROWS else ""))
    texts = texts[:MAX_ROWS]

explain = st.checkbox("Include word-level explanation (slower for large batches)", value=False)

if st.button("Run batch prediction", type="primary", disabled=not texts):
    service = get_service("baseline")
    with st.spinner(f"Classifying {len(texts)} complaints..."):
        results = service.classify_batch(texts, explain=explain)

    rows = []
    for text, result in zip(texts, results):
        row = {
            "text": text,
            "predicted_label": result.predicted_label,
            "confidence": result.confidence,
            "top_2": result.top_k[1]["label"] if len(result.top_k) > 1 else "",
            "top_2_confidence": result.top_k[1]["confidence"] if len(result.top_k) > 1 else None,
            "latency_ms": result.latency_ms,
        }
        if explain and result.explanation:
            row["top_terms"] = ", ".join(t["term"] for t in result.explanation[:5])
        rows.append(row)

    results_df = pd.DataFrame(rows)
    st.divider()
    section_label("Results")
    st.dataframe(results_df, width='stretch', hide_index=True)

    section_label("Predicted category distribution")
    st.bar_chart(results_df["predicted_label"].value_counts())

    st.download_button(
        "Download results as CSV",
        results_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_predictions.csv",
        mime="text/csv",
    )
