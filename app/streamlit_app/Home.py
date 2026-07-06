"""Dashboard — landing page for the Streamlit app.
Run with: `streamlit run app/streamlit_app/Home.py`
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from theme import ACCENT_CYAN, ACCENT_PINK, ACCENT_PURPLE, badge, hero_header, inject_global_css, section_label
from utils import load_production_metadata, load_transformer_metadata, transformer_display_name

st.set_page_config(page_title="Vehicle Complaint Classifier", page_icon="🧠", layout="wide")
inject_global_css()

metadata = load_production_metadata()

if metadata is None:
    st.error(
        "No production model found at `models/production/`. Run "
        "`python -m src.data.pipeline` then `python -m src.training.train` first."
    )
    st.stop()

test_metrics = metadata["test_metrics"]
classes = metadata["classes"]

hero_header(
    "Vehicle Complaint Intelligence",
    "NLP system that classifies free-text NHTSA vehicle complaints into 28 fault categories — "
    "trained on 271,517 real safety-complaint records, benchmarked across three model backends.",
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Macro-F1", f"{test_metrics['macro_f1']:.3f}")
col2.metric("Test Accuracy", f"{test_metrics['accuracy']:.1%}")
col3.metric("Fault Categories", len(classes))
col4.metric("Active Backend", metadata["model_type"].replace("_", " ").title())

st.divider()

left, right = st.columns([1.1, 1])

with left:
    section_label("What this system does")
    st.markdown(
        f"""
        <div style="line-height: 1.9; color: var(--text-primary); font-size: 0.98rem;">
        Predicts the NHTSA fault-category taxonomy (e.g. <code>ENGINE</code>, <code>SERVICE BRAKES</code>,
        <code>AIR BAGS</code>) from a free-text complaint description.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 0.7rem'></div>", unsafe_allow_html=True)
    features = [
        ("Confidence + top-3 alternatives", ACCENT_CYAN),
        ("Word-level SHAP explanations", ACCENT_PURPLE),
        ("Batch prediction for CSV uploads", ACCENT_PINK),
        ("Config-swappable model backend", ACCENT_CYAN),
    ]
    for text, color in features:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.55rem;">
                <span style="width:6px; height:6px; border-radius:50%; background:{color};
                    box-shadow: 0 0 8px {color};"></span>
                <span style="color: var(--text-primary); font-size: 0.94rem;">{text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)
    st.caption("See the **About** page for the full data audit and architecture writeup.")

with right:
    section_label("Fault category coverage")
    class_df = pd.DataFrame({"category": classes})
    st.dataframe(class_df, width="stretch", height=300, hide_index=True)

transformer_metadata = load_transformer_metadata()
if transformer_metadata:
    st.divider()
    section_label("Available model backends")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        with st.container(border=True):
            st.markdown(f"**Baseline** {badge('PRODUCTION', ACCENT_CYAN)}", unsafe_allow_html=True)
            st.markdown(
                f"<span style='font-family: JetBrains Mono, monospace; color: var(--text-primary);'>"
                f"Macro-F1 {test_metrics['macro_f1']:.3f} &nbsp;·&nbsp; Accuracy {test_metrics['accuracy']:.1%}"
                f"</span>",
                unsafe_allow_html=True,
            )
    with bcol2:
        with st.container(border=True):
            t_metrics = transformer_metadata["test_metrics"]
            st.markdown(
                f"**{transformer_display_name(transformer_metadata)}** {badge('BENCHMARK', ACCENT_PURPLE)}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span style='font-family: JetBrains Mono, monospace; color: var(--text-primary);'>"
                f"Macro-F1 {t_metrics['macro_f1']:.3f} &nbsp;·&nbsp; Accuracy {t_metrics['accuracy']:.1%}"
                f"</span>",
                unsafe_allow_html=True,
            )
            if transformer_metadata.get("is_full_dataset_run"):
                st.caption("Trained on the full dataset on GPU — see Model Metrics page for details")
            else:
                st.caption("Trained on a data subsample — see Model Metrics page for details")

st.divider()
section_label("Jump in")
nav1, nav2, nav3 = st.columns(3)
with nav1:
    st.page_link("pages/1_Single_Prediction.py", label="Try a single prediction", icon="🔍")
with nav2:
    st.page_link("pages/2_Compare_Backends.py", label="Compare baseline vs. transformer", icon="⚖️")
with nav3:
    st.page_link("pages/3_Batch_Prediction.py", label="Run batch predictions on a CSV", icon="📄")
