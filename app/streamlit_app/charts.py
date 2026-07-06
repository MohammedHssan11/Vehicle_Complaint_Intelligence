"""Plotly chart helpers — replaces Streamlit's native bar/line charts with
styled figures matching the app's theme (.streamlit/config.toml)."""
from __future__ import annotations

import plotly.graph_objects as go

PRIMARY = "#2563EB"
POSITIVE = "#16A34A"
NEGATIVE = "#DC2626"
NEUTRAL = "#94A3B8"


def confidence_gauge(confidence: float, label: str) -> go.Figure:
    if confidence >= 0.7:
        bar_color = POSITIVE
    elif confidence >= 0.4:
        bar_color = "#D97706"
    else:
        bar_color = NEGATIVE

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": label, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": bar_color},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#FEE2E2"},
                    {"range": [40, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#DCFCE7"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def top_k_bar_chart(top_k: list[dict]) -> go.Figure:
    labels = [item["label"] for item in top_k]
    confidences = [item["confidence"] * 100 for item in top_k]
    colors = [PRIMARY if i == 0 else NEUTRAL for i in range(len(labels))]

    fig = go.Figure(
        go.Bar(
            x=confidences,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{c:.1f}%" for c in confidences],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis=dict(autorange="reversed"),
        height=220,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis_range=[0, max(confidences) * 1.25],
    )
    return fig


def explanation_bar_chart(explanation: list[dict], top_n: int = 10) -> go.Figure:
    items = sorted(explanation[:top_n], key=lambda e: e["contribution"])
    terms = [item["term"] for item in items]
    contributions = [item["contribution"] for item in items]
    colors = [POSITIVE if c >= 0 else NEGATIVE for c in contributions]

    fig = go.Figure(
        go.Bar(
            x=contributions,
            y=terms,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        xaxis_title="Contribution (SHAP value)",
        height=max(220, 28 * len(terms)),
        margin=dict(l=10, r=10, t=10, b=30),
    )
    return fig
