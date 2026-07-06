"""Plotly chart helpers — dark, glass-styled figures matching theme.py's
palette. All charts use transparent backgrounds so they blend into the
glass cards/page background rather than showing a light plotly canvas."""
from __future__ import annotations

import plotly.graph_objects as go

CYAN = "#22D3EE"
PURPLE = "#A78BFA"
PINK = "#F472B6"
SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#F87171"
NEUTRAL = "#4B5566"

TEXT_PRIMARY = "#E7ECF3"
TEXT_SECONDARY = "#94A3B8"
GRID_COLOR = "rgba(148, 163, 184, 0.12)"

_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=13),
)


def confidence_gauge(confidence: float, label: str) -> go.Figure:
    if confidence >= 0.7:
        bar_color = SUCCESS
    elif confidence >= 0.4:
        bar_color = WARNING
    else:
        bar_color = DANGER

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 36, "color": TEXT_PRIMARY, "family": "JetBrains Mono, monospace"}},
            title={"text": label, "font": {"size": 15, "color": TEXT_SECONDARY, "family": "Inter, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": TEXT_SECONDARY, "tickfont": {"color": TEXT_SECONDARY, "size": 10}},
                "bar": {"color": bar_color, "thickness": 0.75},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 1,
                "bordercolor": "rgba(148, 163, 184, 0.14)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(248, 113, 113, 0.12)"},
                    {"range": [40, 70], "color": "rgba(251, 191, 36, 0.12)"},
                    {"range": [70, 100], "color": "rgba(52, 211, 153, 0.12)"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10), **_BASE_LAYOUT)
    return fig


def top_k_bar_chart(top_k: list[dict]) -> go.Figure:
    labels = [item["label"] for item in top_k]
    confidences = [item["confidence"] * 100 for item in top_k]
    colors = [CYAN if i == 0 else NEUTRAL for i in range(len(labels))]

    fig = go.Figure(
        go.Bar(
            x=confidences,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{c:.1f}%" for c in confidences],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono, monospace"),
        )
    )
    fig.update_layout(
        xaxis_title="Confidence (%)",
        yaxis=dict(autorange="reversed", gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        height=220,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis_range=[0, max(confidences) * 1.25],
        **_BASE_LAYOUT,
    )
    return fig


def explanation_bar_chart(explanation: list[dict], top_n: int = 10) -> go.Figure:
    items = sorted(explanation[:top_n], key=lambda e: e["contribution"])
    terms = [item["term"] for item in items]
    contributions = [item["contribution"] for item in items]
    colors = [SUCCESS if c >= 0 else DANGER for c in contributions]

    fig = go.Figure(
        go.Bar(
            x=contributions,
            y=terms,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
        )
    )
    fig.update_layout(
        xaxis_title="Contribution (SHAP value)",
        yaxis=dict(gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor="rgba(148, 163, 184, 0.3)"),
        height=max(220, 28 * len(terms)),
        margin=dict(l=10, r=10, t=10, b=30),
        **_BASE_LAYOUT,
    )
    return fig
