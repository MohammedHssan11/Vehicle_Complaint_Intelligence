"""Design system for the Streamlit app — dark, glassmorphic, glow-accented.

Streamlit's own theming (`.streamlit/config.toml`) sets the dark color base
so native widgets (inputs, the dataframe canvas grid, sliders) render
correctly with no contrast bugs. This module layers the rest on top via
injected CSS: typography (Space Grotesk / Inter / JetBrains Mono), glass
cards, glow effects, gradient text, and a few reusable HTML components for
things Streamlit has no native equivalent for (gradient hero text, glass
metric cards, animated section dividers).

Call `inject_global_css()` once per page, near the top, right after
`st.set_page_config()`.
"""
from __future__ import annotations

import streamlit as st

BG_PRIMARY = "#0A0E1A"
BG_SECONDARY = "#131722"
BG_TERTIARY = "#1A2033"
BORDER_SUBTLE = "rgba(148, 163, 184, 0.14)"
BORDER_GLOW = "rgba(34, 211, 238, 0.45)"

ACCENT_CYAN = "#22D3EE"
ACCENT_PURPLE = "#A78BFA"
ACCENT_PINK = "#F472B6"
ACCENT_GRADIENT = f"linear-gradient(135deg, {ACCENT_CYAN} 0%, {ACCENT_PURPLE} 55%, {ACCENT_PINK} 100%)"

SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#F87171"

TEXT_PRIMARY = "#E7ECF3"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#5B6577"


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

        :root {{
            --bg-primary: {BG_PRIMARY};
            --bg-secondary: {BG_SECONDARY};
            --bg-tertiary: {BG_TERTIARY};
            --border-subtle: {BORDER_SUBTLE};
            --border-glow: {BORDER_GLOW};
            --accent-cyan: {ACCENT_CYAN};
            --accent-purple: {ACCENT_PURPLE};
            --accent-pink: {ACCENT_PINK};
            --accent-gradient: {ACCENT_GRADIENT};
            --text-primary: {TEXT_PRIMARY};
            --text-secondary: {TEXT_SECONDARY};
            --text-muted: {TEXT_MUTED};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        /* ---- Page background: subtle fixed radial glow, not flat black ---- */
        .stApp {{
            background:
                radial-gradient(ellipse 900px 600px at 15% -10%, rgba(34, 211, 238, 0.08), transparent 60%),
                radial-gradient(ellipse 900px 700px at 100% 10%, rgba(167, 139, 250, 0.07), transparent 60%),
                radial-gradient(ellipse 700px 500px at 50% 110%, rgba(244, 114, 182, 0.05), transparent 60%),
                var(--bg-primary);
        }}

        /* ---- Headings: Space Grotesk, tight tracking ---- */
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.02em;
            color: var(--text-primary) !important;
        }}
        h1 {{ font-weight: 700 !important; }}
        h2, h3 {{ font-weight: 600 !important; }}

        p, span, div, label {{ color: var(--text-primary); }}
        .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--text-secondary) !important; }}

        /* ---- Sidebar / nav: glass panel ---- */
        [data-testid="stSidebar"] {{
            background: rgba(19, 23, 34, 0.75);
            backdrop-filter: blur(24px);
            border-right: 1px solid var(--border-subtle);
        }}

        /* ---- Top header bar: transparent, blend into page ---- */
        [data-testid="stHeader"] {{
            background: transparent;
        }}

        /* ---- Metric cards: glass + glow on hover ---- */
        [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.025);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 1.1rem 1.2rem 0.9rem;
            transition: border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: var(--border-glow);
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(34, 211, 238, 0.12);
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--text-secondary) !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 600 !important;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* ---- Containers with border=True: glass cards ---- */
        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(16px);
            border-color: var(--border-subtle) !important;
            border-radius: 16px !important;
        }}

        /* ---- Buttons: gradient fill on primary, glow on hover ---- */
        .stButton button, .stFormSubmitButton button {{
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.2s ease;
            border: 1px solid var(--border-subtle);
        }}
        button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
            background: var(--accent-gradient) !important;
            border: none !important;
            color: #0A0E1A !important;
            box-shadow: 0 4px 20px rgba(34, 211, 238, 0.25);
        }}
        button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{
            box-shadow: 0 6px 28px rgba(34, 211, 238, 0.4);
            transform: translateY(-1px);
        }}
        button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {{
            border-color: var(--border-glow) !important;
            color: var(--accent-cyan) !important;
        }}

        /* ---- Inputs: glass fields with glow focus ---- */
        .stTextArea textarea, .stTextInput input, .stSelectbox [data-baseweb="select"] > div {{
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
        }}
        .stTextArea textarea:focus, .stTextInput input:focus {{
            border-color: var(--accent-cyan) !important;
            box-shadow: 0 0 0 1px var(--accent-cyan), 0 0 16px rgba(34, 211, 238, 0.25) !important;
        }}

        /* ---- Dividers: gradient line instead of flat gray ---- */
        hr {{
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent, var(--border-glow), transparent) !important;
            margin: 1.6rem 0 !important;
        }}

        /* ---- Links / page_link: glow underline on hover ---- */
        [data-testid="stPageLink"] {{
            border-radius: 10px;
            transition: background 0.2s ease;
        }}
        [data-testid="stPageLink"]:hover {{
            background: rgba(34, 211, 238, 0.06);
        }}

        /* ---- Expanders: glass ---- */
        [data-testid="stExpander"] {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle) !important;
            border-radius: 12px !important;
        }}

        /* ---- Alerts (success/info/warning/error): glass + colored glow edge ---- */
        [data-testid="stAlertContainer"] {{
            backdrop-filter: blur(12px);
            border-radius: 12px !important;
            border-width: 1px !important;
            border-style: solid !important;
        }}

        /* ---- Scrollbar ---- */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 6px;
        }}

        /* ---- Fade-in on page content ---- */
        [data-testid="stAppViewContainer"] .main .block-container {{
            animation: fadeInUp 0.5s ease both;
        }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_header(title: str, subtitle: str, icon_svg: str | None = None) -> None:
    """Large gradient-text title with a subtitle — used at the top of each page."""
    st.markdown(
        f"""
        <div style="padding: 0.4rem 0 1.6rem;">
            <h1 style="
                font-size: 2.4rem;
                margin: 0;
                background: {ACCENT_GRADIENT};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                display: inline-block;
            ">{title}</h1>
            <p style="
                color: {TEXT_SECONDARY};
                font-size: 1.02rem;
                margin: 0.3rem 0 0;
                max-width: 720px;
            ">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def glass_card_open(accent: str = ACCENT_CYAN) -> str:
    """Returns an opening <div> for a glass card with a colored top accent bar.
    Must be paired with `st.markdown("</div>", unsafe_allow_html=True)` to close,
    or use `glass_card()` as a context manager instead."""
    return f"""
    <div style="
        background: rgba(255,255,255,0.025);
        backdrop-filter: blur(16px);
        border: 1px solid {BORDER_SUBTLE};
        border-top: 2px solid {accent};
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    ">
    """


def badge(text: str, color: str = ACCENT_CYAN) -> str:
    """Inline HTML for a small glowing pill badge. Returns a string — use with st.markdown."""
    return (
        f'<span style="'
        f'display:inline-block; padding: 0.22rem 0.7rem; border-radius: 999px; '
        f'background: {color}22; border: 1px solid {color}55; '
        f'color: {color}; font-size: 0.78rem; font-weight: 600; '
        f'font-family: JetBrains Mono, monospace; letter-spacing: 0.02em;">{text}</span>'
    )


def section_label(text: str) -> None:
    """Small uppercase eyebrow label above a section, e.g. 'MODEL BACKEND'."""
    st.markdown(
        f"""
        <p style="
            color: {ACCENT_CYAN};
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin: 0 0 0.3rem;
            font-family: 'JetBrains Mono', monospace;
        ">{text}</p>
        """,
        unsafe_allow_html=True,
    )
