# src/floodcast/dashboard/app.py

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.chdir(Path(__file__).resolve().parent.parent.parent.parent)

import streamlit as st

st.set_page_config(
    page_title="FloodCast BD",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    /* ---- Hide sidebar & chrome ---- */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    button[kind="header"],
    #MainMenu, footer, header {
        display: none !important;
        visibility: hidden !important;
    }

    /* ---- Dark background ---- */
    html, body, .stApp, .main,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .block-container,
    [data-testid="stVerticalBlock"] {
        background-color: #0F172A !important;
    }

    /* ---- Header ---- */
    .fc-header {
        text-align: center;
        padding: 32px 0 8px 0;
    }
    .fc-header h1 {
        font-size: 28px;
        font-weight: 800;
        color: #F1F5F9;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .fc-header h1 span { color: #F97316; }
    .fc-header p {
        font-size: 12px;
        color: #64748B;
        margin: 6px 0 0 0;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* ---- Tab pills ---- */
    [data-testid="stTabs"] > div:first-child {
        display: flex !important;
        justify-content: center !important;
        gap: 6px !important;
        border-bottom: none !important;
        padding: 16px 0 24px 0 !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"] {
        background: rgba(30,41,59,0.6) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        transition: all 0.25s ease !important;
        white-space: nowrap !important;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        background: rgba(249,115,22,0.08) !important;
        border-color: #F97316 !important;
        color: #F97316 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #F97316, #EA580C) !important;
        border-color: #F97316 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(249,115,22,0.3) !important;
    }
    [data-testid="stTabs"] [role="tablist"] > div:last-child {
        display: none !important;
    }

    /* ---- Cards ---- */
    .ocard {
        background: rgba(30,41,59,0.7);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 12px;
        border: 1px solid #334155;
    }
    .ocard-glow {
        box-shadow: 0 0 20px rgba(249,115,22,0.08);
        border-color: #475569;
    }
    .ocard-accent { border-left: 3px solid #F97316; }

    /* ---- Metrics ---- */
    .big-num {
        font-size: 34px;
        font-weight: 700;
        color: #F1F5F9;
        margin: 4px 0;
        letter-spacing: -0.5px;
    }
    .sub-label {
        font-size: 10px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 12px;
        color: #64748B;
        margin-top: 2px;
    }

    /* ---- Key finding ---- */
    .finding {
        background: linear-gradient(135deg, rgba(249,115,22,0.12), rgba(30,41,59,0.8));
        border: 1px solid #9A3412;
        border-radius: 14px;
        padding: 18px 22px;
        margin: 12px 0;
        font-size: 14px;
        color: #F1F5F9;
        line-height: 1.7;
    }
    .finding strong { color: #FB923C; }

    /* ---- Gradient line on cards ---- */
    .grad-line {
        height: 3px;
        border-radius: 14px 14px 0 0;
        margin: -20px -24px 16px -24px;
    }

    /* ---- Section header ---- */
    .section-title {
        font-size: 13px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 28px 0 14px 0;
    }

    /* ---- Typography ---- */
    h1, h2, h3 { color: #F1F5F9 !important; }
    .stMarkdown, .stMarkdown p { color: #CBD5E1 !important; }
    .stCaption { color: #64748B !important; }
    .stSelectbox label, .stRadio label { color: #94A3B8 !important; }

    /* ---- Widgets ---- */
    .stSelectbox > div > div {
        background-color: #1E293B !important;
        color: #F1F5F9 !important;
        border-color: #334155 !important;
        border-radius: 8px !important;
    }

    /* ---- Plotly wrapper ---- */
    .stPlotlyChart {
        background: rgba(30,41,59,0.5);
        border-radius: 14px;
        padding: 12px;
        border: 1px solid #334155;
    }

    /* ---- Expander ---- */
    [data-testid="stExpander"] {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpanderDetails"],
    [data-testid="stExpanderDetails"] p {
        color: #F1F5F9 !important;
    }

    /* ---- Alert ---- */
    .stAlert {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F1F5F9 !important;
        border-radius: 12px !important;
    }

    /* ---- Dataframe ---- */
    [data-testid="stDataFrame"] {
        background: #1E293B !important;
        border-radius: 10px;
    }

    hr { border-color: #1E293B !important; }

    /* ---- Footer ---- */
    .fc-footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        font-size: 11px;
        color: #475569;
        border-top: 1px solid #1E293B;
    }
    .fc-footer a { color: #F97316; text-decoration: none; }

    .stSpinner > div { color: #F97316 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown(
    """
<div class="fc-header">
    <h1>🌊 Flood<span>Cast</span> BD</h1>
    <p>Jamuna River Basin · Physics-Informed Flood Nowcasting</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---- Two research tabs ----
tab1, tab2 = st.tabs(["📊 Model Performance", "📈 Historical"])

with tab1:
    from floodcast.dashboard.views.model_performance import render as render_perf

    render_perf()

with tab2:
    from floodcast.dashboard.views.historical import render as render_hist

    render_hist()

# ---- Footer ----
st.markdown(
    """
<div class="fc-footer">
    Research prototype · ERA5-Land (ECMWF) · Not for emergency decisions<br>
    Built by Md. Estiak Rahman Ayon ·
    <a href="https://github.com/estkayon/FloodCast-BD-Jamuna" target="_blank">GitHub</a>
</div>
""",
    unsafe_allow_html=True,
)



