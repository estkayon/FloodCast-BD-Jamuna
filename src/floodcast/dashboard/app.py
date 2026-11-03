# src/floodcast/dashboard/app.py

"""Multi-page Streamlit dashboard for FloodCast BD."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="FloodCast BD",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Slate + Blue Professional Theme ----
st.markdown("""
<style>
    /* Hide sidebar */
    [data-testid="stSidebar"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    button[kind="header"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Background */
    .stApp {
        background-color: #0f172a;
    }

    /* ---- Navbar ---- */
    .navbar {
        background: #1e293b;
        padding: 14px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 0 0 12px 12px;
        margin: -1rem -1rem 24px -1rem;
        border-bottom: 1px solid #334155;
    }
    .navbar-brand {
        font-size: 20px;
        font-weight: 700;
        color: #f1f5f9;
    }
    .navbar-brand span {
        font-weight: 400;
        color: #94a3b8;
        font-size: 13px;
        margin-left: 10px;
    }
    .navbar-links {
        display: flex;
        gap: 4px;
    }
    .nav-btn {
        color: #94a3b8;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
    }
    .nav-btn.active {
        background: #334155;
        color: #f1f5f9;
        font-weight: 600;
    }

    /* ---- Cards ---- */
    .card {
        background: #1e293b;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 12px;
        border: 1px solid #334155;
    }
    .card-accent {
        border-left: 3px solid #3b82f6;
    }

    /* ---- Risk cards ---- */
    .risk-safe {
        background: #1e293b;
        border-left: 3px solid #22c55e;
        border: 1px solid #334155;
        border-left: 3px solid #22c55e;
        border-radius: 10px;
        padding: 24px;
    }
    .risk-caution {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 3px solid #eab308;
        border-radius: 10px;
        padding: 24px;
    }
    .risk-warning {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 3px solid #f97316;
        border-radius: 10px;
        padding: 24px;
    }
    .risk-danger {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 3px solid #ef4444;
        border-radius: 10px;
        padding: 24px;
    }

    /* ---- Typography ---- */
    .label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .big-num {
        font-size: 32px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 4px 0;
    }
    h1 { color: #f1f5f9 !important; font-weight: 700 !important; }
    h2, h3 { color: #e2e8f0 !important; }

    /* ---- Streamlit text ---- */
    .stMarkdown, .stMarkdown p, .stCaption {
        color: #cbd5e1 !important;
    }
    .stSelectbox label, .stRadio label {
        color: #cbd5e1 !important;
    }

    /* ---- Card text ---- */
    .card strong, .risk-safe strong, .risk-caution strong,
    .risk-warning strong, .risk-danger strong {
        color: #f1f5f9;
    }
    .card p, .card td, .card li, .card span,
    .card ol li, .card ul li {
        color: #cbd5e1;
    }
    .risk-safe div, .risk-caution div,
    .risk-warning div, .risk-danger div {
        color: #cbd5e1 !important;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* ---- Selectbox ---- */
    .stSelectbox > div > div {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
        border-color: #334155 !important;
    }

    /* ---- Footer banner ---- */
    .warning-banner {
        background: #1e293b;
        color: #94a3b8;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 12px;
        text-align: center;
        margin-top: 24px;
        border: 1px solid #334155;
    }

    /* ---- Charts container ---- */
    .stPlotlyChart {
        background: #1e293b;
        border-radius: 10px;
        padding: 8px;
        border: 1px solid #334155;
    }

     /* ---- Misc ---- */
    hr { border-color: #f1f5f9 !important; }
    .stAlert {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #f1f5f9 !important;
    }
    .stAlert p, .stAlert div, .stAlert span {
        color: #f1f5f9 !important;
    }
    .streamlit-expanderHeader { color: #f1f5f9 !important; }
    .streamlit-expanderContent { color: #f1f5f9 !important; }
    .stExpander { border-color: #f1f5f9 !important; }
    .stExpander summary { color: #f1f5f9 !important; }
    .stExpander summary span { color: #f1f5f9 !important; }
    .stExpander summary p { color: #f1f5f9 !important; }
    .stExpander details { color: #f1f5f9 !important; }
    [data-testid="stExpander"] summary { color: #f1f5f9 !important; }
    [data-testid="stExpander"] summary span { color: #f1f5f9 !important; }
    [data-testid="stExpanderDetails"] { color: #f1f5f9 !important; }
    [data-testid="stExpanderDetails"] p { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ---- Navbar ----
if "page" not in st.session_state:
    st.session_state.page = "Current Risk"

PAGES = ["Current Risk", "Historical", "Model Comparison", "About"]

nav_links = ""
for p in PAGES:
    active = "active" if p == st.session_state.page else ""
    nav_links += f'<span class="nav-btn {active}">{p}</span>'

st.markdown(f"""
<div class="navbar">
    <div class="navbar-brand">
        🌊 FloodCast BD
        <span>Jamuna River · Sirajganj</span>
    </div>
    <div class="navbar-links">
        {nav_links}
    </div>
</div>
""", unsafe_allow_html=True)

cols = st.columns(len(PAGES))
for i, p in enumerate(PAGES):
    with cols[i]:
        if st.button(
            p,
            key=f"nav_{p}",
            use_container_width=True,
            type="primary" if p == st.session_state.page else "secondary",
        ):
            st.session_state.page = p
            st.rerun()

st.divider()

if st.session_state.page == "Current Risk":
    from floodcast.dashboard.views.current_risk import render
    render()
elif st.session_state.page == "Historical":
    from floodcast.dashboard.views.historical import render
    render()
elif st.session_state.page == "Model Comparison":
    from floodcast.dashboard.views.model_comparison import render
    render()
elif st.session_state.page == "About":
    from floodcast.dashboard.views.about import render
    render()

st.markdown("""
<div class="warning-banner">
    ⚠️ Research prototype — Not for emergency decisions.
    Always follow FFWC and local authority instructions.
</div>
""", unsafe_allow_html=True)
