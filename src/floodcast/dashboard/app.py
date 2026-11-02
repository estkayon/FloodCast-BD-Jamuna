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

# ---- Clean CSS with user color palette ----
st.markdown("""
<style>
    /* Hide default sidebar completely */
    [data-testid="stSidebar"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    button[kind="header"] {display: none;}

    /* Hide hamburger, header, footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Page background — Deep Lagoon */
    .stApp {
        background-color: #0F2A2A;
    }

    /* ---- Top Navbar — Teal Wave ---- */
    .navbar {
        background: linear-gradient(135deg, #00BFA6, #00a08a);
        padding: 14px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 0 0 12px 12px;
        margin: -1rem -1rem 24px -1rem;
        box-shadow: 0 2px 12px rgba(0,191,166,0.3);
    }
    .navbar-brand {
        font-size: 22px;
        font-weight: 700;
        color: #0F2A2A;
        letter-spacing: 0.5px;
    }
    .navbar-brand span {
        font-weight: 400;
        color: #0F2A2A;
        opacity: 0.7;
        font-size: 13px;
        margin-left: 10px;
    }
    .navbar-links {
        display: flex;
        gap: 6px;
    }
    .nav-btn {
        background: transparent;
        color: #0F2A2A;
        border: none;
        padding: 8px 18px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.2s;
        text-decoration: none;
    }
    .nav-btn:hover {
        background: rgba(15,42,42,0.12);
    }
    .nav-btn.active {
        background: rgba(15,42,42,0.2);
        color: #0F2A2A;
        font-weight: 600;
    }

    /* ---- Cards — dark card on dark bg ---- */
    .card {
        background: #163636;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        margin-bottom: 12px;
        border: 1px solid rgba(0,191,166,0.15);
    }
    .card-accent {
        border-left: 4px solid #00BFA6;
    }

    /* ---- Risk cards (semantic — slightly adapted) ---- */
    .risk-safe {
        background: linear-gradient(135deg, #1a3d2a, #163636);
        border-left: 4px solid #4caf50;
        border-radius: 12px;
        padding: 24px;
    }
    .risk-caution {
        background: linear-gradient(135deg, #3d3520, #163636);
        border-left: 4px solid #ffc107;
        border-radius: 12px;
        padding: 24px;
    }
    .risk-warning {
        background: linear-gradient(135deg, #3d2a1a, #163636);
        border-left: 4px solid #ff9800;
        border-radius: 12px;
        padding: 24px;
    }
    .risk-danger {
        background: linear-gradient(135deg, #3d1a1a, #163636);
        border-left: 4px solid #f44336;
        border-radius: 12px;
        padding: 24px;
    }

    /* ---- Labels — Teal Wave ---- */
    .label {
        font-size: 12px;
        color: #00BFA6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* ---- Big numbers — Aqua Silk ---- */
    .big-num {
        font-size: 32px;
        font-weight: 700;
        color: #D9FAF4;
        margin: 4px 0;
    }

    /* ---- Headings — Aqua Silk ---- */
    h1 {
        color: #D9FAF4 !important;
        font-weight: 700 !important;
    }
    h2, h3 {
        color: #D9FAF4 !important;
    }

    /* ---- All Streamlit text — light on dark ---- */
    .stMarkdown, .stMarkdown p, .stCaption {
        color: #D9FAF4 !important;
    }
    .stSelectbox label, .stRadio label {
        color: #D9FAF4 !important;
    }
    .stExpander {
        border-color: #00BFA6 !important;
    }

    /* ---- Card inner text ---- */
    .card strong {
        color: #D9FAF4;
    }
    .card p, .card td, .card li, .card span {
        color: #D9FAF4;
    }

    /* ---- Risk card text (light on dark) ---- */
    .risk-safe, .risk-caution, .risk-warning, .risk-danger {
        color: #D9FAF4;
    }
    .risk-safe div, .risk-caution div,
    .risk-warning div, .risk-danger div {
        color: #D9FAF4 !important;
    }

    /* ---- Streamlit button overrides ---- */
    .stButton > button {
        background-color: #163636 !important;
        color: #D9FAF4 !important;
        border: 1px solid #00BFA6 !important;
        border-radius: 8px !important;
    }
    .stButton > button:hover {
        background-color: #00BFA6 !important;
        color: #0F2A2A !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #00BFA6 !important;
        color: #0F2A2A !important;
        font-weight: 600 !important;
    }

    /* ---- Selectbox/dropdown dark theme ---- */
    .stSelectbox > div > div {
        background-color: #163636 !important;
        color: #D9FAF4 !important;
        border-color: #00BFA6 !important;
    }

    /* ---- Warning banner ---- */
    .warning-banner {
        background: linear-gradient(135deg, #00BFA6, #00a08a);
        color: #0F2A2A;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 12px;
        text-align: center;
        margin-top: 20px;
        font-weight: 500;
    }

    /* ---- Plotly chart dark bg ---- */
    .stPlotlyChart {
        background: #163636;
        border-radius: 12px;
        padding: 8px;
    }

    /* ---- Divider ---- */
    hr {
        border-color: rgba(0,191,166,0.3) !important;
    }

    /* ---- Info box override ---- */
    .stAlert {
        background-color: #163636 !important;
        color: #D9FAF4 !important;
        border: 1px solid #00BFA6 !important;
    }

    /* ---- Expander text ---- */
    .streamlit-expanderHeader {
        color: #D9FAF4 !important;
    }
    .streamlit-expanderContent {
        color: #D9FAF4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- Navbar with Streamlit-compatible navigation ----
# We use query params to handle page state
if "page" not in st.session_state:
    st.session_state.page = "Current Risk"

PAGES = ["Current Risk", "Historical", "Model Comparison", "About"]

# Build navbar HTML
nav_links = ""
for p in PAGES:
    active = "active" if p == st.session_state.page else ""
    nav_links += f'<span class="nav-btn {active}" id="nav-{p}">{p}</span>'

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

# Streamlit buttons row for actual navigation (styled minimal)
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

# ---- Page routing ----
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

# Footer
st.markdown("""
<div class="warning-banner">
    ⚠️ Research prototype — Not for emergency decisions.
    Always follow FFWC and local authority instructions.
</div>
""", unsafe_allow_html=True)
