# src/floodcast/dashboard/views/historical.py

"""Page 2: Historical analysis — blue palette."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from floodcast.dashboard.api_client import get_history


def render() -> None:
    st.title("Historical Analysis")
    st.caption("How well did the model predict past floods?")

    # ---- Controls ----
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Year", list(range(2024, 1999, -1)))
    with col2:
        season = st.selectbox("Season", [
            "Monsoon (Jun-Oct)",
            "Full Year",
            "Pre-Monsoon (Mar-May)",
        ])

    if "Monsoon" in season:
        start, end = date(year, 6, 1), date(year, 10, 31)
    elif "Pre-Monsoon" in season:
        start, end = date(year, 3, 1), date(year, 5, 31)
    else:
        start, end = date(year, 1, 1), date(year, 12, 31)

    try:
        entries = get_history(start, end)["entries"]
    except Exception:
        entries = []

    if not entries:
        st.warning("No data for this period.")
        return

    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["date"])

    # ---- Chart ----
    st.subheader("Predictions vs Actual Floods")

    fig = go.Figure()

    # Risk zone backgrounds
    fig.add_hrect(y0=0, y1=0.25, fillcolor="#e8f5e9",
                  opacity=0.35, line_width=0)
    fig.add_hrect(y0=0.25, y1=0.50, fillcolor="#fffde7",
                  opacity=0.35, line_width=0)
    fig.add_hrect(y0=0.50, y1=0.75, fillcolor="#fff3e0",
                  opacity=0.3, line_width=0)
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor="#ffebee",
                  opacity=0.3, line_width=0)

    # Prediction line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["flood_probability"],
        mode="lines",
        name="What the model predicted",
        line={"color": "#3f6593", "width": 2},
    ))

    # Actual flood markers
    if "actual_flood" in df.columns:
        flood_df = df[df["actual_flood"].astype(bool)]
        if not flood_df.empty:
            fig.add_trace(go.Scatter(
                x=flood_df["date"],
                y=flood_df["flood_probability"],
                mode="markers",
                name="Actual flood happened",
                marker={
                    "color": "#c62828",
                    "size": 9,
                    "symbol": "diamond",
                    "line": {"width": 1, "color": "#ffffff"},
                },
            ))

    fig.update_layout(
        yaxis_title="Flood Chance",
        yaxis_range=[0, 1.02],
        yaxis_tickformat=".0%",
        xaxis_title="",
        height=420,
        margin={"t": 10, "b": 30, "l": 50, "r": 20},
        plot_bgcolor="#163636",
        paper_bgcolor="#163636",
        font={"family": "sans-serif", "color": "#D9FAF4", "size": 13},
        legend={"orientation": "h", "y": -0.12, "font": {"size": 12}},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Scorecard ----
    st.subheader("Model Scorecard")

    if "actual_flood" in df.columns:
        df["pred"] = df["flood_probability"] >= 0.50
        df["actual"] = df["actual_flood"].astype(bool)

        caught = (df["pred"] & df["actual"]).sum()
        missed = (~df["pred"] & df["actual"]).sum()
        false_alarm = (df["pred"] & ~df["actual"]).sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="card" style="text-align:center;
                border-top:3px solid #689f38;">
                <div class="label">Correctly warned</div>
                <div class="big-num" style="color:#689f38;">
                    {int(caught)}</div>
                <div style="font-size:12px; color:#80aad3;">
                    floods the model caught</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card" style="text-align:center;
                border-top:3px solid #c62828;">
                <div class="label">Missed</div>
                <div class="big-num" style="color:#c62828;">
                    {int(missed)}</div>
                <div style="font-size:12px; color:#80aad3;">
                    floods with no warning</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="card" style="text-align:center;
                border-top:3px solid #f9a825;">
                <div class="label">False alarms</div>
                <div class="big-num" style="color:#f9a825;">
                    {int(false_alarm)}</div>
                <div style="font-size:12px; color:#80aad3;">
                    warnings without a flood</div>
            </div>""", unsafe_allow_html=True)

        total = caught + missed
        if total > 0:
            rate = caught / total
            st.markdown(f"""
            <div class="card card-accent">
                The model caught <strong>{rate:.0%}</strong> of actual
                floods in this period.
            </div>""", unsafe_allow_html=True)
