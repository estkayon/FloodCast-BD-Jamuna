# src/floodcast/dashboard/views/current_risk.py

"""Page 1: Current flood risk — clean with blue palette."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from floodcast.dashboard.api_client import get_features, get_predictions, get_status

RISK_MAP = {
    "LOW": {
        "title": "Safe",
        "bangla": "নিরাপদ",
        "css": "risk-safe",
        "color": "#689f38",
        "what": "No flood risk expected. River levels are normal.",
        "do": "No action needed.",
    },
    "MODERATE": {
        "title": "Caution",
        "bangla": "সতর্কতা",
        "css": "risk-caution",
        "color": "#f9a825",
        "what": "Some flood risk. Water levels are rising.",
        "do": "Stay updated. Keep important items ready to move.",
    },
    "HIGH": {
        "title": "Warning",
        "bangla": "সতর্কবার্তা",
        "css": "risk-warning",
        "color": "#e65100",
        "what": "Flooding is likely in the coming days.",
        "do": "Move valuables to higher ground. Prepare to evacuate.",
    },
    "VERY_HIGH": {
        "title": "Danger",
        "bangla": "বিপদ",
        "css": "risk-danger",
        "color": "#c62828",
        "what": "Flooding is very likely or already happening.",
        "do": "Follow local authority instructions. Evacuate if told.",
    },
}


def render() -> None:
    st.title("Current Flood Risk")
    st.caption("Jamuna River at Sirajganj, Bangladesh")

    # ---- Get data ----
    try:
        status = get_status()
    except Exception:
        status = {
            "current_risk": "MODERATE",
            "flood_probability": 0.35,
            "last_updated": "Demo mode",
            "model_version": "1.0.0",
        }

    risk = status["current_risk"]
    prob = status["flood_probability"]
    info = RISK_MAP.get(risk, RISK_MAP["MODERATE"])

    # ---- Risk card ----
    st.markdown(f"""
    <div class="{info['css']}">
        <div style="font-size:14px; color:#5b86b6;">Current Status</div>
        <div style="font-size:36px; font-weight:700; color:#1b3554;
            margin:8px 0;">
            {info['title']} — {info['bangla']}
        </div>
        <div style="font-size:16px; color:#3f6593; margin:4px 0;">
            {info['what']}
        </div>
        <div style="font-size:15px; color:#3f6593; margin:8px 0 0 0;">
            <strong>What to do:</strong> {info['do']}
        </div>
        <div style="font-size:13px; color:#5b86b6; margin-top:12px;">
            Flood chance: {prob:.0%} · Updated: {str(status['last_updated'])[:19]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ---- 3-day forecast cards ----
    st.subheader("Next 3 Days")

    try:
        today = date.today()
        preds = get_predictions(today, today + timedelta(days=2))
        pred_list = preds["predictions"]
    except Exception:
        pred_list = [
            {
                "date": str(date.today() + timedelta(days=i)),
                "flood_probability": 0.25 + i * 0.12,
                "risk_level": ["LOW", "MODERATE", "MODERATE"][i],
                "confidence_interval": [0.15 + i * 0.1, 0.35 + i * 0.15],
            }
            for i in range(3)
        ]

    day_names = ["Today", "Tomorrow", "Day After"]
    cols = st.columns(3)
    for i, (col, pred) in enumerate(zip(cols, pred_list)):
        p = pred["flood_probability"]
        r = pred["risk_level"]
        cfg = RISK_MAP.get(r, RISK_MAP["MODERATE"])
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;
                border-top:3px solid {cfg['color']};">
                <div class="label">{day_names[i]}</div>
                <div style="font-size:13px; color:#80aad3;">
                    {pred['date']}</div>
                <div class="big-num" style="color:{cfg['color']};">
                    {p:.0%}</div>
                <div style="font-size:14px; color:#3f6593;">
                    {cfg['title']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Forecast chart ----
    st.subheader("72-Hour Forecast Chart")

    df = pd.DataFrame(pred_list)
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    # Risk zone backgrounds
    fig.add_hrect(y0=0, y1=0.25, fillcolor="#e8f5e9",
                  opacity=0.4, line_width=0,
                  annotation_text="Safe",
                  annotation_position="top left",
                  annotation_font_size=11,
                  annotation_font_color="#689f38")
    fig.add_hrect(y0=0.25, y1=0.50, fillcolor="#fffde7",
                  opacity=0.4, line_width=0,
                  annotation_text="Caution",
                  annotation_position="top left",
                  annotation_font_size=11,
                  annotation_font_color="#f9a825")
    fig.add_hrect(y0=0.50, y1=0.75, fillcolor="#fff3e0",
                  opacity=0.35, line_width=0,
                  annotation_text="Warning",
                  annotation_position="top left",
                  annotation_font_size=11,
                  annotation_font_color="#e65100")
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor="#ffebee",
                  opacity=0.35, line_width=0,
                  annotation_text="Danger",
                  annotation_position="top left",
                  annotation_font_size=11,
                  annotation_font_color="#c62828")

    # Confidence band
    if "confidence_interval" in df.columns:
        ci_lo = [ci[0] for ci in df["confidence_interval"]]
        ci_hi = [ci[1] for ci in df["confidence_interval"]]
        fig.add_trace(go.Scatter(
            x=list(df["date"]) + list(df["date"][::-1]),
            y=ci_hi + ci_lo[::-1],
            fill="toself",
            fillcolor="rgba(63,101,147,0.12)",
            line={"color": "rgba(0,0,0,0)"},
            name="Uncertainty range",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["flood_probability"],
        mode="lines+markers+text",
        name="Flood chance",
        line={"color": "#3f6593", "width": 3},
        marker={"size": 10, "color": "#3f6593"},
        text=[f"{p:.0%}" for p in df["flood_probability"]],
        textposition="top center",
        textfont={"size": 13, "color": "#1b3554"},
    ))

    fig.update_layout(
        yaxis_title="Flood Chance",
        yaxis_range=[0, 1],
        yaxis_tickformat=".0%",
        xaxis_title="",
        height=380,
        margin={"t": 10, "b": 30, "l": 50, "r": 20},
        plot_bgcolor="#163636",
        paper_bgcolor="#163636",
        font={"family": "sans-serif", "color": "#D9FAF4", "size": 13},
        legend={"orientation": "h", "y": -0.12,
                "font": {"size": 12}},
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Weather data ----
    st.subheader("Current Conditions")

    try:
        feat_data = get_features()
        feat_list = feat_data["features"]
    except Exception:
        feat_list = []

    if feat_list:
        latest = feat_list[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="label">Rainfall</div>
                <div class="big-num">
                    {latest.get('daily_rainfall', '—')}</div>
                <div style="font-size:13px; color:#80aad3;">
                    mm today</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="label">Temperature</div>
                <div class="big-num">
                    {latest.get('temperature_2m', '—')}°</div>
                <div style="font-size:13px; color:#80aad3;">
                    Celsius</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="label">River Flow</div>
                <div class="big-num">
                    {latest.get('discharge', '—')}</div>
                <div style="font-size:13px; color:#80aad3;">
                    m³/s</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="label">Soil Moisture</div>
                <div class="big-num">
                    {latest.get('soil_moisture_layer1', '—')}</div>
                <div style="font-size:13px; color:#80aad3;">
                    top layer</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Weather data unavailable (API offline). Showing demo.")

    # ---- Simple explainer ----
    with st.expander("How to read this page"):
        st.markdown("""
        **Flood Chance** is a percentage from 0% to 100%.
        Think of it like a weather forecast:

        - **0-25%** = Safe, normal conditions
        - **25-50%** = Be careful, water levels are higher than usual
        - **50-75%** = Prepare for flooding
        - **75-100%** = Flooding is very likely

        The **shaded area** around the line shows the range of
        uncertainty. A wider band means less certainty in the
        prediction.
        """)
