# src/floodcast/dashboard/views/current_risk.py

"""Current flood risk — Slate + Blue theme."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from floodcast.dashboard.api_client import get_features, get_predictions, get_status

def _format_time(raw: object) -> str:
    """Turn ISO timestamp into readable format like '03 Nov 2026, 09:49 AM'."""
    from datetime import datetime

    text = str(raw)[:19]
    try:
        dt = datetime.fromisoformat(text)
        return dt.strftime("%d %b %Y, %I:%M %p")
    except (ValueError, TypeError):
        return text

RISK_MAP = {
    "LOW": {
        "title": "Safe",
        "bangla": "নিরাপদ",
        "css": "risk-safe",
        "color": "#22c55e",
        "what": "No flood risk expected. River levels are normal.",
        "do": "No action needed.",
    },
    "MODERATE": {
        "title": "Caution",
        "bangla": "সতর্কতা",
        "css": "risk-caution",
        "color": "#eab308",
        "what": "Some flood risk. Water levels are rising.",
        "do": "Stay updated. Keep important items ready to move.",
    },
    "HIGH": {
        "title": "Warning",
        "bangla": "সতর্কবার্তা",
        "css": "risk-warning",
        "color": "#f97316",
        "what": "Flooding is likely in the coming days.",
        "do": "Move valuables to higher ground. Prepare to evacuate.",
    },
    "VERY_HIGH": {
        "title": "Danger",
        "bangla": "বিপদ",
        "css": "risk-danger",
        "color": "#ef4444",
        "what": "Flooding is very likely or already happening.",
        "do": "Follow local authority instructions. Evacuate if told.",
    },
}


def render() -> None:
    st.title("Current Flood Risk")
    st.caption("Jamuna River at Sirajganj, Bangladesh")

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

    st.markdown(f"""
    <div class="{info['css']}">
        <div style="font-size:12px; color:#94a3b8;
            text-transform:uppercase; letter-spacing:0.5px;">
            Current Status</div>
        <div style="font-size:34px; font-weight:700; color:#f1f5f9;
            margin:8px 0;">
            {info['title']} — {info['bangla']}
        </div>
        <div style="font-size:15px; color:#cbd5e1; margin:4px 0;">
            {info['what']}
        </div>
        <div style="font-size:14px; color:#cbd5e1; margin:8px 0 0 0;">
            <strong style="color:#f1f5f9;">What to do:</strong>
            {info['do']}
        </div>
        <div style="font-size:12px; color:#94a3b8; margin-top:12px;">
            Flood chance: {prob:.0%} ·
            Updated: {_format_time(status['last_updated'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

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
                "confidence_interval": [
                    0.15 + i * 0.1, 0.35 + i * 0.15,
                ],
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
                border-top:2px solid {cfg['color']};">
                <div class="label">{day_names[i]}</div>
                <div style="font-size:12px; color:#64748b;">
                    {pred['date']}</div>
                <div class="big-num" style="color:{cfg['color']};">
                    {p:.0%}</div>
                <div style="font-size:13px; color:#94a3b8;">
                    {cfg['title']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("72-Hour Forecast")

    df = pd.DataFrame(pred_list)
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    fig.add_hrect(y0=0, y1=0.25, fillcolor="#22c55e",
                  opacity=0.06, line_width=0,
                  annotation_text="Safe",
                  annotation_position="top left",
                  annotation_font_size=10,
                  annotation_font_color="#22c55e")
    fig.add_hrect(y0=0.25, y1=0.50, fillcolor="#eab308",
                  opacity=0.05, line_width=0,
                  annotation_text="Caution",
                  annotation_position="top left",
                  annotation_font_size=10,
                  annotation_font_color="#eab308")
    fig.add_hrect(y0=0.50, y1=0.75, fillcolor="#f97316",
                  opacity=0.05, line_width=0,
                  annotation_text="Warning",
                  annotation_position="top left",
                  annotation_font_size=10,
                  annotation_font_color="#f97316")
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor="#ef4444",
                  opacity=0.05, line_width=0,
                  annotation_text="Danger",
                  annotation_position="top left",
                  annotation_font_size=10,
                  annotation_font_color="#ef4444")

    if "confidence_interval" in df.columns:
        ci_lo = [ci[0] for ci in df["confidence_interval"]]
        ci_hi = [ci[1] for ci in df["confidence_interval"]]
        fig.add_trace(go.Scatter(
            x=list(df["date"]) + list(df["date"][::-1]),
            y=ci_hi + ci_lo[::-1],
            fill="toself",
            fillcolor="rgba(59,130,246,0.12)",
            line={"color": "rgba(0,0,0,0)"},
            name="Uncertainty",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["flood_probability"],
        mode="lines+markers+text",
        name="Flood chance",
        line={"color": "#3b82f6", "width": 3},
        marker={"size": 10, "color": "#3b82f6"},
        text=[f"{p:.0%}" for p in df["flood_probability"]],
        textposition="top center",
        textfont={"size": 13, "color": "#e2e8f0"},
    ))

    fig.update_layout(
        yaxis_title="Flood Chance",
        yaxis_range=[0, 1],
        yaxis_tickformat=".0%",
        xaxis_title="",
        height=380,
        margin={"t": 10, "b": 30, "l": 50, "r": 20},
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font={"family": "sans-serif", "color": "#94a3b8", "size": 12},
        legend={"orientation": "h", "y": -0.12,
                "font": {"size": 11, "color": "#94a3b8"}},
        showlegend=True,
        xaxis={"gridcolor": "#1e293b", "linecolor": "#334155"},
        yaxis={"gridcolor": "#1e293b", "linecolor": "#334155"},
    )
    st.plotly_chart(fig, use_container_width=True)

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
                <div style="font-size:12px; color:#64748b;">
                    mm today</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="label">Temperature</div>
                <div class="big-num">
                    {latest.get('temperature_2m', '—')}°</div>
                <div style="font-size:12px; color:#64748b;">
                    Celsius</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="label">River Flow</div>
                <div class="big-num">
                    {latest.get('discharge', '—')}</div>
                <div style="font-size:12px; color:#f2f2f2;">
                    m³/s</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="label">Soil Moisture</div>
                <div class="big-num">
                    {latest.get('soil_moisture_layer1', '—')}</div>
                <div style="font-size:12px; color:#64748b;">
                    top layer</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Weather data unavailable (API offline). Showing demo.")

    with st.expander("How to read this page"):
        st.markdown("""
        <div style="color:#f1f5f9;">
        <strong>Flood Chance</strong> is a percentage from 0% to 100%:
        <br><br>
        • <strong>0-25%</strong> = Safe, normal conditions<br>
        • <strong>25-50%</strong> = Be careful, water levels rising<br>
        • <strong>50-75%</strong> = Prepare for flooding<br>
        • <strong>75-100%</strong> = Flooding very likely
        <br><br>
        The <strong>shaded area</strong> shows uncertainty. Wider = less certain.
        </div>
        """, unsafe_allow_html=True)
