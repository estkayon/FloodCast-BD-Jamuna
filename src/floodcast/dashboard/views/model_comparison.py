# src/floodcast/dashboard/views/model_comparison.py

"""Page 3: Model comparison — blue palette."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from floodcast.dashboard.api_client import get_metrics

COLORS = {
    "XGBoost": "#80aad3",
    "LSTM": "#3f6593",
    "LSTM+Physics": "#1b3554",
}


def _demo_data() -> pd.DataFrame:
    rows = [
        ("XGBoost", "Rain only", 82),
        ("XGBoost", "Rain + River", 87),
        ("XGBoost", "All data", 89),
        ("LSTM", "Rain only", 80),
        ("LSTM", "Rain + River", 86),
        ("LSTM", "All data", 88),
        ("LSTM+Physics", "Rain only", 83),
        ("LSTM+Physics", "Rain + River", 89),
        ("LSTM+Physics", "All data", 91),
    ]
    return pd.DataFrame(rows, columns=["Model", "Data", "Accuracy"])


def render() -> None:
    st.title("Model Comparison")
    st.caption("Which prediction method works best?")

    try:
        _ = get_metrics()
    except Exception:
        pass

    # ---- Plain language explanation ----
    st.markdown("""
    <div class="card card-accent">
        <strong style="color:#1b3554;">
            We tested 3 prediction methods:</strong>
        <table style="margin-top:12px; width:100%;
            border-collapse:collapse;">
        <tr style="border-bottom:1px solid #c0e6fd;">
            <td style="padding:8px; font-weight:600; color:#80aad3;">
                XGBoost</td>
            <td style="padding:8px; color:#3f6593;">
                Finds patterns in data tables. Fast and reliable.</td>
        </tr>
        <tr style="border-bottom:1px solid #c0e6fd;">
            <td style="padding:8px; font-weight:600; color:#3f6593;">
                LSTM</td>
            <td style="padding:8px; color:#3f6593;">
                Reads data like a diary — remembers what happened
                yesterday and last week.</td>
        </tr>
        <tr>
            <td style="padding:8px; font-weight:600; color:#1b3554;">
                LSTM + Physics</td>
            <td style="padding:8px; color:#3f6593;">
                Same as LSTM but also knows how water actually behaves
                (more rain → higher river → more flooding).</td>
        </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "⚠️ Demo numbers shown. Real results appear after training.",
        icon="ℹ️",
    )

    df = _demo_data()

    # ---- Best model chart ----
    st.subheader("Best Method (using all data)")

    best = df[df["Data"] == "All data"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=best["Model"],
        y=best["Accuracy"],
        marker_color=[COLORS[m] for m in best["Model"]],
        text=[f"{a}%" for a in best["Accuracy"]],
        textposition="outside",
        textfont={"size": 14, "color": "#1b3554"},
    ))

    fig.update_layout(
        yaxis_title="Accuracy (%)",
        yaxis_range=[70, 100],
        height=380,
        margin={"t": 10, "b": 30, "l": 50, "r": 20},
        plot_bgcolor="#163636",
        paper_bgcolor="#163636",
        font={"family": "sans-serif", "color": "#D9FAF4", "size": 13},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="card" style="border-left:4px solid #1b3554;">
        <strong style="color:#1b3554;">
            Winner: LSTM + Physics (91%)</strong>
        <br><span style="color:#3f6593;">
        Adding physics rules makes the model smarter. It knows that
        more rain should mean higher flood risk — so it won't make
        illogical predictions.</span>
    </div>
    """, unsafe_allow_html=True)

    # ---- Does more data help? ----
    st.subheader("Does more data help?")

    fig2 = go.Figure()
    for model in ["XGBoost", "LSTM", "LSTM+Physics"]:
        sub = df[df["Model"] == model]
        fig2.add_trace(go.Scatter(
            x=sub["Data"],
            y=sub["Accuracy"],
            mode="lines+markers+text",
            name=model,
            line={"color": COLORS[model], "width": 3},
            marker={"size": 9},
            text=[f"{a}%" for a in sub["Accuracy"]],
            textposition="top center",
            textfont={"size": 11},
        ))

    fig2.update_layout(
        yaxis_title="Accuracy (%)",
        yaxis_range=[75, 95],
        xaxis_title="Data used",
        height=380,
        margin={"t": 10, "b": 30, "l": 50, "r": 20},
        plot_bgcolor="#163636",
        paper_bgcolor="#163636",
        font={"family": "sans-serif", "color": "#D9FAF4", "size": 13},
        legend={"orientation": "h", "y": -0.15, "font": {"size": 12}},
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="card card-accent">
        <strong>Yes!</strong> Every model gets better with more data.
        The biggest improvement comes from adding river water levels
        to rainfall data.
    </div>
    """, unsafe_allow_html=True)
