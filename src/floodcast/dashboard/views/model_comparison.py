# src/floodcast/dashboard/views/model_comparison.py

"""Model comparison — Slate + Blue theme."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from floodcast.dashboard.api_client import get_metrics

COLORS = {
    "XGBoost": "#3b82f6",
    "LSTM": "#8b5cf6",
    "LSTM+Physics": "#22c55e",
}


def _demo_data() -> pd.DataFrame:
    rows = [
        ("XGBoost", "Rain only", 82, 0.78, 0.84, 42),
        ("XGBoost", "Rain + River", 87, 0.85, 0.88, 42),
        ("XGBoost", "All data", 89, 0.87, 0.90, 42),
        ("LSTM", "Rain only", 80, 0.76, 0.82, 185),
        ("LSTM", "Rain + River", 86, 0.83, 0.87, 185),
        ("LSTM", "All data", 88, 0.86, 0.89, 185),
        ("LSTM+Physics", "Rain only", 83, 0.80, 0.85, 190),
        ("LSTM+Physics", "Rain + River", 89, 0.87, 0.90, 190),
        ("LSTM+Physics", "All data", 91, 0.89, 0.92, 190),
    ]
    return pd.DataFrame(
        rows,
        columns=["Model", "Data", "Accuracy", "F1", "AUC", "Time (s)"],
    )


def render() -> None:
    st.title("Model Comparison")
    st.caption("Which prediction method works best?")

    try:
        _ = get_metrics()
    except Exception:
        pass

    st.markdown("""
    <div class="card card-accent">
        <strong style="color:#f1f5f9;">
            3 prediction methods tested:</strong>
        <table style="margin-top:12px; width:100%;
            border-collapse:collapse;">
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; font-weight:600; color:#3b82f6;">
                XGBoost</td>
            <td style="padding:8px; color:#cbd5e1;">
                Finds patterns in data tables.
                Fast and reliable.</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; font-weight:600; color:#8b5cf6;">
                LSTM</td>
            <td style="padding:8px; color:#cbd5e1;">
                Reads data like a diary — remembers yesterday
                and last week.</td>
        </tr>
        <tr>
            <td style="padding:8px; font-weight:600; color:#22c55e;">
                LSTM + Physics</td>
            <td style="padding:8px; color:#cbd5e1;">
                Same as LSTM but knows how water behaves
                (more rain → more flooding).</td>
        </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "⚠️ Demo numbers. Real results after training.",
        icon="ℹ️",
    )

    df = _demo_data()

    st.subheader("Detailed Metrics")

    best = df[df["Data"] == "All data"]
    c1, c2, c3 = st.columns(3)
    for i, (_, row) in enumerate(best.iterrows()):
        col = [c1, c2, c3][i]
        clr = COLORS[row["Model"]]
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;
                border-top:2px solid {clr};">
                <div style="font-size:14px; font-weight:600;
                    color:{clr};">{row['Model']}</div>
                <div style="margin:12px 0;">
                    <div class="label">Accuracy</div>
                    <div style="font-size:22px; font-weight:700;
                        color:#f1f5f9;">{row['Accuracy']}%</div>
                </div>
                <div style="display:flex; justify-content:space-around;">
                    <div>
                        <div class="label">F1</div>
                        <div style="color:#f1f5f9; font-weight:600;">
                            {row['F1']:.2f}</div>
                    </div>
                    <div>
                        <div class="label">AUC</div>
                        <div style="color:#f1f5f9; font-weight:600;">
                            {row['AUC']:.2f}</div>
                    </div>
                    <div>
                        <div class="label">Time</div>
                        <div style="color:#f1f5f9; font-weight:600;">
                            {int(row['Time (s)'])}s</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.subheader("Best Method (all data)")

    best = df[df["Data"] == "All data"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=best["Model"],
        y=best["Accuracy"],
        marker_color=[COLORS[m] for m in best["Model"]],
        text=[f"{a}%" for a in best["Accuracy"]],
        textposition="outside",
        textfont={"size": 14, "color": "#e2e8f0"},
    ))

    fig.update_layout(
        yaxis_title="Accuracy (%)",
        yaxis_range=[70, 100],
        height=380,
        margin={"t": 10, "b": 30, "l": 50, "r": 20},
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font={"family": "sans-serif", "color": "#94a3b8", "size": 12},
        showlegend=False,
        xaxis={"gridcolor": "#1e293b", "linecolor": "#334155"},
        yaxis={"gridcolor": "#1e293b", "linecolor": "#334155"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="card" style="border-left:3px solid #22c55e;">
        <strong style="color:#22c55e;">
            Winner: LSTM + Physics (91%)</strong>
        <br><span style="color:#cbd5e1;">
        Physics rules prevent illogical predictions — the model
        knows more rain means higher flood risk.</span>
    </div>
    """, unsafe_allow_html=True)

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
            marker={"size": 8},
            text=[f"{a}%" for a in sub["Accuracy"]],
            textposition="top center",
            textfont={"size": 11, "color": "#cbd5e1"},
        ))

    fig2.update_layout(
        yaxis_title="Accuracy (%)",
        yaxis_range=[75, 95],
        xaxis_title="Data used",
        height=380,
        margin={"t": 10, "b": 30, "l": 50, "r": 20},
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font={"family": "sans-serif", "color": "#94a3b8", "size": 12},
        legend={"orientation": "h", "y": -0.15,
                "font": {"size": 11, "color": "#94a3b8"}},
        xaxis={"gridcolor": "#1e293b", "linecolor": "#334155"},
        yaxis={"gridcolor": "#1e293b", "linecolor": "#334155"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="card card-accent">
        <strong style="color:#f1f5f9;">Yes!</strong>
        <span style="color:#cbd5e1;">
        Every model improves with more data. Adding river levels
        to rainfall gives the biggest jump.</span>
    </div>
    """, unsafe_allow_html=True)
