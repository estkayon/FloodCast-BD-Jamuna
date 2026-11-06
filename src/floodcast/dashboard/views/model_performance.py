# src/floodcast/dashboard/views/model_performance.py

"""Model Performance — experiment results analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from floodcast.dashboard.theme import (
    ACCENT,
    BORDER,
    DANGER,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    SET_LABELS,
    TEXT,
    TEXT_MUTED,
    chart_layout,
)

ROOT = Path(__file__).resolve().parents[4]


@st.cache_data
def _load() -> pd.DataFrame | None:
    path = ROOT / "results" / "experiment_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _key_finding(df: pd.DataFrame) -> None:
    best = df.loc[df["auc"].idxmax()]
    model_name = MODEL_LABELS.get(best["model"], best["model"])
    fs_name = SET_LABELS.get(best["feature_set"], best["feature_set"])
    lt = int(best["lead_time"])

    st.markdown(
        f"""
    <div class="finding">
        <strong>🏆 Best Result:</strong>
        {model_name} with {fs_name} features at {lt}-day lead
        achieves <strong>AUC {best['auc']:.3f}</strong>,
        F1 {best['f1']:.3f}, Recall {best.get('recall', 0):.3f}
    </div>""",
        unsafe_allow_html=True,
    )


def _model_cards(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
    cols = st.columns(len(MODEL_ORDER))
    for i, mt in enumerate(MODEL_ORDER):
        sub = df[df["model"] == mt]
        if sub.empty:
            continue
        best = sub.loc[sub["auc"].idxmax()]
        color = MODEL_COLORS.get(mt, ACCENT)
        pr_auc = best.get("auc_pr", 0)
        recall = best.get("recall", 0)
        precision = best.get("precision", 0)

        with cols[i]:
            st.markdown(
                f"""
            <div class="ocard ocard-glow" style="text-align:center;">
                <div class="grad-line"
                     style="background:linear-gradient(90deg,{color},{color}44);"></div>
                <div style="font-size:13px;font-weight:700;color:{color};
                    letter-spacing:0.5px;">
                    {MODEL_LABELS.get(mt, mt)}</div>
                <div class="big-num" style="margin:8px 0;">{best['auc']:.3f}</div>
                <div class="sub-label">Best AUC</div>
                <div class="metric-sub">
                    {SET_LABELS.get(best['feature_set'], best['feature_set'])}
                    · {int(best['lead_time'])}d lead
                </div>
                <div style="display:flex;justify-content:space-around;
                    margin-top:14px;padding-top:12px;
                    border-top:1px solid #334155;">
                    <div>
                        <div class="sub-label">F1</div>
                        <div style="color:{TEXT};font-weight:600;font-size:16px;">
                            {best['f1']:.3f}</div>
                    </div>
                    <div>
                        <div class="sub-label">Precision</div>
                        <div style="color:{TEXT};font-weight:600;font-size:16px;">
                            {precision:.3f}</div>
                    </div>
                    <div>
                        <div class="sub-label">Recall</div>
                        <div style="color:{TEXT};font-weight:600;font-size:16px;">
                            {recall:.3f}</div>
                    </div>
                </div>
            </div>""",
                unsafe_allow_html=True,
            )


def _lead_time_chart(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-title">Lead Time vs Accuracy</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_hline(
        y=0.85, line_dash="dot", line_color=DANGER, opacity=0.4,
        annotation_text="Target AUC = 0.85",
        annotation_font_color=DANGER, annotation_font_size=10,
        annotation_position="bottom right",
    )

    for mt in MODEL_ORDER:
        sub = df[df["model"] == mt]
        if sub.empty:
            continue
        avg = sub.groupby("lead_time")["auc"].mean().reset_index()
        color = MODEL_COLORS.get(mt, ACCENT)
        fig.add_trace(
            go.Scatter(
                x=avg["lead_time"], y=avg["auc"],
                mode="lines+markers+text",
                name=MODEL_LABELS.get(mt, mt),
                line={"color": color, "width": 3},
                marker={"size": 10, "line": {"width": 2, "color": "#0F172A"}},
                text=[f"{v:.3f}" for v in avg["auc"]],
                textposition="top center",
                textfont={"size": 11, "color": TEXT_MUTED},
            )
        )

    fig.update_layout(
        **chart_layout(
            height=400,
            xaxis_title="Lead Time (days)",
            yaxis_title="AUC-ROC",
            xaxis={"dtick": 1, "gridcolor": BORDER, "linecolor": BORDER},
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def _feature_ablation(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-title">Feature Set Ablation (1-Day Lead)</div>',
        unsafe_allow_html=True,
    )
    sub = df[df["lead_time"] == 1]
    if sub.empty:
        sub = df[df["lead_time"] == df["lead_time"].min()]

    fig = go.Figure()
    sets = sorted(sub["feature_set"].unique())

    for mt in MODEL_ORDER:
        mt_sub = sub[sub["model"] == mt]
        if mt_sub.empty:
            continue
        vals = []
        for fs in sets:
            fs_rows = mt_sub[mt_sub["feature_set"] == fs]
            vals.append(float(fs_rows["auc"].iloc[0]) if len(fs_rows) > 0 else 0)

        fig.add_trace(
            go.Bar(
                x=[SET_LABELS.get(fs, fs) for fs in sets],
                y=vals,
                name=MODEL_LABELS.get(mt, mt),
                marker_color=MODEL_COLORS.get(mt, ACCENT),
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                textfont={"size": 11, "color": TEXT_MUTED},
            )
        )

    fig.update_layout(**chart_layout(height=380, barmode="group", yaxis_title="AUC-ROC"))
    st.plotly_chart(fig, use_container_width=True)


def _heatmap(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-title">Performance Heatmap</div>',
        unsafe_allow_html=True,
    )
    metric_map = {
        "auc": "AUC-ROC", "f1": "F1 Score", "auc_pr": "PR-AUC",
        "precision": "Precision", "recall": "Recall",
    }
    available = [m for m in metric_map if m in df.columns]
    metric = st.selectbox("Metric", available, format_func=lambda m: metric_map.get(m, m))

    df2 = df.copy()
    df2["label"] = (
        df2["model"].map(MODEL_LABELS).fillna(df2["model"])
        + " · "
        + df2["feature_set"].map(SET_LABELS).fillna(df2["feature_set"])
    )
    pivot = df2.pivot_table(index="label", columns="lead_time", values=metric)
    pivot = pivot.sort_index()

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[f"{int(c)}d" for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale=[
                [0, "#0F172A"], [0.3, "#9A3412"],
                [0.6, "#F97316"], [1, "#FDE68A"],
            ],
            text=[[f"{v:.3f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 12, "color": "#F1F5F9"},
            hovertemplate="Model: %{y}<br>Lead: %{x}<br>Value: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **chart_layout(
            height=max(300, len(pivot) * 42 + 80),
            xaxis_title="Lead Time",
            yaxis={"autorange": "reversed", "gridcolor": BORDER, "linecolor": BORDER},
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def _full_table(df: pd.DataFrame) -> None:
    with st.expander("📋 Full Experiment Results"):
        display = df.copy()
        display["model"] = display["model"].map(MODEL_LABELS).fillna(display["model"])
        display["feature_set"] = (
            display["feature_set"].map(SET_LABELS).fillna(display["feature_set"])
        )
        rename = {
            "model": "Model", "feature_set": "Features", "lead_time": "Lead (d)",
            "auc": "AUC", "f1": "F1",
        }
        for col in ("auc_pr", "precision", "recall"):
            if col in display.columns:
                rename[col] = col.replace("_", " ").title()
        display = display.rename(columns=rename)
        st.dataframe(display, use_container_width=True, hide_index=True)


def render() -> None:
    st.markdown(
        '<h2 style="margin-bottom:2px;">Model Performance</h2>',
        unsafe_allow_html=True,
    )
    st.caption("27 experiments — 3 models × 3 feature sets × 3 lead times")

    df = _load()
    if df is None:
        st.warning(
            "No results at `results/experiment_results.csv`.\n\n"
            "Run training first:\n```\npython -m floodcast.models.train\n```"
        )
        return

    _key_finding(df)
    _model_cards(df)
    col1, col2 = st.columns(2)
    with col1:
        _lead_time_chart(df)
    with col2:
        _feature_ablation(df)
    _heatmap(df)
    _full_table(df)