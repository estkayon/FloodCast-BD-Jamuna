# src/floodcast/dashboard/views/model_comparison.py

"""Model comparison — shows real training results from MLflow or fallback CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[4]

COLORS = {
    "xgboost": "#3b82f6",
    "lstm": "#8b5cf6",
    "lstm_physics": "#22c55e",
}

LABELS = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "lstm_physics": "LSTM + Physics",
}

SET_LABELS = {
    "set_a": "Runoff Only",
    "set_b": "Runoff + Weather",
    "set_c": "Full Features",
}


def _load_mlflow_results() -> pd.DataFrame | None:
    """Try loading results from local MLflow tracking directory."""
    mlruns = ROOT / "mlruns"
    if not mlruns.exists():
        return None

    try:
        import mlflow

        uri = mlruns.resolve().as_posix()
        if not uri.startswith("/"):
            uri = "/" + uri
        mlflow.set_tracking_uri("file://" + uri)
        experiment = mlflow.get_experiment_by_name("floodcast_bd")
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["params.lead_time_days ASC"],
        )
        if runs.empty:
            return None

        rows = []
        for _, run in runs.iterrows():
            # Skip old runs that had no lead_time_days (data leakage)
            if pd.isna(run.get("params.lead_time_days")):
                continue

            model_type = run.get("params.model_type", "")
            feature_set = run.get("params.feature_set", "")
            lead_time = int(run.get("params.lead_time_days") or 1)
            auc = run.get("metrics.auc_roc", 0)
            f1 = run.get("metrics.f1", 0)
            auc_pr = run.get("metrics.auc_pr", 0)
            precision = run.get("metrics.precision", 0)
            recall = run.get("metrics.recall", 0)

            rows.append({
                "model": model_type,
                "feature_set": feature_set,
                "lead_time": lead_time,
                "auc": auc,
                "f1": f1,
                "auc_pr": auc_pr,
                "precision": precision,
                "recall": recall,
            })

        return pd.DataFrame(rows)

    except Exception:
        return None


def _load_results_csv() -> pd.DataFrame | None:
    """Fallback: load results from saved CSV."""
    csv_path = ROOT / "results" / "experiment_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def render() -> None:
    st.title("Model Comparison")
    st.caption("Real training results — 27 experiments (3 models x 3 feature sets x 3 lead times)")

    # Try MLflow first, then CSV fallback
    df = _load_mlflow_results()
    if df is None:
        df = _load_results_csv()

    if df is None or df.empty:
        st.error(
            "No training results found. Run training first:\n\n"
            "```\npython -m floodcast.models.train\n```\n\n"
            "Or place `results/experiment_results.csv` in the project root."
        )
        return

    st.success(f"Loaded {len(df)} experiment results", icon="✅")

    # --- Model descriptions ---
    st.markdown("""
    <div class="card card-accent">
        <strong style="color:#f1f5f9;">
            3 prediction methods tested across 3 lead times:</strong>
        <table style="margin-top:12px; width:100%;
            border-collapse:collapse;">
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; font-weight:600; color:#3b82f6;">
                XGBoost</td>
            <td style="padding:8px; color:#cbd5e1;">
                Gradient boosting on tabular features.
                Fast and reliable.</td>
        </tr>
        <tr style="border-bottom:1px solid #334155;">
            <td style="padding:8px; font-weight:600; color:#8b5cf6;">
                LSTM</td>
            <td style="padding:8px; color:#cbd5e1;">
                Recurrent neural network — learns temporal
                patterns in sequences.</td>
        </tr>
        <tr>
            <td style="padding:8px; font-weight:600; color:#22c55e;">
                LSTM + Physics</td>
            <td style="padding:8px; color:#cbd5e1;">
                LSTM with monotonicity and correlation
                constraints from hydrology.</td>
        </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # --- Lead time selector ---
    lead_times = sorted(df["lead_time"].unique())
    lead_labels = {lt: f"{lt * 24}h ({lt}d)" for lt in lead_times}

    selected_lead = st.selectbox(
        "Lead Time",
        lead_times,
        format_func=lambda x: lead_labels[x],
        index=0,
    )

    df_lead = df[df["lead_time"] == selected_lead].copy()

    # --- Best model cards for selected lead time ---
    st.subheader(f"Performance at {selected_lead * 24}h Lead Time")

    models = ["xgboost", "lstm", "lstm_physics"]
    cols = st.columns(3)

    for i, model in enumerate(models):
        df_model = df_lead[df_lead["model"] == model]
        if df_model.empty:
            continue
        best = df_model.loc[df_model["auc"].idxmax()]
        clr = COLORS.get(model, "#3b82f6")

        with cols[i]:
            st.markdown(f"""
            <div class="card" style="text-align:center;
                border-top:2px solid {clr};">
                <div style="font-size:14px; font-weight:600;
                    color:{clr};">{LABELS.get(model, model)}</div>
                <div style="font-size:11px; color:#64748b;
                    margin:4px 0;">
                    Best: {SET_LABELS.get(best['feature_set'], best['feature_set'])}</div>
                <div style="margin:12px 0;">
                    <div class="label">AUC-ROC</div>
                    <div style="font-size:28px; font-weight:700;
                        color:#f1f5f9;">{best['auc']:.4f}</div>
                </div>
                <div style="display:flex; justify-content:space-around;">
                    <div>
                        <div class="label">F1</div>
                        <div style="color:#f1f5f9; font-weight:600;">
                            {best['f1']:.3f}</div>
                    </div>
                    <div>
                        <div class="label">AUC-PR</div>
                        <div style="color:#f1f5f9; font-weight:600;">
                            {best['auc_pr']:.3f}</div>
                    </div>
                    <div>
                        <div class="label">Recall</div>
                        <div style="color:#f1f5f9; font-weight:600;">
                            {best['recall']:.3f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- AUC by Feature Set (grouped bar) ---
    st.subheader("AUC-ROC by Feature Set")

    fig = go.Figure()
    for model in models:
        df_m = df_lead[df_lead["model"] == model].sort_values("feature_set")
        if df_m.empty:
            continue
        fig.add_trace(go.Bar(
            x=[SET_LABELS.get(fs, fs) for fs in df_m["feature_set"]],
            y=df_m["auc"],
            name=LABELS.get(model, model),
            marker_color=COLORS.get(model, "#3b82f6"),
            text=[f"{v:.3f}" for v in df_m["auc"]],
            textposition="outside",
            textfont={"size": 11, "color": "#e2e8f0"},
        ))

    fig.update_layout(
        barmode="group",
        yaxis_title="AUC-ROC",
        yaxis_range=[
            max(0.7, df_lead["auc"].min() - 0.05),
            min(1.0, df_lead["auc"].max() + 0.03),
        ],
        height=400,
        margin={"t": 10, "b": 30, "l": 50, "r": 20},
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font={"family": "sans-serif", "color": "#94a3b8", "size": 12},
        legend={"orientation": "h", "y": -0.15,
                "font": {"size": 11, "color": "#94a3b8"}},
        xaxis={"gridcolor": "#1e293b", "linecolor": "#334155"},
        yaxis={"gridcolor": "#1e293b", "linecolor": "#334155"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Lead Time Degradation ---
    st.subheader("How Lead Time Affects Accuracy")

    # Best AUC per model per lead time (any feature set)
    fig2 = go.Figure()

    # AUC threshold line
    fig2.add_hline(
        y=0.85,
        line_dash="dash",
        line_color="#ef4444",
        opacity=0.5,
        annotation_text="Reliable threshold (AUC=0.85)",
        annotation_font_color="#ef4444",
        annotation_font_size=10,
        annotation_position="bottom right",
    )

    for model in models:
        df_m = df[df["model"] == model]
        if df_m.empty:
            continue
        # Best AUC per lead time
        best_per_lead = df_m.groupby("lead_time")["auc"].max().reset_index()
        best_per_lead = best_per_lead.sort_values("lead_time")

        fig2.add_trace(go.Scatter(
            x=[f"{lt * 24}h" for lt in best_per_lead["lead_time"]],
            y=best_per_lead["auc"],
            mode="lines+markers+text",
            name=LABELS.get(model, model),
            line={"color": COLORS.get(model, "#3b82f6"), "width": 3},
            marker={"size": 10},
            text=[f"{v:.3f}" for v in best_per_lead["auc"]],
            textposition="top center",
            textfont={"size": 11, "color": "#cbd5e1"},
        ))

    fig2.update_layout(
        yaxis_title="Best AUC-ROC",
        yaxis_range=[
            max(0.7, df["auc"].min() - 0.05),
            min(1.02, df["auc"].max() + 0.03),
        ],
        xaxis_title="Lead Time",
        height=400,
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

    # --- RQ3 Answer ---
    best_overall = df.loc[df["auc"].idxmax()]
    winner_name = LABELS.get(best_overall["model"], best_overall["model"])
    winner_set = SET_LABELS.get(best_overall["feature_set"], best_overall["feature_set"])

    st.markdown(f"""
    <div class="card" style="border-left:3px solid #22c55e;">
        <strong style="color:#22c55e;">
            Best Model: {winner_name} — {winner_set}
            (Lead {int(best_overall['lead_time'])}d, AUC-ROC {best_overall['auc']:.4f})</strong>
        <br><span style="color:#cbd5e1;">
        24h and 48h forecasts are reliable (AUC &gt; 0.85).
        At 72h, XGBoost stays above the threshold but LSTM degrades.
        Physics constraints help LSTM at longer lead times.</span>
    </div>
    """, unsafe_allow_html=True)

    # --- Full Results Table ---
    with st.expander("Full Results Table (all 27 experiments)"):
        display_df = df.copy()
        display_df["model"] = display_df["model"].map(
            lambda x: LABELS.get(x, x)
        )
        display_df["feature_set"] = display_df["feature_set"].map(
            lambda x: SET_LABELS.get(x, x)
        )
        display_df["lead_time"] = display_df["lead_time"].map(
            lambda x: f"{x * 24}h"
        )
        display_df = display_df.rename(columns={
            "model": "Model",
            "feature_set": "Features",
            "lead_time": "Lead Time",
            "auc": "AUC-ROC",
            "f1": "F1",
            "auc_pr": "AUC-PR",
            "precision": "Precision",
            "recall": "Recall",
        })
        display_df = display_df.sort_values(
            ["Lead Time", "AUC-ROC"], ascending=[True, False]
        )

        st.dataframe(
            display_df.style.format({
                "AUC-ROC": "{:.4f}",
                "F1": "{:.4f}",
                "AUC-PR": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )