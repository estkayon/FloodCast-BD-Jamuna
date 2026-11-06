# src/floodcast/dashboard/views/historical.py

"""Historical Analysis + Actual vs Predicted on holdout data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler

from floodcast.config import ROOT_DIR
from floodcast.dashboard.theme import (
    ACCENT,
    BORDER,
    DANGER,
    SUCCESS,
    TEXT,
    TEXT_MUTED,
    WARNING,
    chart_layout,
)
from floodcast.features.build import select_features
from floodcast.features.engineer import engineer_features
from floodcast.features.physics import add_physics_features
from floodcast.models.xgboost_model import load_xgboost, predict_xgboost

ROOT = ROOT_DIR


@st.cache_data
def _load_parquet() -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / "features_raw.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _find_models() -> list[Path]:
    models_dir = ROOT / "models"
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("xgboost_*.json"))


def _parse_model_name(path: Path) -> dict:
    stem = path.stem  # e.g. xgboost_set_a_lead1d
    # Find lead time from the "leadNd" suffix
    import re
    m = re.search(r"lead(\d+)d$", stem)
    if not m:
        return None
    lead_time = int(m.group(1))
    # Everything between first underscore and _leadNd is feature_set
    prefix = stem.split("_", 1)[0]  # xgboost
    feature_set = stem[len(prefix) + 1 : m.start() - 1]  # set_a
    return {
        "model_type": prefix,
        "feature_set": feature_set,
        "lead_time": lead_time,
        "path": path,
    }


@st.cache_data
def _generate_predictions(
    df_hash: str,
    model_path_str: str,
    feature_set: str,
    lead_time: int,
) -> pd.DataFrame | None:
    """Replicate prepare_data() pipeline then predict."""
    df_raw = _load_parquet()
    if df_raw is None:
        return None

    df = df_raw.copy()
    holdout_years = [2023, 2024, 2025]

    df = engineer_features(df)
    df = add_physics_features(df)
    df_sel = select_features(df, feature_set)

    df_sel = df_sel.copy()
    df_sel["flood_label"] = df_sel["flood_label"].shift(-lead_time)
    df_sel = df_sel.dropna(subset=["flood_label"])
    df_sel["flood_label"] = df_sel["flood_label"].astype(int)

    df_sel["year"] = pd.to_datetime(df_sel["date"]).dt.year
    train_mask = ~df_sel["year"].isin(holdout_years)
    test_mask = df_sel["year"].isin(holdout_years)

    feature_cols = [
        c for c in df_sel.columns if c not in ("date", "flood_label", "year")
    ]

    X_train = df_sel.loc[train_mask, feature_cols].values.astype(np.float64)
    X_test = df_sel.loc[test_mask, feature_cols].values.astype(np.float64)
    y_test = df_sel.loc[test_mask, "flood_label"].values.astype(np.int64)
    dates_test = df_sel.loc[test_mask, "date"].values

    if len(X_test) == 0:
        return None

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = load_xgboost(Path(model_path_str))
    y_prob = predict_xgboost(model, X_test_scaled)
    y_pred = (y_prob >= 0.5).astype(int)

    return pd.DataFrame({
        "date": dates_test,
        "flood_label": y_test,
        "probability": y_prob,
        "predicted": y_pred,
    })


# ------------------------------------------------------------------ #
# Historical data sections
# ------------------------------------------------------------------ #


def _runoff_chart(df_year: pd.DataFrame, df_all: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-title">Runoff Time Series</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()

    if "runoff" in df_year.columns:
        fig.add_trace(
            go.Scatter(
                x=df_year["date"], y=df_year["runoff"],
                mode="lines", name="Daily Runoff",
                line={"color": "#FFFFFF", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(255,255,255,0.06)",
            )
        )
        if "flood_label" in df_year.columns:
            pct90 = df_all["runoff"].quantile(0.90)
            fig.add_hline(
                y=pct90, line_dash="dash", line_color=DANGER, opacity=0.5,
                annotation_text=f"90th pct = {pct90:.4f}",
                annotation_font_color=DANGER, annotation_font_size=10,
                annotation_position="top right",
            )
            flood_days = df_year[df_year["flood_label"] == 1]
            if not flood_days.empty:
                fig.add_trace(
                    go.Scatter(
                        x=flood_days["date"], y=flood_days["runoff"],
                        mode="markers", name="Flood Day",
                        marker={"color": DANGER, "size": 7, "symbol": "diamond"},
                    )
                )

    fig.update_layout(**chart_layout(height=380, yaxis_title="Runoff (m)"))
    st.plotly_chart(fig, use_container_width=True)


def _period_summary(df_year: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-title">Period Summary</div>',
        unsafe_allow_html=True,
    )
    total = len(df_year)
    flood_count = (
        int(df_year["flood_label"].sum()) if "flood_label" in df_year.columns else 0
    )
    flood_pct = flood_count / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="ocard" style="text-align:center;">
            <div class="sub-label">Total Days</div>
            <div class="big-num">{total}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        clr = DANGER if flood_pct > 15 else WARNING if flood_pct > 5 else SUCCESS
        st.markdown(
            f"""<div class="ocard" style="text-align:center;border-top:2px solid {clr};">
            <div class="sub-label">Flood Days</div>
            <div class="big-num" style="color:{clr};">{flood_count}</div>
            <div class="metric-sub">{flood_pct:.1f}% of period</div>
        </div>""",
            unsafe_allow_html=True,
        )

    if "runoff" in df_year.columns:
        with c3:
            st.markdown(
                f"""<div class="ocard" style="text-align:center;">
                <div class="sub-label">Mean Runoff</div>
                <div class="big-num">{df_year['runoff'].mean():.4f}</div>
                <div class="metric-sub">m/day</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""<div class="ocard" style="text-align:center;
                    border-top:2px solid {ACCENT};">
                <div class="sub-label">Peak Runoff</div>
                <div class="big-num" style="color:{ACCENT};">
                    {df_year['runoff'].max():.4f}</div>
                <div class="metric-sub">m/day</div>
            </div>""",
                unsafe_allow_html=True,
            )


def _yearly_flood_bars(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-title">Flood Days by Year</div>',
        unsafe_allow_html=True,
    )
    if "flood_label" not in df.columns:
        return

    yearly = (
        df.groupby(df["date"].dt.year)
        .agg(flood_days=("flood_label", "sum"))
        .reset_index()
    )
    yearly.columns = ["year", "flood_days"]
    holdout = [2023, 2024, 2025]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=yearly["year"].astype(str),
            y=yearly["flood_days"],
            marker_color=[
                DANGER if y in holdout else ACCENT for y in yearly["year"]
            ],
            text=[f"{int(v)}" for v in yearly["flood_days"]],
            textposition="outside",
            textfont={"size": 10, "color": TEXT_MUTED},
        )
    )
    fig.update_layout(**chart_layout(height=340, yaxis_title="Flood Days", showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""<div class="ocard ocard-accent" style="font-size:12px;color:{TEXT_MUTED};">
        <span style="color:{DANGER};">■</span> Holdout years (2023-2025) — test only.
        <span style="color:{ACCENT};">■</span> Training years.
    </div>""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ #
# Actual vs Predicted sections
# ------------------------------------------------------------------ #


def _pred_summary_cards(pred_df: pd.DataFrame) -> None:
    total = len(pred_df)
    actual_floods = int(pred_df["flood_label"].sum())
    correct = int((pred_df["predicted"] == pred_df["flood_label"]).sum())
    accuracy = correct / total if total else 0

    tp = int(((pred_df["predicted"] == 1) & (pred_df["flood_label"] == 1)).sum())
    fp = int(((pred_df["predicted"] == 1) & (pred_df["flood_label"] == 0)).sum())
    fn = int(((pred_df["predicted"] == 0) & (pred_df["flood_label"] == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="ocard" style="text-align:center;">
            <div class="sub-label">Accuracy</div>
            <div class="big-num">{accuracy:.1%}</div>
            <div class="metric-sub">{correct}/{total} correct</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="ocard" style="text-align:center;">
            <div class="sub-label">Actual Floods</div>
            <div class="big-num" style="color:{DANGER};">{actual_floods}</div>
            <div class="metric-sub">{actual_floods / total:.1%} of days</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="ocard" style="text-align:center;">
            <div class="sub-label">Precision</div>
            <div class="big-num">{precision:.3f}</div>
            <div class="metric-sub">{tp} TP / {fp} FP</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="ocard" style="text-align:center;">
            <div class="sub-label">Recall</div>
            <div class="big-num">{recall:.3f}</div>
            <div class="metric-sub">{tp} caught / {fn} missed</div>
        </div>""",
            unsafe_allow_html=True,
        )


def _probability_timeline(pred_df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-title">Flood Probability Timeline</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()

    flood_dates = pred_df.loc[pred_df["flood_label"] == 1, "date"]
    for d in flood_dates:
        fig.add_vrect(
            x0=d - pd.Timedelta(hours=12),
            x1=d + pd.Timedelta(hours=12),
            fillcolor=DANGER, opacity=0.1, line_width=0,
        )

    fig.add_trace(
        go.Scatter(
            x=pred_df["date"], y=pred_df["probability"],
            mode="lines", name="Flood Probability",
            line={"color": ACCENT, "width": 2},
            fill="tozeroy", fillcolor="rgba(249,115,22,0.1)",
        )
    )

    fig.add_hline(
        y=0.5, line_dash="dash", line_color=TEXT_MUTED, opacity=0.4,
        annotation_text="Threshold 0.5",
        annotation_font_color=TEXT_MUTED, annotation_font_size=10,
    )

    fig.update_layout(**chart_layout(height=380, yaxis_title="Probability", yaxis_range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""<div class="finding" style="font-size:12px;">
        <span style="color:{DANGER};">▮</span> Shaded = actual flood days ·
        Orange line = model probability ·
        Above 0.5 = flood predicted
    </div>""",
        unsafe_allow_html=True,
    )


def _yearly_comparison_bars(pred_df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-title">Yearly: Actual vs Predicted</div>',
        unsafe_allow_html=True,
    )
    pred_df = pred_df.copy()
    pred_df["year"] = pred_df["date"].dt.year

    rows = []
    for yr, grp in pred_df.groupby("year"):
        rows.append({
            "year": yr,
            "actual": int(grp["flood_label"].sum()),
            "predicted": int(grp["predicted"].sum()),
        })
    yearly = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=yearly["year"].astype(str), y=yearly["actual"],
            name="Actual Floods", marker_color=DANGER,
            text=yearly["actual"].astype(str), textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=yearly["year"].astype(str), y=yearly["predicted"],
            name="Predicted Floods", marker_color=ACCENT,
            text=yearly["predicted"].astype(str), textposition="outside",
        )
    )
    fig.update_layout(**chart_layout(height=340, barmode="group", yaxis_title="Flood Days"))
    st.plotly_chart(fig, use_container_width=True)


def _monthly_pattern(pred_df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-title">Monthly Prediction Pattern</div>',
        unsafe_allow_html=True,
    )
    pred_df = pred_df.copy()
    pred_df["month"] = pred_df["date"].dt.month

    monthly = (
        pred_df.groupby("month")
        .agg(avg_prob=("probability", "mean"), actual_rate=("flood_label", "mean"))
        .reset_index()
    )
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[month_names[m - 1] for m in monthly["month"]],
            y=monthly["actual_rate"],
            mode="lines+markers", name="Actual Flood Rate",
            line={"color": DANGER, "width": 2.5},
            marker={"size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[month_names[m - 1] for m in monthly["month"]],
            y=monthly["avg_prob"],
            mode="lines+markers", name="Avg Predicted Prob.",
            line={"color": ACCENT, "width": 2.5, "dash": "dash"},
            marker={"size": 8},
        )
    )
    fig.update_layout(**chart_layout(height=340, yaxis_title="Rate / Probability"))
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ #
# Main render
# ------------------------------------------------------------------ #


def render() -> None:
    st.markdown(
        '<h2 style="margin-bottom:2px;">Historical Analysis</h2>',
        unsafe_allow_html=True,
    )
    st.caption("25 years of ERA5-Land data + XGBoost predictions on holdout (2023-2025)")

    df = _load_parquet()
    if df is None:
        st.warning(
            "No data at `data/processed/features_raw.parquet`.\n\n"
            "Run the pipeline:\n```\npython -m floodcast.data.process\n"
            "python -m floodcast.features.build\n```"
        )
        return

    st.success(
        f"Loaded {len(df):,} days ({df['date'].min().year}–{df['date'].max().year})",
        icon="📊",
    )

    # ---- Selectors ----
    years = sorted(df["date"].dt.year.unique(), reverse=True)
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Year", years)
    with col2:
        season = st.selectbox(
            "Season",
            ["Monsoon (Jun-Oct)", "Full Year", "Pre-Monsoon (Mar-May)", "Post-Monsoon (Nov-Feb)"],
        )

    df_year = df[df["date"].dt.year == year].copy()
    if "Monsoon" in season:
        df_year = df_year[(df_year["date"].dt.month >= 6) & (df_year["date"].dt.month <= 10)]
    elif "Pre-Monsoon" in season:
        df_year = df_year[(df_year["date"].dt.month >= 3) & (df_year["date"].dt.month <= 5)]
    elif "Post-Monsoon" in season:
        df_year = df_year[(df_year["date"].dt.month >= 11) | (df_year["date"].dt.month <= 2)]

    if df_year.empty:
        st.warning("No data for this period.")
        return

    # ---- Historical sections ----
    _runoff_chart(df_year, df)
    _period_summary(df_year)
    _yearly_flood_bars(df)

       # ---- Actual vs Predicted ----
    st.divider()
    st.markdown(
        '<h3 style="margin-bottom:2px;">Actual vs Predicted</h3>',
        unsafe_allow_html=True,
    )
    st.caption("XGBoost best model on holdout years (2023-2025)")

    model_files = _find_models()
    if not model_files:
        st.info(
            "No XGBoost models found in `models/`. "
            "Run `python -m floodcast.models.train` to generate predictions."
        )
        return

    # Use first available model (best by default sort)
    import re as _re
    best_model = model_files[0]
    stem = best_model.stem
    m = _re.search(r"lead(\d+)d$", stem)
    lead_time = int(m.group(1)) if m else 1
    prefix = stem.split("_", 1)[0]
    feature_set = stem[len(prefix) + 1 : m.start() - 1] if m else "set_a"

    df_hash = f"{df.shape}_{df['date'].iloc[-1]}"

    with st.spinner("Generating predictions..."):
        pred_df = _generate_predictions(df_hash, str(best_model), feature_set, lead_time)

    if pred_df is None or pred_df.empty:
        st.info("Could not generate predictions for holdout years.")
        return

    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df["year"] = pred_df["date"].dt.year

    rows = []
    for yr, grp in pred_df.groupby("year"):
        actual = int(grp["flood_label"].sum())
        tp = int(((grp["predicted"] == 1) & (grp["flood_label"] == 1)).sum())
        missed = actual - tp
        rows.append({"year": str(yr), "correct": tp, "missed": missed})
    yearly = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=yearly["year"], y=yearly["correct"],
            name="Correctly Predicted",
            marker_color=SUCCESS,
            text=yearly["correct"].astype(str),
            textposition="inside",
            textfont={"size": 13, "color": "#0F172A"},
        )
    )
    fig.add_trace(
        go.Bar(
            x=yearly["year"], y=yearly["missed"],
            name="Missed",
            marker_color=DANGER,
            text=yearly["missed"].astype(str),
            textposition="inside",
            textfont={"size": 13, "color": "#FFFFFF"},
        )
    )
    fig.update_layout(
        **chart_layout(
            height=380,
            barmode="stack",
            yaxis_title="Flood Days",
            xaxis_title="Year",
        )
    )
    st.plotly_chart(fig, use_container_width=True)