# src/floodcast/dashboard/theme.py

"""Dark theme constants for FloodCast BD dashboard."""

from __future__ import annotations

# ---- Palette ----
BG = "#0F172A"
CARD_BG = "#1E293B"
ACCENT = "#F97316"
ACCENT_LIGHT = "#431407"
ACCENT_BORDER = "#9A3412"
DARK = "#CBD5E1"
TEXT = "#F1F5F9"
TEXT_MUTED = "#94A3B8"
BORDER = "#334155"
SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#EF4444"

# ---- Model palette (distinct on dark bg) ----
MODEL_COLORS: dict[str, str] = {
    "xgboost": "#F97316",
    "lstm": "#38BDF8",
    "lstm_physics": "#34D399",
}

MODEL_LABELS: dict[str, str] = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "lstm_physics": "LSTM + Physics",
}

SET_LABELS: dict[str, str] = {
    "set_a": "Basic",
    "set_b": "Extended",
    "set_c": "Physics",
}

MODEL_ORDER: list[str] = ["xgboost", "lstm", "lstm_physics"]


def chart_layout(**kw: object) -> dict:
    """Plotly layout for dark theme."""
    defaults: dict = {
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, system-ui, sans-serif", "color": TEXT_MUTED, "size": 12},
        "margin": {"t": 24, "b": 40, "l": 50, "r": 20},
        "xaxis": {"gridcolor": BORDER, "linecolor": BORDER, "zeroline": False},
        "yaxis": {"gridcolor": BORDER, "linecolor": BORDER, "zeroline": False},
        "legend": {
            "orientation": "h",
            "y": -0.18,
            "font": {"size": 11, "color": TEXT_MUTED},
        },
    }
    defaults.update(kw)
    return defaults