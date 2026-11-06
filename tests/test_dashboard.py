# tests/test_dashboard.py

"""Tests for dashboard components."""

from __future__ import annotations


class TestSchemas:
    def test_risk_low(self):
        from floodcast.api.schemas import RiskLevel, risk_from_probability
        assert risk_from_probability(0.10) == RiskLevel.LOW

    def test_risk_moderate(self):
        from floodcast.api.schemas import RiskLevel, risk_from_probability
        assert risk_from_probability(0.30) == RiskLevel.MODERATE

    def test_risk_high(self):
        from floodcast.api.schemas import RiskLevel, risk_from_probability
        assert risk_from_probability(0.60) == RiskLevel.HIGH

    def test_risk_very_high(self):
        from floodcast.api.schemas import RiskLevel, risk_from_probability
        assert risk_from_probability(0.80) == RiskLevel.VERY_HIGH

    def test_risk_boundaries(self):
        from floodcast.api.schemas import RiskLevel, risk_from_probability
        assert risk_from_probability(0.0) == RiskLevel.LOW
        assert risk_from_probability(0.25) == RiskLevel.MODERATE
        assert risk_from_probability(0.50) == RiskLevel.HIGH
        assert risk_from_probability(0.75) == RiskLevel.VERY_HIGH


class TestViewImports:
    def test_import_historical(self):
        from floodcast.dashboard.views.historical import render
        assert callable(render)

    def test_import_model_performance(self):
        from floodcast.dashboard.views.model_performance import render
        assert callable(render)


class TestTheme:
    def test_import_theme(self):
        from floodcast.dashboard.theme import (
            ACCENT,
            BG,
            CARD_BG,
            MODEL_COLORS,
            TEXT,
            chart_layout,
        )
        assert BG == "#0F172A"
        assert CARD_BG == "#1E293B"
        assert ACCENT == "#F97316"
        assert TEXT == "#F1F5F9"
        assert "xgboost" in MODEL_COLORS
        assert "lstm" in MODEL_COLORS
        assert callable(chart_layout)

    def test_chart_layout_returns_dict(self):
        from floodcast.dashboard.theme import chart_layout
        layout = chart_layout()
        assert isinstance(layout, dict)
        assert "paper_bgcolor" in layout
        assert "plot_bgcolor" in layout


class TestAppImport:
    def test_app_module_exists(self):
        import floodcast.dashboard.app  # noqa: F401
