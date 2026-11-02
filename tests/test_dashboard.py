# tests/test_dashboard.py

"""Tests for dashboard components."""

from __future__ import annotations


class TestApiClient:
    def test_imports(self):
        from floodcast.dashboard.api_client import (
            get_features,
            get_health,
            get_history,
            get_metrics,
            get_model_info,
            get_predictions,
            get_status,
        )
        assert callable(get_health)
        assert callable(get_status)
        assert callable(get_predictions)
        assert callable(get_history)
        assert callable(get_features)
        assert callable(get_model_info)
        assert callable(get_metrics)


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
    def test_import_current_risk(self):
        from floodcast.dashboard.views.current_risk import render
        assert callable(render)

    def test_import_historical(self):
        from floodcast.dashboard.views.historical import render
        assert callable(render)

    def test_import_model_comparison(self):
        from floodcast.dashboard.views.model_comparison import render
        assert callable(render)

    def test_import_about(self):
        from floodcast.dashboard.views.about import render
        assert callable(render)
