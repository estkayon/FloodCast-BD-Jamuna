# tests/test_api.py

"""API endpoint tests using httpx TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from floodcast.api.app import app


@pytest.fixture()
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "no_model")
        assert "model_version" in data

    def test_health_has_api_version(self, client):
        resp = client.get("/health")
        assert resp.json()["api_version"] == "1.0.0"


class TestPredictEndpoint:
    def test_predict_valid_request(self, client):
        resp = client.post("/api/v1/predict", json={
            "start_date": "2025-07-15",
            "end_date": "2025-07-17",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["basin"] == "jamuna_sirajganj"
        assert len(data["predictions"]) == 3
        assert "disclaimer" in data

    def test_predict_single_day(self, client):
        resp = client.post("/api/v1/predict", json={
            "start_date": "2025-07-15",
            "end_date": "2025-07-15",
        })
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert len(preds) == 1
        assert 0 <= preds[0]["flood_probability"] <= 1
        assert preds[0]["risk_level"] in ("LOW", "MODERATE", "HIGH", "VERY_HIGH")

    def test_predict_invalid_date_range(self, client):
        resp = client.post("/api/v1/predict", json={
            "start_date": "2025-07-20",
            "end_date": "2025-07-15",
        })
        assert resp.status_code == 400

    def test_predict_confidence_interval(self, client):
        resp = client.post("/api/v1/predict", json={
            "start_date": "2025-07-15",
            "end_date": "2025-07-15",
        })
        ci = resp.json()["predictions"][0]["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] <= ci[1]


class TestStatusEndpoint:
    def test_status_returns_200(self, client):
        resp = client.get("/api/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_risk" in data
        assert "flood_probability" in data


class TestHistoryEndpoint:
    def test_history_returns_entries(self, client):
        resp = client.get("/api/v1/history", params={
            "start_date": "2024-07-01",
            "end_date": "2024-07-10",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entries"]) == 10

    def test_history_default_dates(self, client):
        resp = client.get("/api/v1/history")
        assert resp.status_code == 200
        assert len(resp.json()["entries"]) > 0


class TestFeaturesEndpoint:
    def test_features_returns_7_days(self, client):
        resp = client.get("/api/v1/features")
        assert resp.status_code == 200
        assert len(resp.json()["features"]) == 7


class TestModelInfoEndpoint:
    def test_model_info(self, client):
        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_type" in data
        assert "model_version" in data


class TestMetricsEndpoint:
    def test_metrics(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "auc_roc" in data
        assert "f1" in data
