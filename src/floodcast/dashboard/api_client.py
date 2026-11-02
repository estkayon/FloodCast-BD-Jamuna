# src/floodcast/dashboard/api_client.py

"""HTTP client wrapper to communicate with the FloodCast FastAPI backend."""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import requests

API_URL = os.getenv("FLOODCAST_API_URL", "http://localhost:8000")
TIMEOUT = 10


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET request to the API."""
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST request to the API."""
    resp = requests.post(f"{API_URL}{path}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_health() -> dict[str, Any]:
    return _get("/health")


def get_status() -> dict[str, Any]:
    return _get("/api/v1/status")


def get_predictions(
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    start = start_date or date.today()
    end = end_date or (start + timedelta(days=2))
    return _post("/api/v1/predict", {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    })


def get_history(
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    params: dict[str, str] = {}
    if start_date:
        params["start_date"] = start_date.isoformat()
    if end_date:
        params["end_date"] = end_date.isoformat()
    return _get("/api/v1/history", params=params)


def get_features() -> dict[str, Any]:
    return _get("/api/v1/features")


def get_model_info() -> dict[str, Any]:
    return _get("/api/v1/model/info")


def get_metrics() -> dict[str, Any]:
    return _get("/api/v1/metrics")
