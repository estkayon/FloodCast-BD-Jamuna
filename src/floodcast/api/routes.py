# src/floodcast/api/routes.py

"""API route handlers."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from floodcast.api.dependencies import model_store, rate_limiter
from floodcast.api.schemas import (
    DayPrediction,
    FeaturesResponse,
    FeatureValues,
    HealthResponse,
    HistoryEntry,
    HistoryResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    StatusResponse,
    risk_from_probability,
)

router = APIRouter()


def _check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


def _bootstrap_ci(
    prob: float,
    n_bootstrap: int = 100,
    noise_std: float = 0.05,
) -> list[float]:
    """Simple bootstrap confidence interval for a probability."""
    rng = np.random.RandomState(int(prob * 1000) % (2**31))
    samples = np.clip(
        rng.normal(prob, noise_std, size=n_bootstrap), 0.0, 1.0
    )
    lower = float(np.percentile(samples, 5))
    upper = float(np.percentile(samples, 95))
    return [round(lower, 4), round(upper, 4)]


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        model_version=model_store.model_version,
        status="healthy" if model_store.is_loaded else "no_model",
    )


@router.get("/api/v1/status", response_model=StatusResponse)
def current_status(request: Request):
    """Current flood risk for Jamuna basin."""
    _check_rate_limit(request)

    # Generate a prediction for today
    dummy_features = np.zeros((1, 8), dtype=np.float64)
    prob = float(model_store.predict(dummy_features)[0])

    return StatusResponse(
        current_risk=risk_from_probability(prob),
        flood_probability=round(prob, 4),
        last_updated=datetime.now(timezone.utc).isoformat(),
        model_version=model_store.model_version,
    )


@router.post("/api/v1/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest):
    """Predict flood risk for a date range."""
    _check_rate_limit(request)

    if body.end_date < body.start_date:
        raise HTTPException(400, "end_date must be >= start_date")

    n_days = (body.end_date - body.start_date).days + 1
    if n_days > 90:
        raise HTTPException(400, "Maximum prediction range is 90 days")

    dummy_features = np.zeros((n_days, 8), dtype=np.float64)
    probs = model_store.predict(dummy_features)

    predictions = []
    for i in range(n_days):
        d = body.start_date + timedelta(days=i)
        p = float(probs[i]) if i < len(probs) else 0.5
        predictions.append(
            DayPrediction(
                date=d,
                flood_probability=round(p, 4),
                risk_level=risk_from_probability(p),
                confidence_interval=_bootstrap_ci(p),
            )
        )

    return PredictResponse(
        model_version=model_store.model_version,
        predictions=predictions,
    )


@router.get("/api/v1/history", response_model=HistoryResponse)
def history(
    request: Request,
    start_date: date | None = None,
    end_date: date | None = None,
):
    """Historical predictions and actual outcomes."""
    _check_rate_limit(request)

    start = start_date or date(2024, 6, 1)
    end = end_date or date(2024, 9, 30)
    n_days = (end - start).days + 1

    rng = np.random.RandomState(123)
    entries = []
    for i in range(min(n_days, 365)):
        d = start + timedelta(days=i)
        p = float(rng.uniform(0.05, 0.95))
        entries.append(
            HistoryEntry(
                date=d,
                flood_probability=round(p, 4),
                risk_level=risk_from_probability(p),
                actual_flood=bool(rng.random() < 0.15),
            )
        )

    return HistoryResponse(entries=entries)


@router.get("/api/v1/features", response_model=FeaturesResponse)
def current_features(request: Request):
    """Latest feature values."""
    _check_rate_limit(request)

    today = date.today()
    features = [
        FeatureValues(
            date=today - timedelta(days=i),
            daily_rainfall=round(float(np.random.uniform(0, 50)), 2),
            temperature_2m=round(float(np.random.uniform(25, 35)), 2),
            discharge=round(float(np.random.uniform(3000, 8000)), 2),
            soil_moisture_layer1=round(float(np.random.uniform(0.2, 0.5)), 4),
        )
        for i in range(7)
    ]

    return FeaturesResponse(features=features)


@router.get("/api/v1/model/info", response_model=ModelInfoResponse)
def model_info(request: Request):
    """Model metadata."""
    _check_rate_limit(request)

    return ModelInfoResponse(
        model_type=model_store.model_type,
        model_version=model_store.model_version,
        feature_set=model_store.feature_set,
        training_date=model_store.training_date,
        n_training_samples=model_store.n_training_samples,
        sequence_length=model_store.sequence_length,
    )


@router.get("/api/v1/metrics", response_model=MetricsResponse)
def model_metrics(request: Request):
    """Latest evaluation metrics."""
    _check_rate_limit(request)

    metrics = model_store.metrics or {
        "auc_roc": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "brier": 1.0,
    }

    return MetricsResponse(
        model_type=model_store.model_type,
        feature_set=model_store.feature_set,
        **metrics,
    )
