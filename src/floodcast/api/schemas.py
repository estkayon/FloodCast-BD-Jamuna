# src/floodcast/api/schemas.py

"""Pydantic schemas for the FloodCast API."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


def risk_from_probability(prob: float) -> RiskLevel:
    """Map flood probability to risk level."""
    if prob < 0.25:
        return RiskLevel.LOW
    elif prob < 0.50:
        return RiskLevel.MODERATE
    elif prob < 0.75:
        return RiskLevel.HIGH
    else:
        return RiskLevel.VERY_HIGH


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str = "healthy"
    model_version: str
    api_version: str = "1.0.0"


class PredictRequest(BaseModel):
    start_date: date
    end_date: date
    include_features: bool = False


class DayPrediction(BaseModel):
    date: date
    flood_probability: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    confidence_interval: list[float] = Field(min_length=2, max_length=2)


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    basin: str = "jamuna_sirajganj"
    model_version: str
    predictions: list[DayPrediction]
    disclaimer: str = "Research prototype. Not for emergency decisions."


class StatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    basin: str = "jamuna_sirajganj"
    current_risk: RiskLevel
    flood_probability: float
    last_updated: str
    model_version: str


class FeatureValues(BaseModel):
    date: date
    daily_rainfall: float | None = None
    temperature_2m: float | None = None
    discharge: float | None = None
    soil_moisture_layer1: float | None = None


class FeaturesResponse(BaseModel):
    basin: str = "jamuna_sirajganj"
    features: list[FeatureValues]


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_type: str
    model_version: str
    feature_set: str
    training_date: str
    n_training_samples: int
    sequence_length: int | None = None


class MetricsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_type: str
    feature_set: str
    auc_roc: float
    f1: float
    precision: float
    recall: float
    brier: float


class HistoryEntry(BaseModel):
    date: date
    flood_probability: float
    risk_level: RiskLevel
    actual_flood: bool | None = None


class HistoryResponse(BaseModel):
    basin: str = "jamuna_sirajganj"
    entries: list[HistoryEntry]
