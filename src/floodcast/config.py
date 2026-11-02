"""Project configuration and settings."""

from pathlib import Path

from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # CDS API
    cds_api_key: str = ""
    cds_api_url: str = "https://cds.climate.copernicus.eu/api"

    # Study area - Jamuna River Basin at Sirajganj
    lat_min: float = 24.0
    lat_max: float = 25.5
    lon_min: float = 89.0
    lon_max: float = 90.5

    # Time range
    year_start: int = 2000
    year_end: int = 2025

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Dashboard
    dashboard_port: int = 8501

    # Model defaults
    sequence_length: int = 14
    forecast_horizon: int = 3

    model_config = {"env_file": str(ROOT_DIR / ".env"), "extra": "ignore"}


settings = Settings()
