"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Generate synthetic raw data for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    discharge = np.random.exponential(scale=5000, size=n)
    rainfall = np.random.gamma(shape=2, scale=10, size=n)

    discharge_90 = np.percentile(discharge, 90)
    flood_label = ((discharge > discharge_90) | (rainfall > 50)).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "daily_rainfall": rainfall,
            "temperature_2m": np.random.normal(28, 3, n),
            "soil_moisture_layer1": np.random.uniform(0.2, 0.5, n),
            "soil_moisture_layer2": np.random.uniform(0.3, 0.6, n),
            "discharge": discharge,
            "wind_u_10m": np.random.normal(0, 2, n),
            "wind_v_10m": np.random.normal(0, 2, n),
            "flood_label": flood_label,
        }
    )
    return df


@pytest.fixture
def fixture_dir() -> Path:
    """Return path to test fixtures directory."""
    path = Path(__file__).parent / "fixtures"
    path.mkdir(exist_ok=True)
    return path
