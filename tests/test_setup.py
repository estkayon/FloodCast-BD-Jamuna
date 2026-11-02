"""Phase 1: Project setup tests."""


import pandas as pd
from floodcast import __version__
from floodcast.config import ROOT_DIR, Settings


def test_version():
    """Version string should be 1.0.0."""
    assert __version__ == "1.0.0"


def test_settings_defaults():
    """Settings should load with correct defaults."""
    s = Settings()
    assert s.lat_min == 24.0
    assert s.lat_max == 25.5
    assert s.lon_min == 89.0
    assert s.lon_max == 90.5
    assert s.year_start == 2000
    assert s.year_end == 2025
    assert s.sequence_length == 14
    assert s.forecast_horizon == 3


def test_project_structure():
    """Key directories and files should exist."""
    assert (ROOT_DIR / "src" / "floodcast").is_dir()
    assert (ROOT_DIR / "tests").is_dir()
    assert (ROOT_DIR / "params.yaml").is_file()
    assert (ROOT_DIR / "requirements.txt").is_file()


def test_sample_data_fixture(sample_raw_data: pd.DataFrame):
    """Sample data fixture should have expected shape and columns."""
    assert len(sample_raw_data) == 100
    expected_cols = [
        "date", "daily_rainfall", "temperature_2m",
        "soil_moisture_layer1", "soil_moisture_layer2",
        "discharge", "wind_u_10m", "wind_v_10m", "flood_label",
    ]
    for col in expected_cols:
        assert col in sample_raw_data.columns
    assert sample_raw_data["flood_label"].isin([0, 1]).all()
