"""Phase 2: Data pipeline tests."""


import numpy as np
import pandas as pd
import pytest
from floodcast.data.process import create_flood_labels, merge_datasets
from floodcast.data.validate import validate_dataframe


@pytest.fixture
def era5_df() -> pd.DataFrame:
    """Synthetic ERA5-like data."""
    np.random.seed(10)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": dates,
        "temperature_2m": np.random.normal(300, 5, 100),
        "soil_moisture_layer1": np.random.uniform(0.2, 0.5, 100),
        "soil_moisture_layer2": np.random.uniform(0.3, 0.6, 100),
        "wind_u_10m": np.random.normal(0, 2, 100),
        "wind_v_10m": np.random.normal(0, 2, 100),
    })


@pytest.fixture
def chirps_df() -> pd.DataFrame:
    """Synthetic CHIRPS-like data."""
    np.random.seed(20)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": dates,
        "daily_rainfall": np.random.gamma(2, 10, 100),
    })


@pytest.fixture
def glofas_df() -> pd.DataFrame:
    """Synthetic GloFAS-like data."""
    np.random.seed(30)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": dates,
        "discharge": np.random.exponential(5000, 100),
    })


class TestMergeDatasets:
    """Tests for the merge function."""

    def test_merge_shape(self, era5_df, chirps_df, glofas_df):
        merged = merge_datasets(era5_df, chirps_df, glofas_df)
        assert len(merged) > 0
        assert "date" in merged.columns
        assert "daily_rainfall" in merged.columns
        assert "discharge" in merged.columns
        assert "temperature_2m" in merged.columns

    def test_merge_no_nulls(self, era5_df, chirps_df, glofas_df):
        merged = merge_datasets(era5_df, chirps_df, glofas_df)
        assert merged.isna().sum().sum() == 0

    def test_merge_sorted_dates(self, era5_df, chirps_df, glofas_df):
        merged = merge_datasets(era5_df, chirps_df, glofas_df)
        dates = merged["date"].values
        assert (dates[1:] >= dates[:-1]).all()


class TestFloodLabels:
    """Tests for flood label creation."""

    def test_labels_binary(self, era5_df, chirps_df, glofas_df):
        merged = merge_datasets(era5_df, chirps_df, glofas_df)
        labeled = create_flood_labels(merged)
        assert "flood_label" in labeled.columns
        assert labeled["flood_label"].isin([0, 1]).all()

    def test_labels_not_all_same(self, era5_df, chirps_df, glofas_df):
        merged = merge_datasets(era5_df, chirps_df, glofas_df)
        labeled = create_flood_labels(merged)
        assert labeled["flood_label"].nunique() > 1


class TestValidation:
    """Tests for data validation."""

    def test_valid_data_passes(self, sample_raw_data):
        report = validate_dataframe(sample_raw_data)
        assert report.total_rows == 100
        assert report.duplicate_dates == 0

    def test_missing_data_detected(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "daily_rainfall": [1, 2, np.nan, np.nan, np.nan, np.nan, 7, 8, 9, 10],
            "discharge": range(10),
        })
        report = validate_dataframe(df)
        assert report.missing_counts["daily_rainfall"] == 4

    def test_out_of_range_detected(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=5),
            "daily_rainfall": [0, 10, 600, 20, 5],
            "discharge": [100, 200, -50, 300, 400],
        })
        report = validate_dataframe(df)
        assert report.out_of_range["daily_rainfall"] == 1
        assert report.out_of_range["discharge"] == 1
