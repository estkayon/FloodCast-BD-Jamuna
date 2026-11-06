# tests/test_features.py

"""Phase 3: Feature engineering tests."""

import numpy as np
import pandas as pd
import pytest
from floodcast.features.build import load_feature_sets, select_features
from floodcast.features.engineer import engineer_features
from floodcast.features.physics import (
    add_physics_features,
    antecedent_runoff_index,
    runoff_percentile,
    soil_moisture_anomaly,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Raw processed data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    is_monsoon = ((dates.month >= 6) & (dates.month <= 10)).astype(float)
    runoff = np.maximum(0, np.random.gamma(2, 5 + 10 * is_monsoon))
    return pd.DataFrame({
        "date": dates,
        "runoff": runoff,
        "temperature_2m": np.random.normal(300, 3, n),
        "soil_moisture_layer1": np.random.uniform(0.2, 0.5, n),
        "soil_moisture_layer2": np.random.uniform(0.3, 0.6, n),
        "wind_u_10m": np.random.normal(0, 2, n),
        "wind_v_10m": np.random.normal(0, 2, n),
        "flood_label": np.random.randint(0, 2, n),
    })


class TestEngineerFeatures:
    """Tests for basic feature engineering."""

    def test_output_has_lag_features(self, raw_df):
        result = engineer_features(raw_df)
        assert "runoff_lag_1d" in result.columns
        assert "runoff_lag_7d" in result.columns
        assert "runoff_lag_3d" in result.columns

    def test_output_has_rolling_features(self, raw_df):
        result = engineer_features(raw_df)
        assert "runoff_sum_7d" in result.columns
        assert "runoff_mean_14d" in result.columns

    def test_output_has_calendar_features(self, raw_df):
        result = engineer_features(raw_df)
        assert "month" in result.columns
        assert "day_of_year" in result.columns
        assert "is_monsoon" in result.columns

    def test_no_nans_after_engineering(self, raw_df):
        result = engineer_features(raw_df)
        assert result.isna().sum().sum() == 0

    def test_rows_dropped_from_lagging(self, raw_df):
        result = engineer_features(raw_df)
        assert len(result) < len(raw_df)


class TestPhysicsFeatures:
    """Tests for physics-based features."""

    def test_antecedent_runoff_positive(self, raw_df):
        ari = antecedent_runoff_index(raw_df["runoff"])
        assert (ari >= 0).all()

    def test_runoff_percentile_range(self, raw_df):
        pct = runoff_percentile(raw_df["runoff"])
        assert pct.min() >= 0

    def test_soil_moisture_anomaly_mean_near_zero(self, raw_df):
        anomaly = soil_moisture_anomaly(
            raw_df["soil_moisture_layer1"], raw_df["date"].dt.month
        )
        assert abs(anomaly.mean()) < 0.1

    def test_add_physics_features_columns(self, raw_df):
        result = add_physics_features(raw_df)
        assert "antecedent_runoff_index" in result.columns
        assert "runoff_pct_90" in result.columns
        assert "soil_moisture_anomaly" in result.columns


class TestFeatureSets:
    """Tests for feature set selection."""

    def test_load_feature_sets(self):
        sets = load_feature_sets()
        assert "set_a" in sets
        assert "set_b" in sets
        assert "set_c" in sets
        assert "runoff" in sets["set_a"]

    def test_set_a_basic_features(self, raw_df):
        load_feature_sets()
        engineered = engineer_features(raw_df)
        engineered = add_physics_features(engineered)
        selected = select_features(engineered, "set_a")
        assert "runoff" in selected.columns
        assert "flood_label" in selected.columns

    def test_set_c_is_largest(self, raw_df):
        load_feature_sets()
        engineered = engineer_features(raw_df)
        engineered = add_physics_features(engineered)
        a = select_features(engineered, "set_a")
        c = select_features(engineered, "set_c")
        assert len(c.columns) >= len(a.columns)
