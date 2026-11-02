# src/floodcast/features/physics.py

"""Physics-based feature engineering."""

import numpy as np
import pandas as pd


def antecedent_rainfall_index(rainfall: pd.Series, decay: float = 0.9) -> pd.Series:
    """Exponentially weighted past rainfall — models soil memory effect.

    ARI_t = rainfall_t + decay * ARI_{t-1}
    """
    ari = np.zeros(len(rainfall))
    ari[0] = rainfall.iloc[0]
    for i in range(1, len(rainfall)):
        ari[i] = rainfall.iloc[i] + decay * ari[i - 1]
    return pd.Series(ari, index=rainfall.index, name="antecedent_rainfall_index")


def discharge_percentile(discharge: pd.Series) -> pd.Series:
    """Discharge as fraction of historical 90th percentile — relative severity."""
    pct_90 = discharge.quantile(0.90)
    if pct_90 == 0:
        return pd.Series(0, index=discharge.index, name="discharge_pct_90")
    return (discharge / pct_90).rename("discharge_pct_90")


def soil_moisture_anomaly(
    soil_moisture: pd.Series,
    month: pd.Series,
) -> pd.Series:
    """Soil moisture minus monthly mean — ground saturation beyond normal."""
    monthly_mean = soil_moisture.groupby(month).transform("mean")
    return (soil_moisture - monthly_mean).rename("soil_moisture_anomaly")


def rainfall_intensity(rainfall: pd.Series) -> pd.Series:
    """Max 3-day rainfall / 7-day rainfall — burst intensity vs sustained."""
    roll_3 = rainfall.rolling(3, min_periods=1).max()
    roll_7 = rainfall.rolling(7, min_periods=1).sum()
    intensity = roll_3 / roll_7.replace(0, np.nan)
    return intensity.fillna(0).rename("rainfall_intensity")


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all physics-based features to DataFrame."""
    df = df.copy()

    df["antecedent_rainfall_index"] = antecedent_rainfall_index(df["daily_rainfall"])

    if "discharge" in df.columns:
        df["discharge_pct_90"] = discharge_percentile(df["discharge"])

    if "soil_moisture_layer1" in df.columns:
        df["soil_moisture_anomaly"] = soil_moisture_anomaly(
            df["soil_moisture_layer1"], df["date"].dt.month
        )

    df["rainfall_intensity"] = rainfall_intensity(df["daily_rainfall"])

    return df
