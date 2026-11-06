# src/floodcast/features/physics.py
"""Physics-based feature engineering."""

import numpy as np
import pandas as pd


def runoff_percentile(runoff: pd.Series) -> pd.Series:
    """Runoff as fraction of historical 90th percentile — relative severity."""
    pct_90 = runoff.quantile(0.90)
    if pct_90 == 0:
        return pd.Series(0, index=runoff.index, name="runoff_pct_90")
    return (runoff / pct_90).rename("runoff_pct_90")


def soil_moisture_anomaly(
    soil_moisture: pd.Series,
    month: pd.Series,
) -> pd.Series:
    """Soil moisture minus monthly mean — ground saturation beyond normal."""
    monthly_mean = soil_moisture.groupby(month).transform("mean")
    return (soil_moisture - monthly_mean).rename("soil_moisture_anomaly")


def antecedent_runoff_index(runoff: pd.Series, decay: float = 0.9) -> pd.Series:
    """Exponentially weighted past runoff — models hydrological memory."""
    ari = np.zeros(len(runoff))
    ari[0] = runoff.iloc[0]
    for i in range(1, len(runoff)):
        ari[i] = runoff.iloc[i] + decay * ari[i - 1]
    return pd.Series(ari, index=runoff.index, name="antecedent_runoff_index")


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all physics-based features to DataFrame."""
    df = df.copy()

    if "runoff" in df.columns:
        df["runoff_pct_90"] = runoff_percentile(df["runoff"])
        df["antecedent_runoff_index"] = antecedent_runoff_index(df["runoff"])

    if "soil_moisture_layer1" in df.columns:
        df["soil_moisture_anomaly"] = soil_moisture_anomaly(
            df["soil_moisture_layer1"], df["date"].dt.month
        )

    return df
