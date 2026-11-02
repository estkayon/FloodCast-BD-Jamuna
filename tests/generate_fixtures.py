"""Generate synthetic test fixture data that mimics real data distributions."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_fixture_data(
    n_years: int = 5,
    output_dir: Path | None = None,
) -> Path:
    """Generate synthetic daily data for testing.

    Creates a parquet file with realistic distributions for all features.

    Args:
        n_years: Number of years of data to generate.
        output_dir: Directory to save the fixture. Defaults to tests/fixtures/.

    Returns:
        Path to generated parquet file.
    """
    output_dir = output_dir or (Path(__file__).parent / "fixtures")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=365 * n_years, freq="D")
    n = len(dates)

    # Day of year for seasonal patterns
    doy = dates.dayofyear

    # Monsoon season (June-October) has more rain
    is_monsoon = ((dates.month >= 6) & (dates.month <= 10)).astype(float)

    # Seasonal rainfall: higher in monsoon
    base_rain = 2 + 15 * is_monsoon
    daily_rainfall = np.maximum(0, np.random.gamma(shape=2, scale=base_rain))

    # Temperature: seasonal cycle (warmer in summer)
    temperature_2m = 295 + 8 * np.sin(2 * np.pi * (doy - 100) / 365) + np.random.normal(0, 1.5, n)

    # Soil moisture: higher during monsoon
    soil_moisture_layer1 = 0.25 + 0.15 * is_monsoon + np.random.normal(0, 0.03, n)
    soil_moisture_layer1 = np.clip(soil_moisture_layer1, 0.05, 0.60)

    soil_moisture_layer2 = 0.35 + 0.10 * is_monsoon + np.random.normal(0, 0.02, n)
    soil_moisture_layer2 = np.clip(soil_moisture_layer2, 0.15, 0.65)

    # Discharge: correlated with rainfall (lagged)
    discharge_base = 3000 + 5000 * is_monsoon
    rainfall_effect = np.convolve(daily_rainfall, np.ones(7) / 7, mode="same") * 200
    discharge = np.maximum(100, discharge_base + rainfall_effect + np.random.normal(0, 500, n))

    # Wind
    wind_u_10m = np.random.normal(1.5, 2.0, n)
    wind_v_10m = np.random.normal(0.5, 1.5, n)

    df = pd.DataFrame({
        "date": dates,
        "daily_rainfall": daily_rainfall,
        "temperature_2m": temperature_2m,
        "soil_moisture_layer1": soil_moisture_layer1,
        "soil_moisture_layer2": soil_moisture_layer2,
        "discharge": discharge,
        "wind_u_10m": wind_u_10m,
        "wind_v_10m": wind_v_10m,
    })

    # Flood labels
    discharge_90 = df["discharge"].quantile(0.90)
    df["flood_label"] = ((df["discharge"] > discharge_90) | (df["daily_rainfall"] > 50)).astype(int)

    output_path = output_dir / "synthetic_raw.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Generated {len(df)} rows -> {output_path}")
    print(f"Flood events: {df['flood_label'].sum()} ({100*df['flood_label'].mean():.1f}%)")

    return output_path


if __name__ == "__main__":
    generate_fixture_data()
