# src/floodcast/features/engineer.py

"""Feature engineering: lag, rolling, calendar features."""

import pandas as pd


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged rainfall and discharge features."""
    for lag in [1, 2, 3, 5, 7]:
        df[f"rainfall_lag_{lag}d"] = df["daily_rainfall"].shift(lag)
        if "discharge" in df.columns:
            df[f"discharge_lag_{lag}d"] = df["discharge"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling sum and mean features."""
    for window in [3, 7, 14]:
        df[f"rainfall_sum_{window}d"] = (
            df["daily_rainfall"].rolling(window, min_periods=1).sum()
        )
        df[f"rainfall_mean_{window}d"] = (
            df["daily_rainfall"].rolling(window, min_periods=1).mean()
        )
        if "discharge" in df.columns:
            df[f"discharge_mean_{window}d"] = (
                df["discharge"].rolling(window, min_periods=1).mean()
            )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, day_of_year, and monsoon flag."""
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_monsoon"] = df["month"].between(6, 10).astype(int)
    return df


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rate-of-change features for discharge."""
    if "discharge" not in df.columns:
        return df
    for window in [1, 3]:
        df[f"discharge_change_{window}d"] = df["discharge"].diff(window)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline.

    Args:
        df: Raw processed DataFrame with date, daily_rainfall, discharge, etc.

    Returns:
        DataFrame with all engineered features, NaN rows from lagging dropped.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_change_features(df)

    # Drop rows with NaN from lagging (first 14 rows max)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if before > after:
        print(f"Dropped {before - after} rows from lag NaNs")

    return df
