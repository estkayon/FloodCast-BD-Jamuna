# src/floodcast/features/engineer.py
"""Feature engineering: lag, rolling, calendar features."""

import pandas as pd


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged runoff features."""
    for lag in [1, 2, 3, 5, 7]:
        if "runoff" in df.columns:
            df[f"runoff_lag_{lag}d"] = df["runoff"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean features for runoff."""
    for window in [3, 7, 14]:
        if "runoff" in df.columns:
            df[f"runoff_mean_{window}d"] = (
                df["runoff"].rolling(window, min_periods=1).mean()
            )
            df[f"runoff_sum_{window}d"] = (
                df["runoff"].rolling(window, min_periods=1).sum()
            )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, day_of_year, and monsoon flag."""
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_monsoon"] = df["month"].between(6, 10).astype(int)
    return df


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rate-of-change features for runoff."""
    if "runoff" not in df.columns:
        return df
    for window in [1, 3]:
        df[f"runoff_change_{window}d"] = df["runoff"].diff(window)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_change_features(df)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    if before > after:
        print(f"Dropped {before - after} rows from lag NaNs")

    return df