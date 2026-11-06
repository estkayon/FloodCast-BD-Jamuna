# src/floodcast/data/process.py
"""Process raw data: spatial extraction, temporal alignment, merge."""

from floodcast.config import settings, ROOT_DIR
import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def process_era5(raw_dir: Path) -> pd.DataFrame:
    """Process ERA5-Land netCDF files into daily DataFrame."""
    files = sorted(raw_dir.glob("era5_land_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {raw_dir}")

    datasets = []
    for f in files:
        ds = xr.open_dataset(f, engine="netcdf4")
        ds_mean = ds.mean(dim=["latitude", "longitude"])
        df = ds_mean.to_dataframe().reset_index()
        datasets.append(df)
        ds.close()

    combined = pd.concat(datasets, ignore_index=True)

    rename_map = {
        "t2m": "temperature_2m",
        "swvl1": "soil_moisture_layer1",
        "swvl2": "soil_moisture_layer2",
        "u10": "wind_u_10m",
        "v10": "wind_v_10m",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in combined.columns}
    combined = combined.rename(columns=rename_map)

    if "time" in combined.columns:
        combined["date"] = pd.to_datetime(combined["time"]).dt.date
    elif "valid_time" in combined.columns:
        combined["date"] = pd.to_datetime(combined["valid_time"]).dt.date

    combined["date"] = pd.to_datetime(combined["date"])

    keep_cols = ["date", "temperature_2m", "soil_moisture_layer1",
                 "soil_moisture_layer2", "wind_u_10m", "wind_v_10m"]
    keep_cols = [c for c in keep_cols if c in combined.columns]
    combined = combined[keep_cols].groupby("date").mean().reset_index()

    logger.info("ERA5 processed: %d rows", len(combined))
    return combined


def process_runoff(raw_dir: Path) -> pd.DataFrame:
    """Process ERA5-Land runoff netCDF files into daily runoff DataFrame."""
    files = sorted(raw_dir.glob("runoff_*.nc"))
    if not files:
        raise FileNotFoundError(f"No runoff files found in {raw_dir}")

    datasets = []
    for f in files:
        ds = xr.open_dataset(f, engine="netcdf4")
        ds_mean = ds.mean(dim=["latitude", "longitude"])
        df = ds_mean.to_dataframe().reset_index()
        datasets.append(df)
        ds.close()

    combined = pd.concat(datasets, ignore_index=True)

    # ERA5 runoff variable name can be 'ro' or 'runoff'
    runoff_cols = [c for c in combined.columns if c in ("ro", "runoff")]
    if runoff_cols:
        combined = combined.rename(columns={runoff_cols[0]: "runoff"})

    if "time" in combined.columns:
        combined["date"] = pd.to_datetime(combined["time"]).dt.date
    elif "valid_time" in combined.columns:
        combined["date"] = pd.to_datetime(combined["valid_time"]).dt.date

    combined["date"] = pd.to_datetime(combined["date"])
    # Convert from m to mm for consistency
    combined["runoff"] = combined["runoff"] * 1000.0
    combined = combined[["date", "runoff"]].groupby("date").mean().reset_index()
    combined["runoff"] = combined["runoff"].clip(lower=0)

    logger.info("Runoff processed: %d rows", len(combined))
    return combined


def create_flood_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary flood labels from runoff data.

    Label = 1 if runoff > 90th percentile.
    """
    runoff_90 = df["runoff"].quantile(0.90)
    df["flood_label"] = (df["runoff"] > runoff_90).astype(int)

    n_flood = df["flood_label"].sum()
    logger.info(
        "Flood labels created: %d flood days (%.1f%%) out of %d total, "
        "runoff 90th pct = %.4f mm",
        n_flood, 100 * n_flood / len(df), len(df), runoff_90,
    )
    return df


def merge_datasets(
    era5_df: pd.DataFrame,
    runoff_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ERA5 and runoff data on date, forward-fill small gaps."""
    merged = runoff_df.merge(era5_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    merged = merged.ffill(limit=3)

    before = len(merged)
    merged = merged.dropna()
    after = len(merged)
    if before > after:
        logger.warning("Dropped %d rows with NaN after gap filling", before - after)

    logger.info("Merged dataset: %d rows, columns: %s", len(merged), list(merged.columns))
    return merged


def run_processing(
    raw_base_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run full processing pipeline."""
    raw_base_dir = raw_base_dir or (ROOT_DIR / "data" / "raw")
    output_dir = output_dir or (ROOT_DIR / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    era5_df = process_era5(raw_base_dir / "era5")
    runoff_df = process_runoff(raw_base_dir / "runoff")

    merged = merge_datasets(era5_df, runoff_df)
    merged = create_flood_labels(merged)

    output_path = output_dir / "features_raw.parquet"
    merged.to_parquet(output_path, index=False)
    logger.info("Saved processed data: %s (%d rows)", output_path, len(merged))

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_processing()