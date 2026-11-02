"""Process raw data: spatial extraction, temporal alignment, merge."""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from floodcast.config import settings

logger = logging.getLogger(__name__)


def process_era5(raw_dir: Path) -> pd.DataFrame:
    """Process ERA5-Land netCDF files into daily DataFrame."""
    files = sorted(raw_dir.glob("era5_land_*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found in {raw_dir}")

    datasets = []
    for f in files:
        ds = xr.open_dataset(f)
        # Spatial mean over study area
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
    # Rename only columns that exist
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


def process_chirps(raw_dir: Path) -> pd.DataFrame:
    """Process CHIRPS netCDF files into daily rainfall DataFrame."""
    files = sorted(raw_dir.glob("chirps_*.nc"))
    if not files:
        raise FileNotFoundError(f"No CHIRPS files found in {raw_dir}")

    datasets = []
    for f in files:
        ds = xr.open_dataset(f)
        # Spatial mean over study area
        ds_mean = ds.mean(dim=["latitude", "longitude"])
        df = ds_mean.to_dataframe().reset_index()
        datasets.append(df)
        ds.close()

    combined = pd.concat(datasets, ignore_index=True)

    if "precip" in combined.columns:
        combined = combined.rename(columns={"precip": "daily_rainfall"})

    if "time" in combined.columns:
        combined["date"] = pd.to_datetime(combined["time"]).dt.date
    combined["date"] = pd.to_datetime(combined["date"])

    combined = combined[["date", "daily_rainfall"]].groupby("date").mean().reset_index()
    # Replace negative rainfall with 0
    combined["daily_rainfall"] = combined["daily_rainfall"].clip(lower=0)

    logger.info("CHIRPS processed: %d rows", len(combined))
    return combined


def process_glofas(raw_dir: Path) -> pd.DataFrame:
    """Process GloFAS GRIB/netCDF files into daily discharge DataFrame."""
    files = sorted(raw_dir.glob("glofas_*.nc")) + sorted(raw_dir.glob("glofas_*.grib"))
    if not files:
        raise FileNotFoundError(f"No GloFAS files found in {raw_dir}")

    datasets = []
    for f in files:
        if f.suffix == ".grib":
            ds = xr.open_dataset(f, engine="cfgrib")
        else:
            ds = xr.open_dataset(f)
        ds_mean = ds.mean(dim=["latitude", "longitude"])
        df = ds_mean.to_dataframe().reset_index()
        datasets.append(df)
        ds.close()

    combined = pd.concat(datasets, ignore_index=True)

    # GloFAS variable name can vary
    discharge_cols = [c for c in combined.columns if "dis" in c.lower()]
    if discharge_cols:
        combined = combined.rename(columns={discharge_cols[0]: "discharge"})

    if "time" in combined.columns:
        combined["date"] = pd.to_datetime(combined["time"]).dt.date
    elif "valid_time" in combined.columns:
        combined["date"] = pd.to_datetime(combined["valid_time"]).dt.date

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined[["date", "discharge"]].groupby("date").mean().reset_index()
    combined["discharge"] = combined["discharge"].clip(lower=0)

    logger.info("GloFAS processed: %d rows", len(combined))
    return combined


def create_flood_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary flood labels from discharge data.

    Label = 1 if discharge > 90th percentile OR daily_rainfall > 50mm.
    """
    discharge_90 = df["discharge"].quantile(0.90)
    df["flood_label"] = (
        (df["discharge"] > discharge_90) | (df["daily_rainfall"] > 50)
    ).astype(int)

    n_flood = df["flood_label"].sum()
    logger.info(
        "Flood labels created: %d flood days (%.1f%%) out of %d total, "
        "discharge 90th pct = %.1f",
        n_flood, 100 * n_flood / len(df), len(df), discharge_90,
    )
    return df


def merge_datasets(
    era5_df: pd.DataFrame,
    chirps_df: pd.DataFrame,
    glofas_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources on date, forward-fill small gaps."""
    merged = chirps_df.merge(glofas_df, on="date", how="outer")
    merged = merged.merge(era5_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Forward fill gaps up to 3 days
    merged = merged.ffill(limit=3)

    # Drop rows still having NaN after fill
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
    """Run full processing pipeline: process each source, merge, label, save.

    Returns:
        Path to the output parquet file.
    """
    raw_base_dir = raw_base_dir or (settings.ROOT_DIR / "data" / "raw")
    output_dir = output_dir or (settings.ROOT_DIR / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    era5_df = process_era5(raw_base_dir / "era5")
    chirps_df = process_chirps(raw_base_dir / "chirps")
    glofas_df = process_glofas(raw_base_dir / "glofas")

    merged = merge_datasets(era5_df, chirps_df, glofas_df)
    merged = create_flood_labels(merged)

    output_path = output_dir / "features_raw.parquet"
    merged.to_parquet(output_path, index=False)
    logger.info("Saved processed data: %s (%d rows)", output_path, len(merged))

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_processing()
