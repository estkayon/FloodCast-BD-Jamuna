"""Download CHIRPS daily rainfall data via HTTP."""

import logging
from pathlib import Path

import requests
import xarray as xr

from floodcast.config import settings

logger = logging.getLogger(__name__)

CHIRPS_BASE_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05"
)


def download_chirps(
    output_dir: Path | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> Path:
    """Download CHIRPS daily rainfall for each year.

    Args:
        output_dir: Directory to save files. Defaults to data/raw/chirps/.
        year_start: Start year. Defaults to settings.year_start.
        year_end: End year. Defaults to settings.year_end.

    Returns:
        Path to the output directory containing downloaded files.
    """
    output_dir = output_dir or (settings.ROOT_DIR / "data" / "raw" / "chirps")
    output_dir.mkdir(parents=True, exist_ok=True)

    year_start = year_start or settings.year_start
    year_end = year_end or settings.year_end

    for year in range(year_start, year_end + 1):
        output_file = output_dir / f"chirps_{year}.nc"
        if output_file.exists():
            logger.info("Skipping %d — file exists: %s", year, output_file)
            continue

        url = f"{CHIRPS_BASE_URL}/chirps-v2.0.{year}.days_p05.nc"
        logger.info("Downloading CHIRPS for %d from %s ...", year, url)

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        tmp_file = output_dir / f"chirps_{year}_full.nc"
        with open(tmp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Clip to study area
        logger.info("Clipping to study area ...")
        ds = xr.open_dataset(tmp_file)
        ds_clipped = ds.sel(
            latitude=slice(settings.lat_max, settings.lat_min),
            longitude=slice(settings.lon_min, settings.lon_max),
        )
        ds_clipped.to_netcdf(output_file)
        ds.close()
        ds_clipped.close()
        tmp_file.unlink()

        logger.info("Saved: %s", output_file)

    return output_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_chirps()
