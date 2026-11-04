# src/floodcast/data/download_era5.py

"""Download ERA5-Land reanalysis data from CDS API."""

import logging
import time
from pathlib import Path

import cdsapi

from floodcast.config import settings

logger = logging.getLogger(__name__)

VARIABLES = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def download_era5(
    output_dir: Path | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> Path:
    """Download ERA5-Land daily data for the study area.

    Downloads sequentially with all 5 variables per request
    (one month at a time) to minimize total requests.

    Args:
        output_dir: Directory to save files. Defaults to data/raw/era5/.
        year_start: Start year. Defaults to settings.year_start.
        year_end: End year. Defaults to settings.year_end.

    Returns:
        Path to the output directory containing downloaded files.
    """
    from floodcast.config import ROOT_DIR
    output_dir = output_dir or (ROOT_DIR / "data" / "raw" / "era5")
    output_dir.mkdir(parents=True, exist_ok=True)

    year_start = year_start or settings.year_start
    year_end = year_end or settings.year_end

    client = cdsapi.Client()

    area = [settings.lat_max, settings.lon_min, settings.lat_min, settings.lon_max]
    days = [f"{d:02d}" for d in range(1, 32)]

    total = (year_end - year_start + 1) * 12
    done = 0

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            output_file = output_dir / f"era5_land_{year}_{month:02d}.nc"
            if output_file.exists() and output_file.stat().st_size > 0:
                done += 1
                logger.info("[%d/%d] Skipping %d-%02d — exists", done, total, year, month)
                continue

            # Remove empty file from failed attempt
            if output_file.exists():
                output_file.unlink()

            done += 1
            logger.info("[%d/%d] Downloading ERA5-Land %d-%02d ...", done, total, year, month)

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    client.retrieve(
                        "reanalysis-era5-land",
                        {
                            "variable": VARIABLES,
                            "year": str(year),
                            "month": f"{month:02d}",
                            "day": days,
                            "time": "12:00",
                            "area": area,
                            "data_format": "netcdf",
                        },
                        str(output_file),
                    )
                    logger.info("[%d/%d] Saved: %s", done, total, output_file.name)
                    break
                except Exception as e:
                    logger.warning("Attempt %d/%d failed for %d-%02d: %s", attempt, max_retries, year, month, e)
                    if output_file.exists():
                        output_file.unlink()
                    if attempt < max_retries:
                        logger.info("Waiting 60s before retry...")
                        time.sleep(60)
                    else:
                        logger.error("Giving up on %d-%02d after %d attempts", year, month, max_retries)

    return output_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_era5()