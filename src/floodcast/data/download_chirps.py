# src/floodcast/data/download_chirps.py

"""Download CHIRPS rainfall data from NOAA ERDDAP."""

import logging
import time
from pathlib import Path

import requests

from floodcast.config import settings

logger = logging.getLogger(__name__)

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/chirps20GlobalDailyP05.nc"


def download_chirps(
    output_dir: Path | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> Path:
    """Download CHIRPS daily rainfall for the study area via NOAA ERDDAP.

    Downloads one year at a time, subsetted to the study area.

    Args:
        output_dir: Directory to save files. Defaults to data/raw/chirps/.
        year_start: Start year. Defaults to settings.year_start.
        year_end: End year. Defaults to settings.year_end.

    Returns:
        Path to the output directory containing downloaded files.
    """
    from floodcast.config import ROOT_DIR
    output_dir = output_dir or (ROOT_DIR / "data" / "raw" / "chirps")
    output_dir.mkdir(parents=True, exist_ok=True)

    year_start = year_start or settings.year_start
    year_end = year_end or settings.year_end

    total = year_end - year_start + 1
    done = 0

    for year in range(year_start, year_end + 1):
        output_file = output_dir / f"chirps_{year}.nc"
        if output_file.exists() and output_file.stat().st_size > 0:
            done += 1
            logger.info("[%d/%d] Skipping %d — exists", done, total, year)
            continue

        if output_file.exists():
            output_file.unlink()

        done += 1
        logger.info("[%d/%d] Downloading CHIRPS %d ...", done, total, year)

        # ERDDAP subset query: precip[time][latitude][longitude]
        url = (
            f"{ERDDAP_BASE}?precip"
            f"[({year}-01-01T00:00:00Z):1:({year}-12-31T00:00:00Z)]"
            f"[({settings.lat_min}):1:({settings.lat_max})]"
            f"[({settings.lon_min}):1:({settings.lon_max})]"
        )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, stream=True, timeout=600)
                response.raise_for_status()

                with open(output_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info("[%d/%d] Saved: %s", done, total, output_file.name)
                break
            except Exception as e:
                logger.warning(
                    "Attempt %d/%d failed for %d: %s",
                    attempt, max_retries, year, e,
                )
                if output_file.exists():
                    output_file.unlink()
                if attempt < max_retries:
                    logger.info("Waiting 30s before retry...")
                    time.sleep(30)
                else:
                    logger.error(
                        "Giving up on %d after %d attempts", year, max_retries
                    )

    return output_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_chirps()
