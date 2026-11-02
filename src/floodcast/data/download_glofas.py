"""Download GloFAS river discharge data from CDS API."""

import logging
from pathlib import Path

import cdsapi

from floodcast.config import settings

logger = logging.getLogger(__name__)


def download_glofas(
    output_dir: Path | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> Path:
    """Download GloFAS daily river discharge for the study area.

    Args:
        output_dir: Directory to save files. Defaults to data/raw/glofas/.
        year_start: Start year. Defaults to settings.year_start.
        year_end: End year. Defaults to settings.year_end.

    Returns:
        Path to the output directory containing downloaded files.
    """
    output_dir = output_dir or (settings.ROOT_DIR / "data" / "raw" / "glofas")
    output_dir.mkdir(parents=True, exist_ok=True)

    year_start = year_start or settings.year_start
    year_end = year_end or settings.year_end

    client = cdsapi.Client(
        url=settings.cds_api_url,
        key=settings.cds_api_key,
    )

    area = [settings.lat_max, settings.lon_min, settings.lat_min, settings.lon_max]

    for year in range(year_start, year_end + 1):
        output_file = output_dir / f"glofas_{year}.nc"
        if output_file.exists():
            logger.info("Skipping %d — file exists: %s", year, output_file)
            continue

        logger.info("Downloading GloFAS for %d ...", year)

        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]

        client.retrieve(
            "cems-glofas-historical",
            {
                "system_version": "version_4_0",
                "hydrological_model": "lisflood",
                "product_type": "consolidated",
                "variable": "river_discharge_in_the_last_24_hours",
                "hyear": str(year),
                "hmonth": months,
                "hday": days,
                "area": area,
                "format": "grib",
            },
            str(output_file),
        )
        logger.info("Saved: %s", output_file)

    return output_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_glofas()
