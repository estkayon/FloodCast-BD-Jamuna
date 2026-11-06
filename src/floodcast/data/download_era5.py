# src/floodcast/data/download_era5.py
"""Re-download ERA5-Land data as unarchived NetCDF (not ZIP)."""

import logging

import cdsapi

from floodcast.config import ROOT_DIR, Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_era5() -> None:
    cfg = Settings()
    out_dir = ROOT_DIR / "data" / "raw" / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    months = [f"{m:02d}" for m in range(1, 13)]
    total = (cfg.year_end - cfg.year_start + 1) * 12
    counter = 0

    for year in range(cfg.year_start, cfg.year_end + 1):
        for mm in months:
            counter += 1
            fname = f"era5_land_{year}_{mm}.nc"
            fpath = out_dir / fname
            if fpath.exists() and fpath.stat().st_size > 1000:
                logger.info("[%d/%d] Exists: %s", counter, total, fname)
                continue
            logger.info("[%d/%d] Downloading %s ...", counter, total, fname)
            request = {
                "variable": [
                    "2m_temperature",
                    "volumetric_soil_water_layer_1",
                    "volumetric_soil_water_layer_2",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year": str(year),
                "month": mm,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00"],
                "area": [cfg.lat_max, cfg.lon_min, cfg.lat_min, cfg.lon_max],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            try:
                client.retrieve("reanalysis-era5-land", request, str(fpath))
                size_kb = fpath.stat().st_size / 1024
                logger.info("[%d/%d] Saved: %s (%.0f KB)", counter, total, fname, size_kb)
            except Exception as e:
                logger.error("[%d/%d] FAILED %s: %s", counter, total, fname, e)
                if fpath.exists() and fpath.stat().st_size == 0:
                    fpath.unlink()


if __name__ == "__main__":
    download_era5()
