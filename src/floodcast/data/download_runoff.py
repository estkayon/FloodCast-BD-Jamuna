# src/floodcast/data/download_runoff.py
"""Download ERA5-Land total runoff (replaces GloFAS discharge)."""

import logging

import cdsapi

from floodcast.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_runoff() -> None:
    from floodcast.config import ROOT_DIR

    cfg = Settings()
    out_dir = ROOT_DIR / "data" / "raw" / "runoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    years = range(cfg.year_start, cfg.year_end + 1)
    months = [f"{m:02d}" for m in range(1, 13)]
    total = len(list(years)) * len(months)
    counter = 0

    for year in range(cfg.year_start, cfg.year_end + 1):
        for mm in months:
            counter += 1
            fname = f"runoff_{year}_{mm}.nc"
            fpath = out_dir / fname

            if fpath.exists() and fpath.stat().st_size > 0:
                logger.info("[%d/%d] Exists: %s", counter, total, fname)
                continue

            logger.info("[%d/%d] Downloading %s ...", counter, total, fname)

            request = {
                "variable": ["runoff"],
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
    download_runoff()
