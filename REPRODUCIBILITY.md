<div align="center">

# Reproducibility Guide

*Reproduce all 27 FloodCast BD experiments from scratch*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?style=flat-square&logo=dvc&logoColor=white)](https://dvc.org)

</div>

---

## Prerequisites

| Requirement | Details |
|:---|:---|
| Python | 3.11 |
| Conda | Latest |
| Git | Latest |
| Disk Space | ~2 GB for raw data |
| CDS API Account | Free registration at [CDS](https://cds.climate.copernicus.eu/) |

---

## Step 1: Clone & Install

```bash
git clone https://github.com/estkayon/FloodCast-BD-Jamuna.git
cd FloodCast-BD-Jamuna

conda create -n floodcast python=3.11 -y
conda activate floodcast
pip install -r requirements.txt
```

Set PYTHONPATH (required in every new terminal):

```bash
# Linux/Mac
export PYTHONPATH="$(pwd)/src"

# Windows PowerShell
$env:PYTHONPATH = "$(Get-Location)\src"
```

---

## Step 2: Configure CDS API

Register at https://cds.climate.copernicus.eu/ and create the config file:

**Linux/Mac:** `~/.cdsapirc`
**Windows:** `%USERPROFILE%\.cdsapirc`

```
url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>
```

---

## Step 3: Download Raw Data

```bash
python -m floodcast.data.download_era5
python -m floodcast.data.download_chirps
python -m floodcast.data.download_glofas
```

CDS queue times vary. Downloads may take 1-2 hours.

---

## Step 4: Process & Validate

```bash
python -m floodcast.data.process
```

**Output:** `data/processed/features_raw.parquet`

---

## Step 5: Build Features

```bash
python -m floodcast.features.build
```

**Output:** Feature matrices for sets A, B, C in `data/processed/`

---

## Step 6: Train All Models

```bash
python -m floodcast.models.train
```

Trains 27 experiments (3 models x 3 feature sets x 3 lead times). Results saved to `results/experiment_results.csv` and logged to MLflow.

```bash
mlflow ui    # View at http://localhost:5000
```

---

## Step 7: Verify

```bash
pytest tests/ -v        # 72 tests should pass
ruff check src/ tests/  # Zero lint errors
```

---

## Step 8: Run Services

```bash
# Dashboard
streamlit run src/floodcast/dashboard/app.py    # localhost:8501

# API
uvicorn floodcast.api.app:app --reload          # localhost:8000
```

Or with Docker:

```bash
docker-compose up
```

---

## Expected Outputs

| Stage | Output Path | Description |
|:---|:---|:---|
| Download | `data/raw/era5/` | ERA5-Land NetCDF files |
| Download | `data/raw/chirps/` | CHIRPS daily rainfall |
| Download | `data/raw/glofas/` | GloFAS discharge data |
| Process | `data/processed/features_raw.parquet` | Merged clean dataset |
| Features | `data/processed/features_*.parquet` | Feature matrices A/B/C |
| Train | `models/` | Trained model artifacts |
| Train | `results/experiment_results.csv` | All 27 experiment metrics |
| Train | `mlruns/` | MLflow experiment logs |

---

## Using DVC

If DVC remote is configured:

```bash
dvc pull     # Download cached data/models
dvc repro    # Reproduce entire pipeline
```

---

## Environment Versions

All versions pinned in `requirements.txt`. Key packages:

| Package | Version |
|:---|:---|
| Python | 3.11 |
| PyTorch | 2.2.1+cpu |
| XGBoost | 2.0.3 |
| FastAPI | 0.110.0 |
| Streamlit | 1.31.1 |
| MLflow | 2.10.2 |
| DVC | 3.42.0 |

---

## Troubleshooting

| Problem | Solution |
|:---|:---|
| CDS API timeout | CDS queues can be slow. Retry after a few hours |
| `ModuleNotFoundError: floodcast` | Set PYTHONPATH (see Step 1) |
| `torch not found` | Install from requirements.txt (pins CPU-only torch) |
| Docker build fails for torch | Ensure `--extra-index-url` in Dockerfile |
