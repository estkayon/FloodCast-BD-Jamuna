<div align="center">

<img src="assets/banner.svg" alt="FloodCast BD Banner" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=3B82F6&center=true&vCenter=true&multiline=true&repeat=true&width=600&height=50&lines=AI-Powered+Flood+Prediction+%F0%9F%8C%8A;Physics-Constrained+Deep+Learning+%F0%9F%A7%AA;Real-Time+Risk+Assessment+%E2%9A%A1" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

<br/>

[![Tests](https://img.shields.io/badge/Tests-72%20Passed-brightgreen?style=flat-square&logo=pytest)](tests/)
[![Linting](https://img.shields.io/badge/Linting-Ruff-orange?style=flat-square&logo=ruff)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/estkayon/FloodCast-BD-Jamuna/ci.yml?style=flat-square&label=CI&logo=githubactions)](https://github.com/estkayon/FloodCast-BD-Jamuna/actions)

</div>

---

## 🎯 What is FloodCast BD?

FloodCast BD is a **physics-informed machine learning pipeline** that predicts flood events for Bangladesh's Jamuna River Basin (Sirajganj region) **1-3 days ahead**. It combines satellite data with domain knowledge to enforce physically consistent predictions.

<details>
<summary><b>🔬 Why Physics Constraints?</b></summary>
<br/>

Standard ML models can learn spurious correlations. FloodCast enforces two physical laws in the LSTM loss function:

| Constraint | What it enforces |
|:---|:---|
| **Monotonicity Penalty** | Higher cumulative rainfall → higher flood probability |
| **Discharge Correlation** | Predictions must correlate with upstream river discharge |

This reduces false negatives during critical flood events where pure data-driven models might fail.

</details>

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph 🛰️ Data Sources
        ERA5[ERA5-Land<br/>Rainfall · Temperature · Soil Moisture]
        CHIRPS[CHIRPS<br/>Daily Precipitation]
        GloFAS[GloFAS<br/>River Discharge]
    end

    subgraph ⚙️ Pipeline
        DL[Download & Cache]
        PROC[Process & Validate]
        FEAT[Feature Engineering<br/>+ Physics Features]
    end

    subgraph 🧠 Models
        XGB[XGBoost<br/>Baseline]
        LSTM[LSTM<br/>Baseline]
        LSTMP[LSTM + Physics<br/>Constrained]
    end

    subgraph 🚀 Serving
        API[FastAPI<br/>REST API]
        DASH[Streamlit<br/>Dashboard]
    end

    ERA5 --> DL
    CHIRPS --> DL
    GloFAS --> DL
    DL --> PROC
    PROC --> FEAT
    FEAT --> XGB & LSTM & LSTMP
    XGB & LSTM & LSTMP --> API
    API --> DASH

    style ERA5 fill:#1e3a5f,stroke:#3b82f6,color:#f1f5f9
    style CHIRPS fill:#1e3a5f,stroke:#3b82f6,color:#f1f5f9
    style GloFAS fill:#1e3a5f,stroke:#3b82f6,color:#f1f5f9
    style LSTMP fill:#1e293b,stroke:#22c55e,color:#f1f5f9
    style API fill:#1e293b,stroke:#f59e0b,color:#f1f5f9
    style DASH fill:#1e293b,stroke:#f59e0b,color:#f1f5f9
```

---

## 📊 Experiment Matrix

<div align="center">

|  | **Set A**<br/>(Rainfall Only) | **Set B**<br/>(Rainfall + Discharge) | **Set C**<br/>(Full Features) |
|:---:|:---:|:---:|:---:|
| **XGBoost** | ✅ | ✅ | ✅ |
| **LSTM** | ✅ | ✅ | ✅ |
| **LSTM + Physics** | ✅ | ✅ | ✅ |

*9 experiments total, all tracked in MLflow with time-series cross-validation*

</div>

---

## 🚀 Quick Start

<details>
<summary><b>📋 Prerequisites</b></summary>

- Python 3.11
- Conda (recommended)
- Docker (optional)
- CDS API account for ERA5/GloFAS data → [Register (free)](https://cds.climate.copernicus.eu/)

</details>

### 1️⃣ Setup

```bash
git clone https://github.com/estkayon/FloodCast-BD-Jamuna.git
cd FloodCast-BD-Jamuna

conda create -n floodcast python=3.11 -y
conda activate floodcast
pip install -r requirements.txt
```

> **Windows PowerShell** — set in each new terminal:
> ```powershell
> $env:PYTHONPATH = "D:\Study\Projects\FloodCast-BD-Jamuna\src"
> ```

### 2️⃣ Configure CDS API

Create `~/.cdsapirc`:

url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>


### 3️⃣ Download & Process Data

```bash
python -m floodcast.data.download_era5
python -m floodcast.data.download_chirps
python -m floodcast.data.download_glofas
python -m floodcast.data.process
```

### 4️⃣ Build Features & Train

```bash
python -m floodcast.features.build
python -m floodcast.models.train      # Trains all 9 experiments
mlflow ui                             # View results at localhost:5000
```

### 5️⃣ Run Services

```bash
# API
uvicorn floodcast.api.app:app --reload    # localhost:8000

# Dashboard
streamlit run src/floodcast/dashboard/app.py   # localhost:8501
```

### 🐳 Or use Docker

```bash
docker-compose up
```

| Service | URL |
|:---|:---|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |

---

## 🗺️ Study Area

<div align="center">

| Parameter | Value |
|:---|:---|
| **River** | Jamuna (Brahmaputra) |
| **Station** | Sirajganj |
| **Bounding Box** | 24.0°N - 25.5°N, 89.0°E - 90.5°E |
| **Period** | 2000 - 2025 |
| **Target** | Binary flood (discharge > danger level) |

</div>

---

## 🧪 Testing

```bash
pytest tests/ -v          # 72 tests
ruff check src/ tests/    # Linting
```

<details>
<summary><b>📋 Test Coverage Breakdown</b></summary>

| Module | Tests |
|:---|:---:|
| Data Pipeline | 16 |
| Features | 10 |
| Models | 11 |
| API | 12 |
| Dashboard | 10 |
| Docker & Deployment | 13 |
| **Total** | **72** |

</details>

---

## 📁 Project Structure

<details>
<summary><b>Click to expand</b></summary>

FloodCast-BD-Jamuna/
├── 📂 src/floodcast/
│ ├── config.py # Pydantic settings
│ ├── 📂 data/
│ │ ├── download_era5.py # ERA5-Land via CDS API
│ │ ├── download_chirps.py # CHIRPS via HTTP
│ │ ├── download_glofas.py # GloFAS via CDS API
│ │ ├── process.py # Merge & clean
│ │ └── validate.py # Schema validation
│ ├── 📂 features/
│ │ ├── engineer.py # Feature transforms
│ │ ├── physics.py # Physics-based features
│ │ └── build.py # Feature pipeline
│ ├── 📂 models/
│ │ ├── evaluate.py # Metrics & evaluation
│ │ ├── xgboost_model.py # XGBoost
│ │ ├── lstm_model.py # LSTM
│ │ ├── lstm_physics.py # LSTM + Physics constraints
│ │ └── train.py # Training orchestrator
│ ├── 📂 api/
│ │ ├── app.py # FastAPI app
│ │ ├── routes.py # Endpoints
│ │ ├── schemas.py # Pydantic models
│ │ └── dependencies.py # DI
│ └── 📂 dashboard/
│ ├── app.py # Streamlit main
│ ├── api_client.py # HTTP client
│ └── 📂 views/
│ ├── current_risk.py # Live risk assessment
│ ├── historical.py # Historical analysis
│ ├── model_comparison.py # Model metrics
│ └── about.py # Project info
├── 📂 tests/ # 72 tests
├── 📂 docker/
│ ├── Dockerfile.api
│ └── Dockerfile.dashboard
├── docker-compose.yml
├── 📂 .github/workflows/
│ ├── ci.yml # Lint + Test
│ └── cd.yml # Docker → GHCR
├── dvc.yaml
├── render.yaml
├── requirements.txt
├── pyproject.toml
├── REPRODUCIBILITY.md
└── RESULTS.md


</details>

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technologies |
|:---|:---|
| **ML / DL** | PyTorch, XGBoost, Optuna |
| **Data** | pandas, xarray, netCDF4, DVC |
| **API** | FastAPI, uvicorn, pydantic |
| **Dashboard** | Streamlit |
| **MLOps** | MLflow, Docker, GitHub Actions |
| **Quality** | ruff, mypy, pytest |

</div>

---

## ⚠️ Limitations

- Research prototype — **not** an operational flood warning system
- Predictions are for Jamuna at Sirajganj only; other basins require retraining
- No real-time data ingestion; uses historical satellite data
- CPU-only training; large experiments benefit from GPU
- Model accuracy depends on ERA5/CHIRPS/GloFAS data quality

---

## 📄 Documentation

| Document | Description |
|:---|:---|
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Step-by-step experiment reproduction |
| [RESULTS.md](RESULTS.md) | Model comparison results and tables |

---

## 👤 Author

<div align="center">

**Md. Estiak Rahman Ayon**

[![GitHub](https://img.shields.io/badge/GitHub-estkayon-181717?style=for-the-badge&logo=github)](https://github.com/estkayon)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/md-estiak-rahman-ayon)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF5722?style=for-the-badge&logo=firefox)](https://estkayon.github.io/Portfolio/)

</div>

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=3b82f6&height=80&section=footer" width="100%"/>

</div>

url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>


---

## Step 3 → Download Raw Data

```bash
python -m floodcast.data.download_era5
python -m floodcast.data.download_chirps
python -m floodcast.data.download_glofas
```

> ⏱️ CDS queue times vary. Downloads may take 1-2 hours.

---

## Step 4 → Process & Validate

```bash
python -m floodcast.data.process
```

**Output:** `data/processed/features_raw.parquet`

---

## Step 5 → Build Features

```bash
python -m floodcast.features.build
```

**Output:** Feature matrices for sets A, B, C in `data/processed/`

---

## Step 6 → Train All Models

```bash
python -m floodcast.models.train
```

Trains 9 experiments (3 models × 3 feature sets) with time-series cross-validation. All metrics logged to MLflow.

```bash
mlflow ui    # View at http://localhost:5000
```

---

## Step 7 → Verify

```bash
pytest tests/ -v        # 72 tests should pass
ruff check src/ tests/  # Zero lint errors
```

---

## Step 8 → Run Services

```bash
# Terminal 1 — API
uvicorn floodcast.api.app:app --reload

# Terminal 2 — Dashboard
streamlit run src/floodcast/dashboard/app.py
```

Or with Docker:

```bash
docker-compose up
```

---

## 📁 Expected Outputs

| Stage | Output Path | Description |
|:---|:---|:---|
| Download | `data/raw/era5/` | ERA5-Land NetCDF files |
| Download | `data/raw/chirps/` | CHIRPS daily rainfall |
| Download | `data/raw/glofas/` | GloFAS discharge data |
| Process | `data/processed/features_raw.parquet` | Merged clean dataset |
| Features | `data/processed/features_*.parquet` | Feature matrices A/B/C |
| Train | `models/` | Trained model artifacts |
| Train | `mlruns/` | MLflow experiment logs |

---

## 🔧 Using DVC

If DVC remote is configured:

```bash
dvc pull     # Download cached data/models
dvc repro    # Reproduce entire pipeline
```

---

## ❓ Troubleshooting

| Problem | Solution |
|:---|:---|
| CDS API timeout | CDS queues can be slow. Retry after a few hours |
| `ModuleNotFoundError: floodcast` | Set PYTHONPATH (see Step 1) |
| `torch not found` | Install from requirements.txt (pins CPU-only torch) |
| Docker build fails for torch | Ensure `--extra-index-url https://download.pytorch.org/whl/cpu` in Dockerfile |

---

## 🔢 Environment Versions

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