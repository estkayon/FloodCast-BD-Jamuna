<div align="center">

# 🔄 Reproducibility Guide

*Step-by-step instructions to reproduce all FloodCast BD experiments from scratch*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?style=flat-square&logo=dvc&logoColor=white)](https://dvc.org)

</div>

---

## 📋 Prerequisites

| Requirement | Details |
|:---|:---|
| Python | 3.11 |
| Conda | Latest |
| Git | Latest |
| Disk Space | ~2 GB for raw data |
| CDS API Account | Free → [Register here](https://cds.climate.copernicus.eu/) |

---

## Step 1 → Clone & Install

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

## Step 2 → Configure CDS API

Register at https://cds.climate.copernicus.eu/ and create the config file:

**Linux/Mac** → `~/.cdsapirc`
**Windows** → `%USERPROFILE%\.cdsapirc`