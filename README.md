# FloodCast BD — Jamuna River Basin Flood Nowcasting

Physics-informed flood nowcasting pipeline for Bangladesh's Jamuna River Basin (Sirajganj region).

## Overview

FloodCast BD combines satellite-derived hydrometeorological data with physics-constrained deep learning to predict flood events 1-3 days ahead.

## Quick Start

```bash
conda create -n floodcast python=3.11 -y
conda activate floodcast
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cp .env.example .env
pytest tests/ -v
```

## License

MIT