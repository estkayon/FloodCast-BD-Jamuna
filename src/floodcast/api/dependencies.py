# src/floodcast/api/dependencies.py

"""Shared dependencies: model loading, rate limiting."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from floodcast import __version__

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]


class ModelStore:
    """Holds the loaded model and metadata."""

    def __init__(self) -> None:
        self.model: Any = None
        self.model_type: str = "xgboost"
        self.feature_set: str = "set_c"
        self.model_version: str = __version__
        self.training_date: str = "N/A"
        self.n_training_samples: int = 0
        self.sequence_length: int | None = None
        self.scaler: Any = None
        self.metrics: dict[str, float] = {}
        self._loaded: bool = False

    def load(self, model_dir: Path | None = None) -> None:
        """Load the best model from disk."""
        model_dir = model_dir or (ROOT / "models")

        # Try loading XGBoost first (most common best model)
        xgb_path = model_dir / "xgboost_set_c.json"
        if xgb_path.exists():
            from floodcast.models.xgboost_model import load_xgboost

            self.model = load_xgboost(xgb_path)
            self.model_type = "xgboost"
            self._loaded = True
            logger.info("Loaded XGBoost from %s", xgb_path)
            return

        # Try LSTM physics
        lstm_path = model_dir / "lstm_physics_set_c.pt"
        if lstm_path.exists():
            from floodcast.models.lstm_model import load_lstm

            self.model = load_lstm(lstm_path, input_size=1)
            self.model_type = "lstm_physics"
            self.sequence_length = 14
            self._loaded = True
            logger.info("Loaded LSTM physics from %s", lstm_path)
            return

        logger.warning("No trained model found in %s. API will return mock predictions.", model_dir)
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run prediction. Returns probabilities."""
        if not self._loaded:
            # Return mock predictions for testing
            rng = np.random.RandomState(42)
            return rng.uniform(0.1, 0.9, size=len(features)).astype(np.float64)

        if self.model_type == "xgboost":
            from floodcast.models.xgboost_model import predict_xgboost

            return predict_xgboost(self.model, features)
        else:
            from floodcast.models.lstm_model import predict_lstm

            return predict_lstm(self.model, features, seq_len=self.sequence_length or 14)


# Simple in-memory rate limiter
class RateLimiter:
    """Token bucket rate limiter per IP."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        self.rpm = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window = now - 60.0

        # Clean old entries
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window
        ]

        if len(self.requests[client_ip]) >= self.rpm:
            return False

        self.requests[client_ip].append(now)
        return True


# Singletons
model_store = ModelStore()
rate_limiter = RateLimiter()
