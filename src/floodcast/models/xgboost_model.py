# src/floodcast/models/xgboost_model.py

"""XGBoost flood classifier with Optuna hyperparameter tuning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 1.0,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
}


def _objective(
    trial: optuna.Trial,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_splits: int = 5,
) -> float:
    """Optuna objective: maximize AUC-ROC with time-series CV."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Skip folds with single class
        if len(np.unique(y_val)) < 2:
            continue

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, y_prob))

    return float(np.mean(scores)) if scores else 0.0


def tune_xgboost(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_trials: int = 30,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run Optuna tuning and return best params."""
    study = optuna.create_study(direction="maximize", study_name="xgboost_flood")
    study.optimize(
        lambda trial: _objective(trial, X, y, n_splits),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = {**DEFAULT_PARAMS, **study.best_params}
    logger.info("Best XGBoost AUC=%.4f params=%s", study.best_value, study.best_params)
    return best


def train_xgboost(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    """Train XGBClassifier with given or default params."""
    params = params or DEFAULT_PARAMS.copy()
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    logger.info("XGBoost trained on %d samples", len(y_train))
    return model


def predict_xgboost(
    model: XGBClassifier,
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return flood probabilities."""
    return model.predict_proba(X)[:, 1].astype(np.float64)


def save_xgboost(model: XGBClassifier, path: Path) -> None:
    """Save model to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("XGBoost saved -> %s", path)


def load_xgboost(path: Path) -> XGBClassifier:
    """Load model from JSON."""
    model = XGBClassifier()
    model.load_model(str(path))
    return model
