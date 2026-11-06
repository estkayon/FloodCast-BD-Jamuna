# src/floodcast/models/train.py

"""Training orchestrator: run all model x feature set x lead time combinations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from floodcast.features.build import load_feature_sets, select_features
from floodcast.features.engineer import engineer_features
from floodcast.features.physics import add_physics_features
from floodcast.models.evaluate import EvalResult, evaluate
from floodcast.models.lstm_model import (
    predict_lstm,
    save_lstm,
    train_lstm,
)
from floodcast.models.lstm_physics import train_lstm_physics
from floodcast.models.xgboost_model import (
    predict_xgboost,
    save_xgboost,
    train_xgboost,
    tune_xgboost,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]


def load_params() -> dict[str, Any]:
    """Load params.yaml."""
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def prepare_data(
    df: pd.DataFrame,
    feature_set: str,
    lead_time: int = 1,
    holdout_years: list[int] | None = None,
) -> dict[str, Any]:
    """Engineer features, select set, shift label by lead_time, split train/test.

    Features from day T predict flood_label on day T+lead_time.
    This prevents data leakage since the label is no longer from the same day.
    """
    holdout_years = holdout_years or [2023, 2024, 2025]

    df = engineer_features(df)
    df = add_physics_features(df)
    df_selected = select_features(df, feature_set)

    # Shift flood_label into the future by lead_time days
    df_selected = df_selected.copy()
    df_selected["flood_label"] = df_selected["flood_label"].shift(-lead_time)
    n_before = len(df_selected)
    df_selected = df_selected.dropna(subset=["flood_label"])
    df_selected["flood_label"] = df_selected["flood_label"].astype(int)
    n_dropped = n_before - len(df_selected)
    logger.info("Lead %dd: dropped %d rows from label shift", lead_time, n_dropped)

    # Split by year
    df_selected["year"] = pd.to_datetime(df_selected["date"]).dt.year
    train_mask = ~df_selected["year"].isin(holdout_years)
    test_mask = df_selected["year"].isin(holdout_years)

    feature_cols = [c for c in df_selected.columns if c not in ("date", "flood_label", "year")]

    X_train = df_selected.loc[train_mask, feature_cols].values.astype(np.float64)
    y_train = df_selected.loc[train_mask, "flood_label"].values.astype(np.int64)
    X_test = df_selected.loc[test_mask, feature_cols].values.astype(np.float64)
    y_test = df_selected.loc[test_mask, "flood_label"].values.astype(np.int64)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Physics auxiliary arrays (runoff-based)
    runoff_sum_col = "runoff_sum_7d"
    runoff_pct_col = "runoff_pct_90"

    if runoff_sum_col in df.columns:
        runoff_train = df.loc[df_selected.index[train_mask], runoff_sum_col].values.astype(np.float64)
        runoff_test = df.loc[df_selected.index[test_mask], runoff_sum_col].values.astype(np.float64)
    else:
        runoff_train = np.zeros(len(y_train))
        runoff_test = np.zeros(len(y_test))

    if runoff_pct_col in df.columns:
        disc_train = df.loc[df_selected.index[train_mask], runoff_pct_col].values.astype(np.float64)
        disc_test = df.loc[df_selected.index[test_mask], runoff_pct_col].values.astype(np.float64)
    else:
        disc_train = None
        disc_test = None

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
        "rainfall_7d_train": runoff_train,
        "rainfall_7d_test": runoff_test,
        "discharge_anomaly_train": disc_train,
        "discharge_anomaly_test": disc_test,
    }


def run_experiment(
    model_type: str,
    feature_set: str,
    data: dict[str, Any],
    params: dict[str, Any],
    lead_time: int = 1,
    output_dir: Path | None = None,
) -> EvalResult:
    """Train one model on one feature set at one lead time and evaluate."""
    output_dir = output_dir or (ROOT / "models")
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    run_name = f"{model_type}_{feature_set}_lead{lead_time}d"

    if model_type == "xgboost":
        xgb_params = params.get("xgboost", {})
        n_trials = xgb_params.get("optuna_trials", 30)
        best_params = tune_xgboost(X_train, y_train, n_trials=n_trials)
        model = train_xgboost(X_train, y_train, best_params)
        y_prob = predict_xgboost(model, X_test)
        save_xgboost(model, output_dir / f"{run_name}.json")

    elif model_type == "lstm":
        lstm_cfg = params.get("lstm", {})
        seq_len = lstm_cfg.get("sequence_length", 14)
        model, losses = train_lstm(
            X_train, y_train,
            hidden_size=lstm_cfg.get("hidden_size", 64),
            num_layers=lstm_cfg.get("num_layers", 2),
            dropout=lstm_cfg.get("dropout", 0.3),
            lr=lstm_cfg.get("learning_rate", 1e-3),
            epochs=lstm_cfg.get("epochs", 50),
            batch_size=lstm_cfg.get("batch_size", 32),
            seq_len=seq_len,
            patience=lstm_cfg.get("patience", 10),
        )
        y_prob = predict_lstm(model, X_test, seq_len=seq_len)
        y_test = y_test[seq_len - 1:]
        save_lstm(model, output_dir / f"{run_name}.pt")

    elif model_type == "lstm_physics":
        lstm_cfg = params.get("lstm", {})
        physics_cfg = params.get("physics", {})
        seq_len = lstm_cfg.get("sequence_length", 14)

        model, losses = train_lstm_physics(
            X_train, y_train,
            rainfall_7d=data["rainfall_7d_train"],
            discharge_anomaly=data["discharge_anomaly_train"],
            hidden_size=lstm_cfg.get("hidden_size", 64),
            num_layers=lstm_cfg.get("num_layers", 2),
            dropout=lstm_cfg.get("dropout", 0.3),
            lr=lstm_cfg.get("learning_rate", 1e-3),
            epochs=lstm_cfg.get("epochs", 50),
            batch_size=lstm_cfg.get("batch_size", 32),
            seq_len=seq_len,
            patience=lstm_cfg.get("patience", 10),
            lambda_mono=physics_cfg.get("lambda_mono", 0.1),
            lambda_corr=physics_cfg.get("lambda_corr", 0.1),
        )
        y_prob = predict_lstm(model, X_test, seq_len=seq_len)
        y_test = y_test[seq_len - 1:]
        save_lstm(model, output_dir / f"{run_name}.pt")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    result = evaluate(y_test, y_prob)
    logger.info("[%s] %s", run_name, result.to_dict())
    return result


def run_all_experiments(
    data_path: Path | None = None,
    output_dir: Path | None = None,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, EvalResult]:
    """Run 3 models x 3 feature sets x 3 lead times = 27 experiments."""
    data_path = data_path or (ROOT / "data" / "processed" / "features_raw.parquet")
    output_dir = output_dir or (ROOT / "models")

    params = load_params()
    load_feature_sets()

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("floodcast_bd")

    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    model_types = ["xgboost", "lstm", "lstm_physics"]
    feature_sets = ["set_a", "set_b", "set_c"]
    holdout_years = params.get("evaluation", {}).get("holdout_years", [2023, 2024, 2025])
    lead_times = params.get("evaluation", {}).get("lead_times", [1, 2, 3])

    results: dict[str, EvalResult] = {}

    for lead_time in lead_times:
        logger.info("===== Lead time: %d day(s) =====", lead_time)
        for fs in feature_sets:
            data = prepare_data(df, fs, lead_time=lead_time, holdout_years=holdout_years)

            for mt in model_types:
                run_name = f"{mt}_{fs}_lead{lead_time}d"
                logger.info("=== Running %s ===", run_name)

                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("model_type", mt)
                    mlflow.log_param("feature_set", fs)
                    mlflow.log_param("lead_time_days", lead_time)
                    mlflow.log_param("n_train", len(data["y_train"]))
                    mlflow.log_param("n_test", len(data["y_test"]))
                    mlflow.log_param("n_features", len(data["feature_names"]))

                    result = run_experiment(mt, fs, data, params, lead_time, output_dir)

                    mlflow.log_metrics(result.to_dict())
                    results[run_name] = result

    logger.info("All %d experiments complete.", len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_experiments()