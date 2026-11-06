# src/floodcast/models/export_results.py

"""Export MLflow experiment results to CSV for dashboard use."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]


def export_results(output_path: Path | None = None) -> pd.DataFrame:
    """Read MLflow runs and save as CSV."""
    output_path = output_path or (ROOT / "results" / "experiment_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mlruns_path = ROOT / "mlruns"
    # Windows needs file:/// URI for MLflow tracking
    uri = mlruns_path.resolve().as_posix()
    if not uri.startswith("/"):
        uri = "/" + uri
    mlflow.set_tracking_uri("file://" + uri)

    experiment = mlflow.get_experiment_by_name("floodcast_bd")
    if experiment is None:
        raise RuntimeError("No MLflow experiment 'floodcast_bd' found. Run training first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
    )

    if runs.empty:
        raise RuntimeError("No MLflow runs found. Run training first.")

    rows = []
    for _, run in runs.iterrows():
        # Skip old runs that had no lead_time_days (data leakage)
        if pd.isna(run.get("params.lead_time_days")):
            continue

        model_type = run.get("params.model_type", "")
        feature_set = run.get("params.feature_set", "")
        lead_time = int(run.get("params.lead_time_days") or 1)
        auc = run.get("metrics.auc_roc", 0)
        f1 = run.get("metrics.f1", 0)
        auc_pr = run.get("metrics.auc_pr", 0)
        precision = run.get("metrics.precision", 0)
        recall = run.get("metrics.recall", 0)

        rows.append({
            "model": model_type,
            "feature_set": feature_set,
            "lead_time": lead_time,
            "auc": auc,
            "f1": f1,
            "auc_pr": auc_pr,
            "precision": precision,
            "recall": recall,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["lead_time", "model", "feature_set"])
    df.to_csv(output_path, index=False)
    logger.info("Exported %d results to %s", len(df), output_path)
    print(f"Exported {len(df)} results to {output_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_results()