# src/floodcast/features/build.py
"""Feature pipeline: orchestrate engineering + physics + feature set selection."""

import logging
from pathlib import Path

import pandas as pd
import yaml

from floodcast.features.engineer import engineer_features
from floodcast.features.physics import add_physics_features

logger = logging.getLogger(__name__)

FEATURE_SETS: dict[str, list[str]] = {}

ROOT = Path(__file__).resolve().parents[3]


def load_feature_sets(params_path: Path | None = None) -> dict[str, list[str]]:
    """Load feature set definitions from params.yaml."""
    global FEATURE_SETS
    params_path = params_path or (ROOT / "params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    FEATURE_SETS = {
        name: cfg["features"]
        for name, cfg in params["feature_sets"].items()
    }
    return FEATURE_SETS


def select_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Select columns for a given feature set (set_a, set_b, set_c)."""
    if not FEATURE_SETS:
        load_feature_sets()

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}. Choose from {list(FEATURE_SETS)}")

    base_features = FEATURE_SETS[feature_set]

    selected = ["date"]
    for col in df.columns:
        if col in ("date", "flood_label"):
            continue
        for base in base_features:
            if col == base or col.startswith(base):
                selected.append(col)
                break
        # Always include calendar and physics features for all sets
        if col in ("month", "day_of_year", "is_monsoon", "antecedent_runoff_index"):
            if col not in selected:
                selected.append(col)
        # Include runoff-derived features only for sets with runoff
        if "runoff" in base_features:
            if col in ("runoff_pct_90", "soil_moisture_anomaly"):
                if col not in selected:
                    selected.append(col)

    selected.append("flood_label")

    seen = set()
    unique = []
    for c in selected:
        if c not in seen and c in df.columns:
            seen.add(c)
            unique.append(c)

    return df[unique].copy()


def build_features(
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Run full feature pipeline: engineer + physics + save."""
    input_path = input_path or (ROOT / "data" / "processed" / "features_raw.parquet")
    output_dir = output_dir or (ROOT / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)

    df = engineer_features(df)
    df = add_physics_features(df)

    output_path = output_dir / "features_engineered.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Saved %d rows, %d columns -> %s", len(df), len(df.columns), output_path)

    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_features()