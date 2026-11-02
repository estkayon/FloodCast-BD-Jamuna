"""Data quality validation checks."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from floodcast.config import settings

logger = logging.getLogger(__name__)

# Physical value ranges for Bangladesh region
VALID_RANGES: dict[str, tuple[float, float]] = {
    "daily_rainfall": (0, 500),
    "temperature_2m": (250, 330),
    "soil_moisture_layer1": (0, 1),
    "soil_moisture_layer2": (0, 1),
    "discharge": (0, 200000),
    "wind_u_10m": (-30, 30),
    "wind_v_10m": (-30, 30),
}


@dataclass
class ValidationReport:
    """Results of data validation checks."""

    total_rows: int = 0
    date_range: str = ""
    missing_counts: dict[str, int] = field(default_factory=dict)
    missing_pct: dict[str, float] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    duplicate_dates: int = 0
    passed: bool = True
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Rows: {self.total_rows}",
            f"Date range: {self.date_range}",
            f"Duplicate dates: {self.duplicate_dates}",
            "Missing values:",
        ]
        for col, count in self.missing_counts.items():
            lines.append(f"  {col}: {count} ({self.missing_pct[col]:.1f}%)")
        lines.append("Out-of-range values:")
        for col, count in self.out_of_range.items():
            lines.append(f"  {col}: {count}")
        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  - {e}")
        lines.append(f"Overall: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Run all validation checks on a processed DataFrame."""
    report = ValidationReport()
    report.total_rows = len(df)

    # Date range
    if "date" in df.columns:
        report.date_range = f"{df['date'].min()} to {df['date'].max()}"
    else:
        report.errors.append("No 'date' column found")
        report.passed = False
        return report

    # Duplicate dates
    report.duplicate_dates = df["date"].duplicated().sum()
    if report.duplicate_dates > 0:
        report.errors.append(f"{report.duplicate_dates} duplicate dates found")

    # Missing values
    for col in df.columns:
        if col == "date":
            continue
        n_missing = df[col].isna().sum()
        report.missing_counts[col] = n_missing
        report.missing_pct[col] = 100 * n_missing / len(df) if len(df) > 0 else 0
        if report.missing_pct[col] > 10:
            report.errors.append(f"{col}: {report.missing_pct[col]:.1f}% missing (>10%)")
            report.passed = False

    # Range checks
    for col, (vmin, vmax) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        out = ((df[col] < vmin) | (df[col] > vmax)).sum()
        report.out_of_range[col] = out
        if out > 0:
            report.errors.append(f"{col}: {out} values outside [{vmin}, {vmax}]")

    return report


def validate_processed_file(filepath: Path | None = None) -> ValidationReport:
    """Validate the processed parquet file."""
    filepath = filepath or (settings.ROOT_DIR / "data" / "processed" / "features_raw.parquet")

    if not filepath.exists():
        report = ValidationReport()
        report.passed = False
        report.errors.append(f"File not found: {filepath}")
        return report

    df = pd.read_parquet(filepath)
    report = validate_dataframe(df)
    logger.info("Validation report:\n%s", report.summary())
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = validate_processed_file()
    print(r.summary())
