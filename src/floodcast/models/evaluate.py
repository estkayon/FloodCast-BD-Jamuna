# src/floodcast/models/evaluate.py

"""Evaluation metrics for flood prediction models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation metrics."""

    auc_roc: float
    f1: float
    precision: float
    recall: float
    brier: float
    confusion: NDArray[np.int64]
    fpr: NDArray[np.float64] = field(repr=False)
    tpr: NDArray[np.float64] = field(repr=False)
    pr_curve_precision: NDArray[np.float64] = field(repr=False)
    pr_curve_recall: NDArray[np.float64] = field(repr=False)
    auc_pr: float = 0.0
    threshold: float = 0.5

    def to_dict(self) -> dict:
        """Return metrics as a flat dict (for MLflow logging)."""
        return {
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "brier": self.brier,
            "threshold": self.threshold,
        }


def evaluate(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    threshold: float = 0.5,
) -> EvalResult:
    """Compute all evaluation metrics.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_prob: Predicted probabilities for the positive (flood) class.
        threshold: Classification threshold.

    Returns:
        EvalResult with all metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(pr_rec, pr_prec)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    result = EvalResult(
        auc_roc=auc_roc,
        f1=f1,
        precision=prec,
        recall=rec,
        brier=brier,
        confusion=cm,
        fpr=fpr,
        tpr=tpr,
        pr_curve_precision=pr_prec,
        pr_curve_recall=pr_rec,
        auc_pr=auc_pr,
        threshold=threshold,
    )

    logger.info(
        "AUC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f  Brier=%.4f",
        auc_roc, f1, prec, rec, brier,
    )
    return result
