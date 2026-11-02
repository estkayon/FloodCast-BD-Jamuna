# src/floodcast/models/lstm_physics.py

"""LSTM with physics-constrained loss (monotonicity + discharge correlation)."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from floodcast.models.lstm_model import FloodLSTM, create_sequences

logger = logging.getLogger(__name__)


def monotonicity_penalty(
    probs: torch.Tensor,
    rainfall_7d: torch.Tensor,
) -> torch.Tensor:
    """Penalize when 7-day cumulative rainfall increases but flood prob decreases.

    Args:
        probs: Predicted flood probabilities (batch,).
        rainfall_7d: 7-day cumulative rainfall at last timestep (batch,).
    """
    if len(probs) < 2:
        return torch.tensor(0.0)

    rain_diff = rainfall_7d[1:] - rainfall_7d[:-1]
    prob_diff = probs[1:] - probs[:-1]

    # Violation: rain increases but prob decreases
    violations = torch.relu(-prob_diff) * torch.relu(rain_diff)
    return violations.mean()


def discharge_correlation_penalty(
    probs: torch.Tensor,
    discharge_anomaly: torch.Tensor,
) -> torch.Tensor:
    """Penalize negative correlation between predicted prob and discharge anomaly.

    We want positive correlation: high discharge anomaly -> high flood prob.
    Penalty = max(0, -correlation).
    """
    if len(probs) < 2:
        return torch.tensor(0.0)

    p_centered = probs - probs.mean()
    d_centered = discharge_anomaly - discharge_anomaly.mean()

    cov = (p_centered * d_centered).mean()
    std_p = probs.std() + 1e-8
    std_d = discharge_anomaly.std() + 1e-8
    corr = cov / (std_p * std_d)

    return torch.relu(-corr)


def create_physics_sequences(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    rainfall_7d: NDArray[np.float64],
    discharge_anomaly: NDArray[np.float64] | None,
    seq_len: int = 14,
) -> tuple[NDArray, NDArray, NDArray, NDArray | None]:
    """Create sequences with aligned physics auxiliary arrays."""
    X_seq, y_seq = create_sequences(X, y, seq_len)

    # Align auxiliary arrays to sequence labels (last element of each window)
    start = seq_len - 1
    rain_seq = rainfall_7d[start : start + len(y_seq)]

    disc_seq = None
    if discharge_anomaly is not None:
        disc_seq = discharge_anomaly[start : start + len(y_seq)]

    return X_seq, y_seq, rain_seq, disc_seq


def train_lstm_physics(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    rainfall_7d: NDArray[np.float64],
    discharge_anomaly: NDArray[np.float64] | None = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    seq_len: int = 14,
    patience: int = 10,
    lambda_mono: float = 0.1,
    lambda_corr: float = 0.1,
) -> tuple[FloodLSTM, list[float]]:
    """Train LSTM with physics-constrained loss.

    Total loss = BCE + lambda_mono * monotonicity + lambda_corr * discharge_corr

    Returns:
        Trained model and list of epoch losses.
    """
    X_seq, y_seq, rain_seq, disc_seq = create_physics_sequences(
        X_train, y_train, rainfall_7d, discharge_anomaly, seq_len
    )

    X_t = torch.tensor(X_seq, dtype=torch.float32)
    y_t = torch.tensor(y_seq, dtype=torch.float32)
    rain_t = torch.tensor(rain_seq, dtype=torch.float32)
    disc_t = torch.tensor(disc_seq, dtype=torch.float32) if disc_seq is not None else None

    # Build dataset with auxiliary tensors
    if disc_t is not None:
        dataset = TensorDataset(X_t, y_t, rain_t, disc_t)
    else:
        dataset = TensorDataset(X_t, y_t, rain_t)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = FloodLSTM(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    losses: list[float] = []
    best_loss = float("inf")
    wait = 0
    has_discharge = disc_t is not None

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            if has_discharge:
                X_b, y_b, rain_b, disc_b = batch
            else:
                X_b, y_b, rain_b = batch
                disc_b = None

            optimizer.zero_grad()
            logits = model(X_b)
            probs = torch.sigmoid(logits)

            loss_bce = bce(logits, y_b)
            loss_mono = monotonicity_penalty(probs, rain_b)

            total = loss_bce + lambda_mono * loss_mono

            if disc_b is not None and lambda_corr > 0:
                loss_corr = discharge_correlation_penalty(probs, disc_b)
                total = total + lambda_corr * loss_corr

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += total.item() * len(y_b)

        epoch_loss /= len(y_seq)
        losses.append(epoch_loss)

        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Physics LSTM early stop at epoch %d", epoch + 1)
                break

    logger.info(
        "Physics LSTM trained %d epochs, final loss=%.4f (lambda_mono=%.3f, lambda_corr=%.3f)",
        len(losses), losses[-1], lambda_mono, lambda_corr,
    )
    return model, losses
