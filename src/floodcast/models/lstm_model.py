# src/floodcast/models/lstm_model.py

"""LSTM flood classifier with standard BCE loss (PyTorch, CPU only)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class FloodLSTM(nn.Module):
    """2-layer LSTM for binary flood classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (batch_size,)."""
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


def create_sequences(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    seq_len: int = 14,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Create sliding window sequences for LSTM.

    Returns:
        X_seq: shape (n_samples - seq_len + 1, seq_len, n_features)
        y_seq: shape (n_samples - seq_len + 1,)
    """
    sequences = []
    labels = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i : i + seq_len])
        labels.append(y[i + seq_len - 1])
    return np.array(sequences, dtype=np.float64), np.array(labels, dtype=np.int64)


def train_lstm(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    seq_len: int = 14,
    patience: int = 10,
) -> tuple[FloodLSTM, list[float]]:
    """Train LSTM with standard BCE loss.

    Returns:
        Trained model and list of epoch losses.
    """
    X_seq, y_seq = create_sequences(X_train, y_train, seq_len)

    X_t = torch.tensor(X_seq, dtype=torch.float32)
    y_t = torch.tensor(y_seq, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = FloodLSTM(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    losses: list[float] = []
    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        epoch_loss /= len(y_seq)
        losses.append(epoch_loss)

        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    logger.info("LSTM trained %d epochs, final loss=%.4f", len(losses), losses[-1])
    return model, losses


def predict_lstm(
    model: FloodLSTM,
    X: NDArray[np.float64],
    seq_len: int = 14,
) -> NDArray[np.float64]:
    """Return flood probabilities."""
    X_seq, _ = create_sequences(X, np.zeros(len(X), dtype=np.int64), seq_len)
    X_t = torch.tensor(X_seq, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).numpy()

    return probs.astype(np.float64)


def save_lstm(model: FloodLSTM, path: Path) -> None:
    """Save model state dict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("LSTM saved -> %s", path)


def load_lstm(path: Path, input_size: int, hidden_size: int = 64) -> FloodLSTM:
    """Load model from state dict."""
    model = FloodLSTM(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
