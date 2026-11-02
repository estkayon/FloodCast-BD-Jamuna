# tests/test_models.py

"""Tests for model training and evaluation on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
from floodcast.models.evaluate import evaluate
from floodcast.models.lstm_model import FloodLSTM, create_sequences, predict_lstm, train_lstm
from floodcast.models.lstm_physics import (
    discharge_correlation_penalty,
    monotonicity_penalty,
    train_lstm_physics,
)
from floodcast.models.xgboost_model import predict_xgboost, train_xgboost


@pytest.fixture()
def synthetic_data():
    """Create synthetic flood dataset for testing."""
    rng = np.random.RandomState(42)
    n = 300
    n_features = 8

    X = rng.randn(n, n_features).astype(np.float64)
    # Make labels somewhat correlated with first feature
    prob = 1 / (1 + np.exp(-X[:, 0]))
    y = (prob > 0.5).astype(np.int64)
    # Ensure both classes present
    y[:10] = 0
    y[-10:] = 1

    split = 200
    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
        "n_features": n_features,
    }


@pytest.fixture()
def rainfall_7d(synthetic_data):
    """Fake 7-day cumulative rainfall array."""
    rng = np.random.RandomState(99)
    return np.abs(rng.randn(len(synthetic_data["X_train"]))).astype(np.float64)


class TestXGBoost:
    def test_train_and_predict(self, synthetic_data):
        model = train_xgboost(synthetic_data["X_train"], synthetic_data["y_train"])
        probs = predict_xgboost(model, synthetic_data["X_test"])
        assert probs.shape == (len(synthetic_data["y_test"]),)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_predict_probabilities_sum(self, synthetic_data):
        model = train_xgboost(synthetic_data["X_train"], synthetic_data["y_train"])
        probs = predict_xgboost(model, synthetic_data["X_test"])
        # Probabilities should be reasonable (not all 0 or all 1)
        assert 0 < probs.mean() < 1


class TestLSTM:
    def test_create_sequences(self, synthetic_data):
        seq_len = 14
        X_seq, y_seq = create_sequences(
            synthetic_data["X_train"], synthetic_data["y_train"], seq_len
        )
        expected_len = len(synthetic_data["X_train"]) - seq_len + 1
        assert X_seq.shape == (expected_len, seq_len, synthetic_data["n_features"])
        assert y_seq.shape == (expected_len,)

    def test_train_and_predict(self, synthetic_data):
        model, losses = train_lstm(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
            hidden_size=16,
            num_layers=1,
            epochs=5,
            batch_size=32,
            seq_len=14,
            patience=3,
        )
        assert isinstance(model, FloodLSTM)
        assert len(losses) > 0

        probs = predict_lstm(model, synthetic_data["X_test"], seq_len=14)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_loss_decreases(self, synthetic_data):
        _, losses = train_lstm(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
            hidden_size=16,
            num_layers=1,
            epochs=10,
            seq_len=14,
            patience=20,
        )
        # Loss should generally decrease (first > last)
        assert losses[0] >= losses[-1]


class TestPhysicsLSTM:
    def test_monotonicity_penalty_shape(self):
        import torch

        probs = torch.tensor([0.3, 0.5, 0.4, 0.6])
        rain = torch.tensor([10.0, 15.0, 20.0, 25.0])
        penalty = monotonicity_penalty(probs, rain)
        assert penalty.shape == ()
        assert penalty.item() >= 0

    def test_discharge_penalty_shape(self):
        import torch

        probs = torch.tensor([0.3, 0.5, 0.7, 0.9])
        disc = torch.tensor([1.0, 2.0, 3.0, 4.0])
        penalty = discharge_correlation_penalty(probs, disc)
        assert penalty.shape == ()
        # Positive correlation -> penalty should be 0
        assert penalty.item() == pytest.approx(0.0, abs=1e-5)

    def test_train_physics_lstm(self, synthetic_data, rainfall_7d):
        model, losses = train_lstm_physics(
            synthetic_data["X_train"],
            synthetic_data["y_train"],
            rainfall_7d=rainfall_7d,
            discharge_anomaly=None,
            hidden_size=16,
            num_layers=1,
            epochs=5,
            seq_len=14,
            patience=3,
            lambda_mono=0.1,
            lambda_corr=0.0,
        )
        assert isinstance(model, FloodLSTM)
        assert len(losses) > 0


class TestEvaluate:
    def test_evaluate_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate(y_true, y_prob)
        assert result.auc_roc == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)

    def test_evaluate_returns_dict(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.2, 0.3, 0.7, 0.8, 0.4, 0.6])
        result = evaluate(y_true, y_prob)
        d = result.to_dict()
        assert "auc_roc" in d
        assert "f1" in d
        assert "brier" in d
        assert 0 <= d["brier"] <= 1

    def test_evaluate_output_ranges(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        y_prob = rng.uniform(0, 1, size=100)
        result = evaluate(y_true, y_prob)
        assert 0 <= result.auc_roc <= 1
        assert 0 <= result.f1 <= 1
        assert result.confusion.shape == (2, 2)
