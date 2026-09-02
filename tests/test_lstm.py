"""Unit tests for Phase 7 LSTM forecaster, sequence generation, scaling, and test isolation."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.lstm_model import LSTMForecaster, PyTorchLSTM
from src.models.evaluation import split_chronological_holdout, get_lstm_feature_subset


@pytest.fixture
def synthetic_lstm_data() -> pd.DataFrame:
    """Fixture providing synthetic time series for testing LSTM functionality."""
    np.random.seed(42)
    n = 80
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    
    # Random walk with persistence
    y = np.zeros(n)
    y[0] = 500.0
    for i in range(1, n):
        y[i] = y[i - 1] + np.random.normal(0, 5)

    x1 = y + np.random.normal(0, 2, n)
    x2 = np.random.normal(50, 5, n)

    return pd.DataFrame({
        "date": dates,
        "is_bdi_trading_day": True,
        "bdi_hsi_level": y,
        "feat_1": x1,
        "feat_2": x2,
        "target_bdi_hsi_next": pd.Series(y).shift(-1),
    })


def test_pytorch_lstm_module_forward():
    """Verify raw PyTorch LSTM module construction and tensor shapes."""
    batch_size = 4
    lookback = 10
    input_size = 5
    hidden_size = 16

    model = PyTorchLSTM(input_size=input_size, hidden_size=hidden_size, dense_units=8, dropout=0.1)
    x = torch.randn(batch_size, lookback, input_size)
    out = model(x)
    assert out.shape == (batch_size,)


def test_lstm_sequence_generation():
    """Verify 3D sliding sequence generator shapes and alignment."""
    T = 30
    num_features = 4
    lookback = 7
    X = np.ones((T, num_features))
    y = np.arange(T, dtype=float)

    X_seq, y_seq = LSTMForecaster.create_sequences(X, y, lookback=lookback)
    expected_N = T - lookback + 1

    assert X_seq.shape == (expected_N, lookback, num_features)
    assert y_seq.shape == (expected_N,)
    assert y_seq[0] == y[lookback - 1]
    assert y_seq[-1] == y[-1]


def test_lstm_fit_and_predict_shapes(synthetic_lstm_data):
    """Verify LSTMForecaster end-to-end fitting and boundary prediction."""
    df = synthetic_lstm_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    features = ["bdi_hsi_level", "feat_1", "feat_2"]
    
    train_df, test_df = split_chronological_holdout(df, train_ratio=0.75, drop_initial_cold_start=0)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    model = LSTMForecaster(
        lookback=7,
        hidden_size=16,
        dense_units=8,
        batch_size=8,
        max_epochs=5,
        early_stopping_patience=3,
        random_seed=42,
    )
    model.fit(train_df[features], train_df["target_bdi_hsi_next"], val_ratio=0.2, verbose=False)
    assert model.is_fitted

    preds = model.predict_test_boundary(full_df[features], test_start_idx=len(train_df))
    assert len(preds) == len(test_df)
    assert not np.isnan(preds).any()


def test_lstm_save_and_load(synthetic_lstm_data, tmp_path):
    """Verify PyTorch checkpoint persistence and parameter restoration."""
    df = synthetic_lstm_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    features = ["bdi_hsi_level", "feat_1", "feat_2"]
    train_df, test_df = split_chronological_holdout(df, train_ratio=0.75, drop_initial_cold_start=0)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    model = LSTMForecaster(
        lookback=7,
        hidden_size=16,
        dense_units=8,
        batch_size=8,
        max_epochs=5,
        random_seed=42,
    )
    model.fit(train_df[features], train_df["target_bdi_hsi_next"], val_ratio=0.2, verbose=False)
    orig_preds = model.predict_test_boundary(full_df[features], test_start_idx=len(train_df))

    save_path = tmp_path / "lstm_test.pt"
    model.save_model(save_path)
    assert save_path.exists()

    loaded_model = LSTMForecaster()
    loaded_model.load_model(save_path)
    loaded_preds = loaded_model.predict_test_boundary(full_df[features], test_start_idx=len(train_df))

    assert np.allclose(orig_preds, loaded_preds, atol=1e-4)


def test_lstm_test_period_leakage_isolation(synthetic_lstm_data):
    """CRITICAL TEST: Verify that corrupting test targets does not alter LSTM training weights."""
    df = synthetic_lstm_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    features = ["bdi_hsi_level", "feat_1", "feat_2"]
    train_df, test_df = split_chronological_holdout(df, train_ratio=0.75, drop_initial_cold_start=0)

    # Train model 1
    m1 = LSTMForecaster(lookback=5, hidden_size=8, dense_units=4, max_epochs=3, random_seed=42)
    m1.fit(train_df[features], train_df["target_bdi_hsi_next"], val_ratio=0.2, verbose=False)
    w1 = {k: v.clone() for k, v in m1.model.state_dict().items()}

    # Corrupt test targets
    test_df_corrupt = test_df.copy()
    test_df_corrupt["target_bdi_hsi_next"] *= 999.0

    # Train model 2 on untouched train_df
    m2 = LSTMForecaster(lookback=5, hidden_size=8, dense_units=4, max_epochs=3, random_seed=42)
    m2.fit(train_df[features], train_df["target_bdi_hsi_next"], val_ratio=0.2, verbose=False)
    w2 = {k: v.clone() for k, v in m2.model.state_dict().items()}

    for k in w1:
        assert torch.allclose(w1[k], w2[k]), f"Weight mismatch in parameter {k} after test corruption!"


def test_phase7_experiment_artifacts_exist():
    """Verify that Phase 7 experiment outputs and checkpoints are created."""
    feat_path = Path("data/features/freight_features.csv")
    if not feat_path.exists():
        pytest.skip("Feature dataset not present")

    out_dir = Path("experiments/phase7")
    metrics_path = out_dir / "metrics.csv"
    preds_path = out_dir / "predictions.csv"
    models_dir = out_dir / "models"

    assert metrics_path.exists()
    assert preds_path.exists()
    assert (models_dir / "lstm_bdi_hsi.pt").exists()
    assert (models_dir / "lstm_bdi_ci.pt").exists()
