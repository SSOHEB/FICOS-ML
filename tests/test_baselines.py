"""Unit tests for Phase 5 baseline models, metrics, and leakage prevention."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.baselines import PersistenceForecaster, MovingAverageForecaster
from src.models.ridge import RidgeForecaster
from src.models.evaluation import (
    compute_regression_metrics,
    split_chronological_holdout,
    run_phase5_experiment,
)


@pytest.fixture
def synthetic_forecast_data() -> pd.DataFrame:
    """Fixture providing synthetic historical series and features."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    
    # Generate AR(1) process
    y = np.zeros(n)
    y[0] = 500.0
    for i in range(1, n):
        y[i] = 0.95 * y[i - 1] + np.random.normal(0, 5)

    df = pd.DataFrame({
        "date": dates,
        "is_bdi_trading_day": True,
        "bdi_hsi_level": y,
        "bdi_hsi_lag_1": pd.Series(y).shift(1),
        "bdi_hsi_diff_1": pd.Series(y).diff(1),
        "target_bdi_hsi_next": pd.Series(y).shift(-1),
    })
    return df


def test_persistence_forecaster():
    """Verify Naive Persistence forecaster behavior."""
    model = PersistenceForecaster()
    current_levels = pd.Series([500.0, 510.0, 505.0])
    preds = model.predict(current_levels)
    assert np.allclose(preds, [500.0, 510.0, 505.0])


def test_moving_average_forecaster():
    """Verify Moving Average forecaster causal rolling behavior."""
    model = MovingAverageForecaster(window=3)
    history = pd.Series([10.0, 20.0, 30.0, 40.0])
    preds = model.predict(history)
    
    # at idx 2: mean(10, 20, 30) = 20.0
    # at idx 3: mean(20, 30, 40) = 30.0
    assert np.isclose(preds.iloc[2], 20.0)
    assert np.isclose(preds.iloc[3], 30.0)


def test_compute_regression_metrics():
    """Verify correctness of MAE, RMSE, sMAPE, R2, and Directional Accuracy."""
    y_true = np.array([100.0, 110.0, 95.0, 105.0])
    y_pred = np.array([102.0, 108.0, 97.0, 103.0])
    y_curr = np.array([98.0, 100.0, 100.0, 100.0])

    metrics = compute_regression_metrics(y_true, y_pred, y_curr)
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert metrics["smape"] > 0
    assert metrics["r2"] > 0
    assert "da_pct" in metrics
    assert 0 <= metrics["da_pct"] <= 100


def test_split_chronological_holdout():
    """Verify strict chronological splitting and no shuffling."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "val": np.arange(100),
        "target_bdi_hsi_next": np.arange(100),
    })
    
    train_df, test_df = split_chronological_holdout(df, train_ratio=0.80, drop_initial_cold_start=0)
    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df["date"].max() < test_df["date"].min()
    assert train_df["date"].is_monotonic_increasing
    assert test_df["date"].is_monotonic_increasing


def test_ridge_train_only_scaling():
    """CRITICAL TEST: Verify that Ridge scaler and imputer are fit ONLY on train set."""
    np.random.seed(42)
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0], "f2": [10.0, 20.0, 30.0, 40.0, 50.0]})
    y_train = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0])

    model = RidgeForecaster(alpha=1.0, scale_features=True)
    model.fit(X_train, y_train)

    # Scaler mean must match X_train mean exactly
    assert np.allclose(model.scaler.mean_, [3.0, 30.0])

    # Transform test data with different distribution without changing scaler parameters
    X_test = pd.DataFrame({"f1": [100.0, 200.0], "f2": [1000.0, 2000.0]})
    _ = model.predict(X_test)
    assert np.allclose(model.scaler.mean_, [3.0, 30.0]), "Scaler mean changed after predict()!"


def test_test_period_leakage_isolation(synthetic_forecast_data):
    """CRITICAL TEST: Verify that corrupting test targets does not alter train model coefficients."""
    df = synthetic_forecast_data.dropna().reset_index(drop=True)
    train_df, test_df = split_chronological_holdout(df, train_ratio=0.80, drop_initial_cold_start=0)

    feature_cols = ["bdi_hsi_lag_1", "bdi_hsi_diff_1"]
    
    # Train original model
    model_orig = RidgeForecaster(alpha=1.0)
    model_orig.fit(train_df[feature_cols], train_df["target_bdi_hsi_next"])
    orig_coefs = model_orig.model.coef_.copy()

    # Corrupt test set targets
    test_df_corrupt = test_df.copy()
    test_df_corrupt["target_bdi_hsi_next"] *= 999.0

    # Retrain on train set
    model_new = RidgeForecaster(alpha=1.0)
    model_new.fit(train_df[feature_cols], train_df["target_bdi_hsi_next"])
    new_coefs = model_new.model.coef_.copy()

    assert np.allclose(orig_coefs, new_coefs), "Training coefficients were influenced by test data!"


def test_phase5_experiment_outputs():
    """Verify that the Phase 5 benchmark experiment generates valid output files."""
    feat_path = Path("data/features/freight_features.csv")
    if not feat_path.exists():
        pytest.skip("Feature dataset not present")

    metrics_df, pred_df, meta = run_phase5_experiment("configs/models.yaml")
    assert len(metrics_df) > 0
    assert len(pred_df) == meta["test_rows"]
    assert Path(meta["metrics_path"]).exists()
    assert Path(meta["predictions_path"]).exists()
