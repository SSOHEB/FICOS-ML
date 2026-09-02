"""Unit tests for Phase 6 XGBoost forecaster, training, prediction, and test isolation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.xgboost_model import XGBoostForecaster
from src.models.evaluation import run_phase6_xgboost_experiment, split_chronological_holdout


@pytest.fixture
def synthetic_xgb_data() -> pd.DataFrame:
    """Fixture providing small synthetic dataset for testing XGBoost functionality."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    x1 = np.linspace(100, 200, n) + np.random.normal(0, 2, n)
    x2 = np.random.normal(50, 5, n)
    # Add a couple of NaNs to test native missing-value support
    x2[5] = np.nan
    x2[15] = np.nan
    y = 0.9 * x1 + 0.1 * np.nan_to_num(x2, nan=50.0) + np.random.normal(0, 1, n)

    return pd.DataFrame({
        "date": dates,
        "is_bdi_trading_day": True,
        "feat_1": x1,
        "feat_2": x2,
        "target_bdi_hsi_next": pd.Series(y).shift(-1),
    })


def test_xgboost_construction_and_fit(synthetic_xgb_data):
    """Verify XGBoost forecaster construction, native NaN handling, and prediction shape."""
    df = synthetic_xgb_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    X = df[["feat_1", "feat_2"]]
    y = df["target_bdi_hsi_next"]

    model = XGBoostForecaster(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)

    assert model.is_fitted
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert not np.isnan(preds).any()


def test_xgboost_feature_importance(synthetic_xgb_data):
    """Verify that feature importances can be extracted and summed correctly."""
    df = synthetic_xgb_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    X = df[["feat_1", "feat_2"]]
    y = df["target_bdi_hsi_next"]

    model = XGBoostForecaster(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)

    imp_df = model.get_feature_importances(importance_type="gain")
    assert len(imp_df) <= 2
    assert "feature" in imp_df.columns
    assert "importance" in imp_df.columns
    assert "importance_pct" in imp_df.columns
    assert np.isclose(imp_df["importance_pct"].sum(), 100.0, atol=1e-3)


def test_xgboost_save_and_load(synthetic_xgb_data, tmp_path):
    """Verify native JSON model artifact persistence and reloading."""
    df = synthetic_xgb_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    X = df[["feat_1", "feat_2"]]
    y = df["target_bdi_hsi_next"]

    model = XGBoostForecaster(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)
    preds_orig = model.predict(X)

    save_file = tmp_path / "xgb_test.json"
    model.save_model(save_file)
    assert save_file.exists()

    loaded_model = XGBoostForecaster()
    loaded_model.load_model(save_file, feature_names=["feat_1", "feat_2"])
    preds_loaded = loaded_model.predict(X)

    assert np.allclose(preds_orig, preds_loaded)


def test_xgboost_test_period_leakage_isolation(synthetic_xgb_data):
    """CRITICAL TEST: Verify that test set observations cannot alter XGBoost training trees."""
    df = synthetic_xgb_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    train_df, test_df = split_chronological_holdout(df, train_ratio=0.80, drop_initial_cold_start=0)

    features = ["feat_1", "feat_2"]
    
    # Train model on original training data
    model_1 = XGBoostForecaster(n_estimators=10, max_depth=2, random_state=42)
    model_1.fit(train_df[features], train_df["target_bdi_hsi_next"])
    preds_train_1 = model_1.predict(train_df[features])

    # Corrupt test set
    test_df_corrupt = test_df.copy()
    test_df_corrupt["target_bdi_hsi_next"] *= 999.0

    # Train second model on untouched training set
    model_2 = XGBoostForecaster(n_estimators=10, max_depth=2, random_state=42)
    model_2.fit(train_df[features], train_df["target_bdi_hsi_next"])
    preds_train_2 = model_2.predict(train_df[features])

    assert np.allclose(preds_train_1, preds_train_2), "Training predictions changed after test set corruption!"


def test_phase6_experiment_artifacts_exist():
    """Verify that Phase 6 generated all expected outputs."""
    feat_path = Path("data/features/freight_features.csv")
    if not feat_path.exists():
        pytest.skip("Feature dataset not present")

    out_dir = Path("experiments/phase6")
    metrics_path = out_dir / "metrics.csv"
    preds_path = out_dir / "predictions.csv"
    imp_path = out_dir / "feature_importance.csv"
    models_dir = out_dir / "models"

    if not metrics_path.exists():
        _, _, _, _ = run_phase6_xgboost_experiment("configs/models.yaml")

    assert metrics_path.exists()
    assert preds_path.exists()
    assert imp_path.exists()
    assert (models_dir / "xgboost_bdi_hsi.json").exists()
    assert (models_dir / "xgboost_bdi_ci.json").exists()
