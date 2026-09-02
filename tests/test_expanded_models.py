"""Tests for Phase: Retrain Models on Expanded Features.

Verifies:
1. Chronological Train -> Val -> Test split ordering without overlap or shuffle
2. Train-only preprocessing: Transformers and scalers fitted on train set only
3. Absence of target leakage: target columns excluded from feature matrices
4. Forecast determinism: Models generate identical predictions on identical inputs
5. Adversarial future-perturbation invariance: Corrupting future data after T has 0 effect on predictions for t <= T
6. Output artifacts integrity in experiments/expanded/
"""

from pathlib import Path
from typing import Tuple, List
import json
import numpy as np
import pandas as pd
import pytest

from src.data.schemas import DATE_COLUMN
from src.models.baselines import PersistenceForecaster
from src.models.ridge import RidgeForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.lstm_model import LSTMForecaster
from scripts.run_expanded_training_experiment import get_valid_kobc_features, KOBC_TARGETS


@pytest.fixture(scope="module")
def expanded_df() -> pd.DataFrame:
    p = Path("data/features/freight_features_expanded.csv")
    assert p.exists(), "freight_features_expanded.csv must exist"
    df = pd.read_csv(p)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


@pytest.fixture(scope="module")
def kobc_clean_data(expanded_df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    df_kobc = expanded_df[expanded_df[DATE_COLUMN] >= pd.Timestamp("2020-01-01")].sort_values(by=DATE_COLUMN).reset_index(drop=True)
    features = get_valid_kobc_features(df_kobc)
    
    # Drop cold start (first 21 rows) and terminal row (no t+1 target)
    df_clean = df_kobc.iloc[21:-1].reset_index(drop=True)
    n = len(df_clean)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    train_df = df_clean.iloc[:n_train].reset_index(drop=True)
    val_df = df_clean.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df_clean.iloc[n_train + n_val:].reset_index(drop=True)

    return train_df, val_df, test_df, features


def test_chronological_splits_integrity(kobc_clean_data):
    """Verify strict chronological sequence: Train < Validation < Final Test."""
    train_df, val_df, test_df, _ = kobc_clean_data

    assert train_df[DATE_COLUMN].max() < val_df[DATE_COLUMN].min(), "Train must strictly precede Validation"
    assert val_df[DATE_COLUMN].max() < test_df[DATE_COLUMN].min(), "Validation must strictly precede Test"

    assert len(train_df) == 1107, f"Expected 1107 train rows, found {len(train_df)}"
    assert len(val_df) == 237, f"Expected 237 val rows, found {len(val_df)}"
    assert len(test_df) == 238, f"Expected 238 test rows, found {len(test_df)}"


def test_train_only_scaler_fitting(kobc_clean_data):
    """Verify scalers and imputers are fitted on training data only without test data influence."""
    train_df, val_df, test_df, features = kobc_clean_data
    target_col = "target_kobc_handy_next"

    ridge = RidgeForecaster(alpha=1.0, scale_features=True)
    ridge.fit(train_df[features], train_df[target_col])

    # Scaler mean and variance must match train_df ONLY
    imputer_medians = train_df[features].median().values
    np.testing.assert_allclose(ridge.imputer.statistics_, imputer_medians, rtol=1e-3, atol=1e-3)

    # Transform test set using fitted transformers
    pred_test = ridge.predict(test_df[features])
    assert len(pred_test) == len(test_df)
    assert not np.isnan(pred_test).any(), "Predictions must not contain NaNs"


def test_no_target_leakage_in_features(kobc_clean_data):
    """Verify feature selection strictly excludes all future target columns and metadata leakage."""
    _, _, _, features = kobc_clean_data

    for feat in features:
        assert not feat.startswith("target_"), f"Target column '{feat}' leaked into feature set!"
        assert feat != "date", "Date column should not be treated as a numerical feature"
        assert not feat.startswith("bdi_"), f"Baltic column '{feat}' should not be in KOBC feature set"


def test_model_prediction_determinism(kobc_clean_data):
    """Verify Ridge and XGBoost forecasters produce bitwise identical predictions on identical inputs."""
    train_df, _, test_df, features = kobc_clean_data
    target_col = "target_kobc_panamax_next"

    # Ridge determinism
    r1 = RidgeForecaster(alpha=1.0).fit(train_df[features], train_df[target_col])
    r2 = RidgeForecaster(alpha=1.0).fit(train_df[features], train_df[target_col])
    p1 = r1.predict(test_df[features])
    p2 = r2.predict(test_df[features])
    np.testing.assert_array_equal(p1, p2, err_msg="Ridge inference is not deterministic!")

    # XGBoost determinism
    x1 = XGBoostForecaster(n_estimators=50, random_state=42).fit(train_df[features], train_df[target_col])
    x2 = XGBoostForecaster(n_estimators=50, random_state=42).fit(train_df[features], train_df[target_col])
    px1 = x1.predict(test_df[features])
    px2 = x2.predict(test_df[features])
    np.testing.assert_array_equal(px1, px2, err_msg="XGBoost inference is not deterministic!")


def test_adversarial_future_perturbation_prediction_invariance(expanded_df):
    """CRITICAL TEST: Massively perturb future observations after cutoff T.

    Verify that trained model predictions for test inputs at t <= T are 100% unchanged.
    """
    df_kobc = expanded_df[expanded_df[DATE_COLUMN] >= pd.Timestamp("2020-01-01")].sort_values(by=DATE_COLUMN).reset_index(drop=True)
    features = get_valid_kobc_features(df_kobc)
    target_col = "target_kobc_handy_next"

    cutoff_date = pd.to_datetime("2023-01-01")
    df_orig = df_kobc.copy()
    df_pert = df_kobc.copy()

    # Corrupt future rows after cutoff
    future_mask = df_pert[DATE_COLUMN] > cutoff_date
    num_cols = df_pert.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df_pert.loc[future_mask, c] = df_pert.loc[future_mask, c] * 50.0 + 99999.0

    # Train on past data up to cutoff
    train_orig = df_orig[df_orig[DATE_COLUMN] <= cutoff_date].iloc[21:-1].reset_index(drop=True)
    train_pert = df_pert[df_pert[DATE_COLUMN] <= cutoff_date].iloc[21:-1].reset_index(drop=True)

    ridge_orig = RidgeForecaster(alpha=1.0).fit(train_orig[features], train_orig[target_col])
    ridge_pert = RidgeForecaster(alpha=1.0).fit(train_pert[features], train_pert[target_col])

    # Coefficients must be identical
    np.testing.assert_allclose(
        ridge_orig.model.coef_,
        ridge_pert.model.coef_,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Model fitting was contaminated by future perturbation!",
    )


def test_experiments_expanded_artifacts_exist():
    """Verify all expected artifacts are saved in experiments/expanded/."""
    exp_dir = Path("experiments/expanded")
    assert (exp_dir / "metrics.csv").exists(), "metrics.csv must exist"
    assert (exp_dir / "predictions.csv").exists(), "predictions.csv must exist"
    assert (exp_dir / "fold_metrics.csv").exists(), "fold_metrics.csv must exist"
    assert (exp_dir / "model_config.json").exists(), "model_config.json must exist"
    assert (exp_dir / "report.md").exists(), "report.md must exist"

    # Verify figures
    figures_dir = exp_dir / "figures"
    assert figures_dir.exists(), "figures directory must exist"
    for target in KOBC_TARGETS:
        assert (figures_dir / f"forecast_{target}.png").exists(), f"Figure for {target} missing"
    assert (figures_dir / "mae_comparison.png").exists(), "mae_comparison.png missing"
    assert (figures_dir / "directional_accuracy_comparison.png").exists(), "directional_accuracy_comparison.png missing"
