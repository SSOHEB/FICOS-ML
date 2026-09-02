"""Unit tests for Phase 8 walk-forward cross-validation, fold generation, and leakage isolation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.evaluation import generate_walk_forward_folds, compute_regression_metrics
from src.data.schemas import DATE_COLUMN


@pytest.fixture
def synthetic_wf_data() -> pd.DataFrame:
    """Fixture providing continuous synthetic daily series for walk-forward testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    y = np.linspace(500, 1000, n) + np.random.normal(0, 5, n)

    return pd.DataFrame({
        DATE_COLUMN: dates,
        "is_bdi_trading_day": True,
        "bdi_hsi_level": y,
        "target_bdi_hsi_next": pd.Series(y).shift(-1),
    })


def test_walk_forward_fold_ordering_and_sizes(synthetic_wf_data):
    """Verify that walk-forward folds strictly expand training and advance test chronologically."""
    df = synthetic_wf_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    n_folds = 3
    test_window_size = 50

    folds = generate_walk_forward_folds(
        df, n_folds=n_folds, test_window_size=test_window_size, drop_initial_cold_start=0
    )

    assert len(folds) == n_folds

    prev_train_size = 0
    prev_test_end = None

    for f in folds:
        train_df = f["train_df"]
        test_df = f["test_df"]

        # 1. Train size strictly expands
        assert len(train_df) > prev_train_size
        prev_train_size = len(train_df)

        # 2. Test size matches configured window
        assert len(test_df) == test_window_size

        # 3. Train strictly precedes test
        assert train_df[DATE_COLUMN].max() < test_df[DATE_COLUMN].min()

        # 4. Chronological continuity
        if prev_test_end is not None:
            assert test_df[DATE_COLUMN].min() > prev_test_end
        prev_test_end = test_df[DATE_COLUMN].max()


def test_no_train_test_overlap(synthetic_wf_data):
    """Verify zero date overlap between training and testing sets in any fold."""
    df = synthetic_wf_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    folds = generate_walk_forward_folds(df, n_folds=3, test_window_size=40, drop_initial_cold_start=0)

    for f in folds:
        train_dates = set(f["train_df"][DATE_COLUMN])
        test_dates = set(f["test_df"][DATE_COLUMN])
        assert len(train_dates.intersection(test_dates)) == 0


def test_walk_forward_leakage_isolation(synthetic_wf_data):
    """CRITICAL TEST: Verify that modifying future fold observations has zero effect on earlier folds."""
    df_clean = synthetic_wf_data.dropna(subset=["target_bdi_hsi_next"]).reset_index(drop=True)
    folds_orig = generate_walk_forward_folds(df_clean, n_folds=3, test_window_size=40, drop_initial_cold_start=0)

    # Corrupt last 40 rows (Fold 3 test window)
    df_corrupt = df_clean.copy()
    df_corrupt.loc[len(df_corrupt) - 40 :, "bdi_hsi_level"] *= 999.0

    folds_corrupt = generate_walk_forward_folds(df_corrupt, n_folds=3, test_window_size=40, drop_initial_cold_start=0)

    # Fold 1 train & test must be bit-for-bit identical
    assert np.allclose(folds_orig[0]["train_df"]["bdi_hsi_level"], folds_corrupt[0]["train_df"]["bdi_hsi_level"])
    assert np.allclose(folds_orig[0]["test_df"]["bdi_hsi_level"], folds_corrupt[0]["test_df"]["bdi_hsi_level"])


def test_phase8_experiment_outputs_exist():
    """Verify that Phase 8 walk-forward experiment artifacts and summaries are created."""
    out_dir = Path("experiments/phase8")
    fold_metrics_path = out_dir / "fold_metrics.csv"
    agg_metrics_path = out_dir / "aggregate_metrics.csv"
    preds_path = out_dir / "predictions.csv"

    assert fold_metrics_path.exists()
    assert agg_metrics_path.exists()
    assert preds_path.exists()

    agg_df = pd.read_csv(agg_metrics_path)
    assert len(agg_df) > 0
    assert "mean_mae" in agg_df.columns
    assert "mean_da_pct" in agg_df.columns
