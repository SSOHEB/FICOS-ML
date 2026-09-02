"""Unit tests and data leakage verification tests for Phase 4 feature engineering."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.lags import create_autoregressive_lags, create_cross_vessel_lags
from src.features.transformations import (
    create_differences,
    create_percentage_changes,
    create_log_returns,
)
from src.features.rolling import (
    create_rolling_statistics,
    create_rolling_volatility,
)
from src.features.exogenous import (
    create_macro_features,
    create_geopolitical_features,
    create_weather_features,
    create_calendar_features,
)
from src.features.pipeline import (
    build_features_dataframe,
    run_feature_pipeline,
    load_feature_config,
)
from src.data.schemas import DATE_COLUMN, TARGET_COLUMNS


@pytest.fixture
def synthetic_trading_df() -> pd.DataFrame:
    """Fixture providing synthetic consecutive trading observations."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=50, freq="B")  # Business/trading days
    data = {
        "date": dates,
        "is_bdi_trading_day": True,
        "bdi_hsi": np.linspace(400, 600, 50),
        "bdi_si": np.linspace(800, 1000, 50),
        "bdi_pi": np.linspace(900, 1200, 50),
        "bdi_ci": np.linspace(1500, 2500, 50),
        "wti_usd_bbl": np.random.uniform(50, 70, 50),
        "brent_usd_bbl": np.random.uniform(55, 75, 50),
        "usd_inr": np.random.uniform(70, 75, 50),
        "gpr": np.random.uniform(80, 120, 50),
        "gpr_acts": np.random.uniform(20, 60, 50),
        "gpr_threats": np.random.uniform(40, 80, 50),
        "gpr_ma7": np.random.uniform(85, 115, 50),
        "gpr_ma30": np.random.uniform(90, 110, 50),
        "wind_speed_max_kmh": np.random.uniform(10, 30, 50),
        "precip_mm": np.random.uniform(0, 10, 50),
        "pressure_hpa": np.random.uniform(1000, 1020, 50),
    }
    return pd.DataFrame(data)


# -------------------------------------------------------------
# 1. Unit Tests for Feature Functions
# -------------------------------------------------------------

def test_autoregressive_lags_causality(synthetic_trading_df):
    """Verify that lag features strictly shift values backward."""
    df_lags = create_autoregressive_lags(synthetic_trading_df, ["bdi_hsi"], lags=[1, 2, 5])
    
    # Lag 1 at index 1 must equal original level at index 0
    assert np.isnan(df_lags["bdi_hsi_lag_1"].iloc[0])
    assert df_lags["bdi_hsi_lag_1"].iloc[1] == synthetic_trading_df["bdi_hsi"].iloc[0]
    assert df_lags["bdi_hsi_lag_2"].iloc[2] == synthetic_trading_df["bdi_hsi"].iloc[0]


def test_invalid_lag_rejection():
    """Verify that non-positive lags are rejected to prevent future leakage."""
    df = pd.DataFrame({"y": [1, 2, 3]})
    with pytest.raises(ValueError, match="Lag must be >= 1"):
        create_autoregressive_lags(df, ["y"], lags=[0])

    with pytest.raises(ValueError, match="Lag must be >= 1"):
        create_autoregressive_lags(df, ["y"], lags=[-1])


def test_cross_vessel_lags(synthetic_trading_df):
    """Verify causal cross-vessel feature construction."""
    cross_map = {"bdi_hsi": ["bdi_si"]}
    df_cross = create_cross_vessel_lags(synthetic_trading_df, cross_map, lags=[1, 5])
    assert "cross_bdi_si_lag_1" in df_cross.columns
    assert df_cross["cross_bdi_si_lag_1"].iloc[1] == synthetic_trading_df["bdi_si"].iloc[0]


def test_differences_and_pct_changes(synthetic_trading_df):
    """Verify causal rate of change and difference transformations."""
    df_diff = create_differences(synthetic_trading_df, ["bdi_hsi"], windows=[1])
    df_pct = create_percentage_changes(synthetic_trading_df, ["bdi_hsi"], windows=[1])

    expected_diff = synthetic_trading_df["bdi_hsi"].iloc[1] - synthetic_trading_df["bdi_hsi"].iloc[0]
    assert np.isclose(df_diff["bdi_hsi_diff_1"].iloc[1], expected_diff)

    expected_pct = (expected_diff / synthetic_trading_df["bdi_hsi"].iloc[0]) * 100.0
    assert np.isclose(df_pct["bdi_hsi_pct_change_1"].iloc[1], expected_pct)


def test_rolling_statistics_causality(synthetic_trading_df):
    """Verify that rolling statistics use only past and current values (no future data)."""
    df_roll = create_rolling_statistics(synthetic_trading_df, ["bdi_hsi"], windows=[7], statistics=["mean", "max"])
    
    # Check that rolling mean at index 6 equals mean of indices 0 through 6
    expected_mean = synthetic_trading_df["bdi_hsi"].iloc[:7].mean()
    assert np.isclose(df_roll["bdi_hsi_roll_mean_7"].iloc[6], expected_mean)

    expected_max = synthetic_trading_df["bdi_hsi"].iloc[:7].max()
    assert np.isclose(df_roll["bdi_hsi_roll_max_7"].iloc[6], expected_max)


def test_exogenous_and_calendar_features(synthetic_trading_df):
    """Verify exogenous and calendar feature generation."""
    df_macro = create_macro_features(synthetic_trading_df, energy_cols=["wti_usd_bbl"], fx_cols=["usd_inr"], lags=[1])
    df_gpr = create_geopolitical_features(synthetic_trading_df, gpr_cols=["gpr"], lags=[1])
    df_cal = create_calendar_features(synthetic_trading_df, date_col="date", features=["month", "quarter", "day_of_week"])

    assert "wti_usd_bbl_lag_1" in df_macro.columns
    assert "gpr_lag_1" in df_gpr.columns
    assert "cal_month" in df_cal.columns
    assert "cal_quarter" in df_cal.columns
    assert "cal_day_of_week" in df_cal.columns


# -------------------------------------------------------------
# 2. Critical Data Leakage & Target Alignment Tests
# -------------------------------------------------------------

def test_target_alignment_strict_causality(synthetic_trading_df):
    """CRITICAL TEST: Verify that target_{col}_next at row t strictly equals observation at t+1."""
    feature_df = build_features_dataframe(synthetic_trading_df, filter_trading_days=False)

    for col in ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"]:
        target_col = f"target_{col}_next"
        level_col = f"{col}_level"
        
        # Verify alignment for all rows except the final unobserved forecast origin
        for t in range(len(synthetic_trading_df) - 1):
            assert feature_df[target_col].iloc[t] == synthetic_trading_df[col].iloc[t + 1], (
                f"Leakage or alignment error at index {t} for target {col}"
            )
        
        # Final row must be NaN (future unobserved target)
        assert np.isnan(feature_df[target_col].iloc[-1])


def test_future_independence(synthetic_trading_df):
    """CRITICAL TEST: Modify future values at t+5 and verify that feature values at t remain identical."""
    df_original = synthetic_trading_df.copy()
    features_orig = build_features_dataframe(df_original, filter_trading_days=False)

    # Corrupt future rows at index 20 onwards
    df_corrupted = df_original.copy()
    df_corrupted.loc[20:, ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci", "wti_usd_bbl"]] *= 99.0
    features_corrupt = build_features_dataframe(df_corrupted, filter_trading_days=False)

    # Features at rows 0..19 must be bit-for-bit identical
    feat_cols = [c for c in features_orig.columns if not c.startswith("target_") and c != DATE_COLUMN]
    for col in feat_cols:
        orig_slice = features_orig[col].iloc[:20].values
        corrupt_slice = features_corrupt[col].iloc[:20].values
        assert np.allclose(orig_slice, corrupt_slice, equal_nan=True), (
            f"DATA LEAKAGE DETECTED in feature '{col}': past values changed when future data was modified!"
        )


def test_master_dataset_remains_unmodified():
    """Verify that master_dataset.csv file is never altered by feature engineering."""
    master_path = Path("data/processed/master_dataset.csv")
    if not master_path.exists():
        pytest.skip("Master dataset file not present")

    mtime_before = master_path.stat().st_mtime
    _ = run_feature_pipeline("configs/features.yaml")
    mtime_after = master_path.stat().st_mtime
    assert mtime_before == mtime_after, "Master dataset was modified during feature pipeline execution!"
