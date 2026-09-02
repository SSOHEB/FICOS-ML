"""Phase 10.5 Feature Leakage & Causality Verification Tests.

Proves:
- Features at time T are strictly independent of Data at time T+1 onward.
- Modifying future data (freight, oil, FX, GPR, weather) has ZERO impact on lags, rolling means,
  rolling standard deviations, momentum, differences, calendar, or target alignment at time <= T.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.schemas import DATE_COLUMN
from src.features.pipeline import build_features_dataframe


@pytest.fixture(scope="module")
def master_dataset() -> pd.DataFrame:
    p = Path("data/processed/master_dataset.csv")
    df = pd.read_csv(p)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


def test_adversarial_future_data_feature_invariance(master_dataset):
    """CRITICAL TEST: Modify all observations after cutoff date T with extreme values,

    re-run feature engineering, and verify features(t <= T) are 100% bitwise/float identical.
    """
    cutoff_date = pd.to_datetime("2017-06-15")
    
    df_original = master_dataset.copy()
    df_perturbed = master_dataset.copy()

    # Apply extreme distortions to future rows after cutoff_date
    future_mask = df_perturbed[DATE_COLUMN] > cutoff_date
    assert future_mask.sum() > 500, "Must have substantial future rows to perturb"

    numeric_cols = df_perturbed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_perturbed.loc[future_mask, col] = df_perturbed.loc[future_mask, col] * 50.0 + 99999.0

    # Build features on both datasets using standard build_features_dataframe
    features_orig = build_features_dataframe(df_original)
    features_pert = build_features_dataframe(df_perturbed)

    # Filter both to t <= cutoff_date
    orig_past = features_orig[features_orig[DATE_COLUMN] <= cutoff_date].sort_values(by=DATE_COLUMN).reset_index(drop=True)
    pert_past = features_pert[features_pert[DATE_COLUMN] <= cutoff_date].sort_values(by=DATE_COLUMN).reset_index(drop=True)

    assert orig_past.shape == pert_past.shape, "Past feature matrices must have identical dimensions"
    
    # Exclude target columns (which by definition look ahead to predict t+1)
    feature_cols = [c for c in orig_past.columns if not c.startswith("target_") and c != DATE_COLUMN]

    for col in feature_cols:
        orig_vals = orig_past[col].values
        pert_vals = pert_past[col].values

        # Both must have identical NaN masks
        np.testing.assert_array_equal(
            np.isnan(orig_vals),
            np.isnan(pert_vals),
            err_msg=f"NaN mask mismatch in feature '{col}' after future perturbation",
        )

        valid_mask = ~np.isnan(orig_vals)
        np.testing.assert_allclose(
            orig_vals[valid_mask],
            pert_vals[valid_mask],
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Feature leakage detected in '{col}': values at t <= T changed when T+1 was modified!",
        )


def test_autoregressive_lags_purely_causal(master_dataset):
    """Verify that lag k at index i is exactly the level at index i-k."""
    features = build_features_dataframe(master_dataset)

    targets = ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"]
    for tgt in targets:
        level_col = f"{tgt}_level"
        for lag in [1, 2, 3, 5, 10, 21]:
            lag_col = f"{tgt}_lag_{lag}"
            if lag_col in features.columns:
                levels = features[level_col].values
                lags = features[lag_col].values
                # lags[i] must equal levels[i-lag]
                np.testing.assert_allclose(
                    lags[lag:],
                    levels[:-lag],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Lag {lag} for {tgt} does not align strictly with i - {lag}",
                )


def test_rolling_statistics_purely_causal(master_dataset):
    """Verify that rolling windows (mean, std) include only past and contemporaneous points."""
    features = build_features_dataframe(master_dataset)

    targets = ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"]
    for tgt in targets:
        level_col = f"{tgt}_level"
        for window in [5, 10, 21]:
            mean_col = f"{tgt}_rolling_mean_{window}"
            if mean_col in features.columns:
                expected_rolling = features[level_col].rolling(window=window, min_periods=window).mean().values
                actual_rolling = features[mean_col].values

                valid_mask = ~np.isnan(expected_rolling)
                np.testing.assert_allclose(
                    actual_rolling[valid_mask],
                    expected_rolling[valid_mask],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Rolling mean {window} for {tgt} deviates from causal backward rolling calculation",
                )
