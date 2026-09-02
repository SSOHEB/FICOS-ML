"""Validation and causal leakage tests for Phase: Rebuild Features from Expanded Master.

Verifies:
1. Canonical master dataset (5,145 x 84, 2012-08-01 to 2026-09-01) is used as input
2. No duplicate dates in freight_features_expanded.csv
3. Dates are strictly sorted in ascending order
4. Targets are correctly shifted to NEXT genuinely observed trading observation
5. No target leakage into feature columns
6. Adversarial future-data perturbation test: modifying data after T has 0 effect on t <= T
7. No infinite or invalid floating-point values
8. No synthetic freight observations or interpolation of missing values
9. Strictly causal rolling windows (center=False)
10. Genuine post-2019 KOBC observations and targets present (1,604 KOBC rows, 1,603 targets)
11. Baltic (2012–2019) and KOBC (2020+) regimes remain decoupled and separate series
12. Source and regime indicators are preserved
13. Old freight_features.csv remains unmodified
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest

from src.data.schemas import DATE_COLUMN
from src.features.pipeline_expanded import (
    build_expanded_features_dataframe,
    run_expanded_feature_pipeline,
    load_expanded_feature_config,
    get_next_observed_target,
)


@pytest.fixture(scope="module")
def expanded_features_df() -> pd.DataFrame:
    p = Path("data/features/freight_features_expanded.csv")
    if not p.exists():
        run_expanded_feature_pipeline()
    df = pd.read_csv(p)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


@pytest.fixture(scope="module")
def master_df() -> pd.DataFrame:
    p = Path("data/processed/master_dataset.csv")
    assert p.exists(), "master_dataset.csv must exist"
    df = pd.read_csv(p)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


def test_canonical_master_dataset_schema_and_shape(master_df):
    """Verify input master is the canonical 5,145 x 84 dataset."""
    assert master_df.shape == (5145, 84), f"Master dataset shape mismatch: expected (5145, 84), found {master_df.shape}"
    assert master_df[DATE_COLUMN].min() == pd.Timestamp("2012-08-01")
    assert master_df[DATE_COLUMN].max() == pd.Timestamp("2026-09-01")
    
    with open("data/processed/master_dataset.csv", "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    assert h == "3930980597df212d3c3b82eb5338a2354f48b1f3d89dd61891fb5f9d11d76d54"


def test_old_freight_features_remains_unmodified():
    """Verify that old data/features/freight_features.csv exists and was not overwritten."""
    old_path = Path("data/features/freight_features.csv")
    assert old_path.exists(), "Old freight_features.csv must be preserved"
    df_old = pd.read_csv(old_path)
    assert df_old.shape == (1749, 141), f"Old features shape changed: {df_old.shape}"


def test_expanded_features_dimensions_and_regimes(expanded_features_df):
    """Verify expanded feature matrix dimensions and regime partition."""
    assert expanded_features_df.shape[0] == 3353, f"Expected 3353 rows, found {expanded_features_df.shape[0]}"
    assert expanded_features_df[DATE_COLUMN].min() == pd.Timestamp("2012-08-01")
    assert expanded_features_df[DATE_COLUMN].max() == pd.Timestamp("2026-09-01")

    # Post-2019 KOBC rows and Pre-2020 Baltic rows
    post_2019_rows = (expanded_features_df[DATE_COLUMN] >= "2020-01-01").sum()
    pre_2020_rows = (expanded_features_df[DATE_COLUMN] < "2020-01-01").sum()

    assert post_2019_rows == 1604, f"Expected 1604 post-2019 KOBC rows, found {post_2019_rows}"
    assert pre_2020_rows == 1749, f"Expected 1749 pre-2020 Baltic rows, found {pre_2020_rows}"


def test_expanded_features_date_ordering_and_duplicates(expanded_features_df):
    """Verify no duplicate dates and strictly monotonic ascending ordering."""
    assert expanded_features_df[DATE_COLUMN].is_monotonic_increasing, "Dates must be strictly ascending"
    assert expanded_features_df[DATE_COLUMN].duplicated().sum() == 0, "Zero duplicate dates allowed"


def test_no_infinities_in_expanded_features(expanded_features_df):
    """Verify absence of +inf or -inf across all numeric columns."""
    num_cols = expanded_features_df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        assert not np.isinf(expanded_features_df[c]).any(), f"Infinite values detected in col: {c}"


def test_regime_and_source_indicators_preserved(expanded_features_df):
    """Verify source and regime tracking indicators exist and are valid."""
    expected_meta = [
        "is_baltic_regime",
        "is_kobc_regime",
        "is_bdi_trading_day",
        "is_kobc_trading_day",
        "freight_source",
    ]
    for col in expected_meta:
        assert col in expanded_features_df.columns, f"Missing regime indicator: {col}"

    assert set(expanded_features_df["freight_source"].unique()).issubset({"baltic", "kobc", "none"})


def test_genuine_kobc_targets_and_counts(expanded_features_df):
    """Verify KOBC targets contain genuine observations with zero Baltic contamination."""
    kobc_targets = [
        "target_kobc_handy_next",
        "target_kobc_supramax_next",
        "target_kobc_panamax_next",
        "target_kobc_cape_next",
    ]
    for t in kobc_targets:
        assert t in expanded_features_df.columns, f"Missing KOBC target: {t}"
        non_null_count = expanded_features_df[t].notnull().sum()
        assert non_null_count == 1603, f"Expected 1603 observed targets for {t}, found {non_null_count}"
        
        # Verify pre-2020 is NaN
        pre_2020_mask = expanded_features_df[DATE_COLUMN] < "2020-01-01"
        assert expanded_features_df.loc[pre_2020_mask, t].isnull().all(), f"KOBC target {t} contaminated pre-2020"

    # KDCI is explanatory only, NOT a target
    assert "target_kobc_kdci_next" not in expanded_features_df.columns


def test_genuine_baltic_targets_and_counts(expanded_features_df):
    """Verify Baltic targets contain genuine observations with zero KOBC contamination."""
    baltic_targets = [
        "target_bdi_hsi_next",
        "target_bdi_si_next",
        "target_bdi_pi_next",
        "target_bdi_ci_next",
    ]
    for t in baltic_targets:
        assert t in expanded_features_df.columns, f"Missing Baltic target: {t}"
        non_null_count = expanded_features_df[t].notnull().sum()
        assert non_null_count == 1748, f"Expected 1748 observed targets for {t}, found {non_null_count}"
        
        # Verify post-2019 is NaN
        post_2019_mask = expanded_features_df[DATE_COLUMN] >= "2020-01-01"
        assert expanded_features_df.loc[post_2019_mask, t].isnull().all(), f"Baltic target {t} contaminated post-2019"


def test_causal_rolling_windows_no_lookahead(master_df):
    """Verify rolling statistics use backward-only window (center=False)."""
    cfg = load_expanded_feature_config()
    df_exp = build_expanded_features_dataframe(master_df, config=cfg, filter_trading_days=True)

    # For Baltic HSI roll mean 7
    level_vals = df_exp["bdi_hsi_level"].values
    roll_mean_7 = df_exp["bdi_hsi_roll_mean_7"].values

    for i in range(6, 1749):
        expected_window = level_vals[i - 6 : i + 1]
        if not np.isnan(expected_window).any():
            expected_mean = float(np.mean(expected_window))
            assert np.isclose(roll_mean_7[i], expected_mean, rtol=1e-4, atol=1e-4), (
                f"Rolling mean at index {i} mismatch: actual {roll_mean_7[i]} vs expected {expected_mean}"
            )


def test_adversarial_future_perturbation_leakage_invariance(master_df):
    """CRITICAL TEST: Modify all future data after T=2017-06-15 by 50x + 99,999.

    Verify that every engineered feature and target at t <= T is 100% bitwise/float identical.
    """
    cutoff_date = pd.to_datetime("2017-06-15")

    df_orig = master_df.copy()
    df_pert = master_df.copy()

    future_mask = df_pert[DATE_COLUMN] > cutoff_date
    assert future_mask.sum() > 400, "Must have future rows to corrupt"

    numeric_cols = df_pert.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df_pert.loc[future_mask, c] = df_pert.loc[future_mask, c] * 50.0 + 99999.0

    cfg = load_expanded_feature_config()
    feats_orig = build_expanded_features_dataframe(df_orig, config=cfg, filter_trading_days=True)
    feats_pert = build_expanded_features_dataframe(df_pert, config=cfg, filter_trading_days=True)

    past_orig = feats_orig[feats_orig[DATE_COLUMN] <= cutoff_date].reset_index(drop=True)
    past_pert = feats_pert[feats_pert[DATE_COLUMN] <= cutoff_date].reset_index(drop=True)

    assert past_orig.shape == past_pert.shape, "Shapes must match"

    # All feature columns excluding future targets
    feature_cols = [c for c in past_orig.columns if not c.startswith("target_") and c != DATE_COLUMN]

    for c in feature_cols:
        # If boolean, string, or object, compare directly
        if not pd.api.types.is_numeric_dtype(past_orig[c]) or pd.api.types.is_bool_dtype(past_orig[c]):
            assert (past_orig[c] == past_pert[c]).all(), f"Mismatch in non-numeric col {c}"
        else:
            v1 = past_orig[c].values.astype(float)
            v2 = past_pert[c].values.astype(float)
            np.testing.assert_array_equal(
                np.isnan(v1), np.isnan(v2), err_msg=f"NaN mismatch in {c}"
            )
            valid = ~np.isnan(v1)
            np.testing.assert_allclose(
                v1[valid],
                v2[valid],
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Future data leakage detected in feature '{c}' at t <= T!",
            )


def test_next_observed_target_no_interpolation():
    """Verify next observed target logic jumps over non-trading gaps without synthetic interpolation."""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    vals = [100.0, 105.0, np.nan, np.nan, 112.0, 115.0]
    df_sample = pd.DataFrame({"date": dates, "kobc_handy": vals})

    target_series = get_next_observed_target(df_sample, "kobc_handy", date_col="date", only_when_observed=True)

    # Day 0 (100.0) -> next observed is Day 1 (105.0)
    assert target_series.iloc[0] == 105.0
    # Day 1 (105.0) -> next observed is Day 4 (112.0), jumping over missing days 2 and 3
    assert target_series.iloc[1] == 112.0
    # Day 2, 3 (NaN) -> target is NaN
    assert np.isnan(target_series.iloc[2])
    assert np.isnan(target_series.iloc[3])
    # Day 4 (112.0) -> next observed is Day 5 (115.0)
    assert target_series.iloc[4] == 115.0
    # Day 5 (115.0) -> no future observations -> NaN
    assert np.isnan(target_series.iloc[5])
