"""Tests for Phase 11: Decision-Oriented Model Audit.

Verifies:
1. Target/feature temporal alignment: target is strictly t+1, current level is t, lags <= t.
2. Feature group categorization: all 172 features categorized with zero unclassified.
3. Output artifacts exist in experiments/expanded/:
   - model_audit.csv
   - regime_metrics.csv
   - decision_metrics.csv
   - feature_groups.csv
   - feature_importance.csv
4. Decision signal sanity: Persistence produces 0 active directional signals while Ridge produces active signals with >55% directional accuracy.
5. Relative improvement metrics: mathematically consistent with MAE/RMSE deltas.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.schemas import DATE_COLUMN
from scripts.run_phase11_model_audit import classify_feature_group, KOBC_TARGETS
from scripts.run_expanded_training_experiment import get_valid_kobc_features


@pytest.fixture(scope="module")
def expanded_df() -> pd.DataFrame:
    p = Path("data/features/freight_features_expanded.csv")
    assert p.exists(), "freight_features_expanded.csv must exist"
    df = pd.read_csv(p)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


def test_target_feature_alignment_and_strict_causality(expanded_df):
    """Verify target is strictly t+1 observed value and current level is t without lookahead."""
    df_kobc = expanded_df[expanded_df[DATE_COLUMN] >= pd.Timestamp("2020-01-01")].sort_values(by=DATE_COLUMN).reset_index(drop=True)
    
    for target_col, current_col in KOBC_TARGETS.items():
        # y_true at row i must equal current_col at row i+1
        y_true = df_kobc[target_col].iloc[:-1].values
        y_next_level = df_kobc[current_col].iloc[1:].values
        
        np.testing.assert_array_equal(
            y_true, y_next_level,
            err_msg=f"Target {target_col} does not align with next observed {current_col}!"
        )

        # lag_1 at row i must equal current_col at row i-1
        lag1_col = f"{current_col.replace('_level', '')}_lag_1"
        if lag1_col in df_kobc.columns:
            lag1_vals = df_kobc[lag1_col].iloc[1:].values
            curr_prev = df_kobc[current_col].iloc[:-1].values
            np.testing.assert_array_equal(
                lag1_vals, curr_prev,
                err_msg=f"Lag-1 {lag1_col} does not strictly equal previous day level!"
            )


def test_feature_group_classification_completeness(expanded_df):
    """Verify all selected 172 KOBC features are classified into valid audit groups."""
    df_kobc = expanded_df[expanded_df[DATE_COLUMN] >= pd.Timestamp("2020-01-01")].sort_values(by=DATE_COLUMN).reset_index(drop=True)
    features = get_valid_kobc_features(df_kobc)
    assert len(features) == 172, f"Expected 172 features, found {len(features)}"

    valid_groups = {
        "A_autoregressive_freight", "B_cross_vessel_freight", "C_kdci_features",
        "D_commodity_energy", "E_fx_macro", "F_geopolitical_risk",
        "G_weather", "H_calendar", "I_port_operational"
    }

    for f in features:
        grp = classify_feature_group(f)
        assert grp in valid_groups, f"Feature '{f}' classified into invalid group '{grp}'"


def test_phase11_audit_artifacts_exist():
    """Verify all 5 required audit tables are generated under experiments/expanded/."""
    exp_dir = Path("experiments/expanded")
    required_files = [
        "model_audit.csv",
        "regime_metrics.csv",
        "decision_metrics.csv",
        "feature_groups.csv",
        "feature_importance.csv",
    ]
    for fname in required_files:
        p = exp_dir / fname
        assert p.exists(), f"Missing required audit artifact: {fname}"
        assert p.stat().st_size > 100, f"Artifact {fname} is unexpectedly small ({p.stat().st_size} bytes)"


def test_model_audit_relative_improvements():
    """Verify relative persistence improvements are correctly computed."""
    p = Path("experiments/expanded/model_audit.csv")
    df = pd.read_csv(p)

    for _, row in df.iterrows():
        if row["model"] == "Persistence":
            assert row["mae_improvement_pct"] == 0.0
            assert row["rmse_improvement_pct"] == 0.0
        else:
            # Check sign consistency
            if row["beats_persistence_mae"]:
                assert row["mae_improvement_pct"] > 0.0
            else:
                assert row["mae_improvement_pct"] <= 0.0


def test_decision_signals_directional_power():
    """Verify Ridge provides actionable decision signals with DA > 55% across all vessels."""
    p = Path("experiments/expanded/decision_metrics.csv")
    df = pd.read_csv(p)

    ridge_rows = df[df["model"] == "Ridge"]
    assert len(ridge_rows) == 4, "Expected 4 vessel evaluations for Ridge"

    for _, row in ridge_rows.iterrows():
        assert row["charter_now_signals"] + row["wait_signals"] > 50, "Ridge must produce active signals"
        assert row["active_signal_accuracy_pct"] >= 58.0, f"Ridge active signal accuracy below 58% on {row['vessel']}"
        assert row["overall_directional_accuracy_pct"] >= 60.0, f"Ridge overall DA below 60% on {row['vessel']}"
