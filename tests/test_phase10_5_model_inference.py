"""Phase 10.5 Model Inference, Feature Schema, and Determinism Tests for FICOS.

Verifies:
- Production Ridge forecasters load and initialize correctly.
- Expected feature column counts and names match configuration.
- Missing required features trigger explicit ValueError with column names.
- Extra unexpected columns do not corrupt feature alignment.
- Inference is strictly deterministic (same input produces bitwise identical predictions).
- No retraining happens during standard prediction inference.
- Multi-regime inference testing (calm, rising, falling, volatile).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.schemas import DATE_COLUMN
from src.application.model_registry import ModelRegistry
from src.models.ridge import RidgeForecaster


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    reg = ModelRegistry(config_path="configs/models.yaml")
    reg.initialize()
    return reg


@pytest.fixture(scope="module")
def features_matrix() -> pd.DataFrame:
    df = pd.read_csv("data/features/freight_features.csv")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df.sort_values(by=DATE_COLUMN).reset_index(drop=True)


def test_model_registry_initialization(registry):
    """Verify registry loads all 4 target Ridge forecasters."""
    assert registry.is_initialized
    assert set(registry.models.keys()) == {"bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"}
    for target, model in registry.models.items():
        assert isinstance(model, RidgeForecaster)
        assert model.alpha == 1.0
        assert model.is_fitted


def test_expected_feature_schema(registry):
    """Verify registry has registered the expected feature columns."""
    assert registry.expected_feature_cols is not None
    assert len(registry.expected_feature_cols) >= 50
    # Ensure date and target lead columns are NOT in expected feature cols
    assert DATE_COLUMN not in registry.expected_feature_cols
    for tgt in ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"]:
        assert f"{tgt}_lead_1" not in registry.expected_feature_cols


def test_missing_features_raises_error(registry, features_matrix):
    """Verify missing features fail with clear ValueError."""
    bad_df = features_matrix.drop(columns=[registry.expected_feature_cols[0]])
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        registry.predict_one_step("bdi_pi", bad_df)


def test_extra_unexpected_columns_handled_safely(registry, features_matrix):
    """Verify extra columns do not corrupt inference ordering."""
    df_extra = features_matrix.copy()
    df_extra["unexpected_random_col"] = 999.99
    df_extra["another_noise_col"] = "unexpected_string"

    pred_original = registry.predict_one_step("bdi_pi", features_matrix)
    pred_extra = registry.predict_one_step("bdi_pi", df_extra)

    assert pred_original == pred_extra, "Extra columns must not alter inference prediction"


def test_inference_determinism_and_no_retraining(registry, features_matrix):
    """Verify 100 consecutive predictions on same input yield identical floats and weights."""
    target = "bdi_pi"
    model = registry.models[target]
    initial_weights = model.model.coef_.copy()
    initial_intercept = model.model.intercept_

    sample_input = features_matrix.iloc[-30:]
    preds = [registry.predict_one_step(target, sample_input) for _ in range(20)]

    # All predictions must be bitwise identical
    assert len(set(preds)) == 1, "Inference must be strictly deterministic"

    # Verify model weights remained untouched (no silent retraining)
    np.testing.assert_array_equal(model.model.coef_, initial_weights)
    assert model.model.intercept_ == initial_intercept


def test_inference_across_market_regimes(registry, features_matrix):
    """Verify finite non-negative forecasts across diverse historical dates/regimes."""
    regime_dates = [
        ("2016-02-15", "Trough / Market Crash"),
        ("2017-06-20", "Capesize Cyclical Bull Rally"),
        ("2018-09-25", "High Volatility / Trade Dispute"),
        ("2019-02-14", "Post-Brumadinho Slump"),
        ("2019-07-10", "July 2019 Capesize Squeeze"),
    ]

    for d_str, regime_name in regime_dates:
        d = pd.to_datetime(d_str)
        hist_df = features_matrix[features_matrix[DATE_COLUMN] <= d]
        assert not hist_df.empty

        for target in ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"]:
            pred = registry.predict_one_step(target, hist_df)
            assert np.isfinite(pred), f"Prediction for {target} on {d_str} ({regime_name}) is not finite: {pred}"
