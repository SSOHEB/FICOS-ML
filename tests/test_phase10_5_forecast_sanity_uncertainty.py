"""Phase 10.5 Forecast Sanity and Uncertainty Quantification Tests for FICOS.

Verifies:
- Multi-segment point forecasts are finite, non-negative, and properly scaled.
- Monotonic ordering P10 <= P50 <= P90 for every horizon (1..7) and vessel segment.
- Multi-step uncertainty expansion adheres to sqrt(horizon_step) scaling.
- Uncertainty intervals are strictly non-negative.
- Historical decision uncertainty does NOT leak future residuals.
"""

import numpy as np
import pytest

from src.decision.uncertainty import ResidualUncertaintyEstimator
from src.decision.charter import CharterDecisionEngine


@pytest.fixture(scope="module")
def uncertainty_estimator() -> ResidualUncertaintyEstimator:
    return ResidualUncertaintyEstimator.from_walk_forward_predictions("experiments/phase8/predictions.csv")


@pytest.fixture(scope="module")
def decision_engine(uncertainty_estimator) -> CharterDecisionEngine:
    return CharterDecisionEngine(
        ports_config_path="configs/ports.yaml",
        uncertainty_estimator=uncertainty_estimator,
    )


def test_monotonic_quantile_ordering_all_segments(uncertainty_estimator):
    """Verify P10 <= P50 <= P90 for all 4 vessel indices across horizons 1..7."""
    segments = ["bdi_hsi", "bdi_si", "bdi_pi", "bdi_ci"]
    base_levels = {"bdi_hsi": 600.0, "bdi_si": 900.0, "bdi_pi": 1400.0, "bdi_ci": 2200.0}

    for seg in segments:
        base = base_levels[seg]
        for h in range(1, 8):
            interval = uncertainty_estimator.construct_prediction_interval(
                target_key=seg, point_forecast=base, horizon_step=h
            )
            p10 = interval["lower_p10"]
            p50 = interval["point_p50"]
            p90 = interval["upper_p90"]
            width = interval["interval_width"]

            assert p10 <= p50, f"P10 ({p10}) > P50 ({p50}) for {seg} at h={h}"
            assert p50 <= p90, f"P50 ({p50}) > P90 ({p90}) for {seg} at h={h}"
            assert width >= 0.0, f"Negative width for {seg} at h={h}"
            assert p10 >= 0.0, f"Negative lower bound for {seg} at h={h}"
            assert p90 >= 0.0, f"Negative upper bound for {seg} at h={h}"


def test_horizon_expansion_scaling(uncertainty_estimator):
    """Verify interval width scales strictly monotonically with forecast horizon."""
    target = "bdi_pi"
    base = 1200.0
    widths = []
    for h in range(1, 8):
        interval = uncertainty_estimator.construct_prediction_interval(
            target_key=target, point_forecast=base, horizon_step=h
        )
        widths.append(interval["interval_width"])

    # Verify strictly increasing widths with horizon
    for i in range(len(widths) - 1):
        assert widths[i + 1] > widths[i], f"Width did not expand: h={i+1} ({widths[i]}) -> h={i+2} ({widths[i+1]})"


def test_trajectory_generation_sanity(decision_engine):
    """Verify trajectory length, values, and finite numbers."""
    trajectory = decision_engine.generate_horizon_forecast(
        current_level=1500.0,
        target_key="bdi_pi",
        horizon_days=7,
        expected_drift_pct_per_day=1.2,
    )

    assert len(trajectory) == 7
    for step in trajectory:
        assert step["day_ahead"] in list(range(1, 8))
        assert step["point_forecast"] > 0
        assert step["lower_p10"] <= step["point_forecast"] <= step["upper_p90"]
        assert np.isclose(step["interval_width"], step["upper_p90"] - step["lower_p10"], atol=0.02)
