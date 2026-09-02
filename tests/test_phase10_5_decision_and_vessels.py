"""Phase 10.5 Decision Rule Engine and Vessel/Port Feasibility Tests for FICOS.

Verifies:
- All 3 decision actions (CHARTER NOW, WAIT, FLEXIBLE / MONITOR) trigger under controlled conditions.
- Vessel feasibility across all 4 vessel classes and 7 East Coast Indian ports.
- Navigational draft and DWT constraint filtering.
- Specific validation for Haldia shallow-draft restrictions and Gangavaram/Dhamra deepwater Capesize operations.
- Vessel capacity selection optimality.
"""

import pytest

from src.decision.charter import (
    CharterDecisionEngine,
    CharterDecisionRequest,
    CharterRecommendation,
)
from src.decision.uncertainty import ResidualUncertaintyEstimator


@pytest.fixture(scope="module")
def decision_engine() -> CharterDecisionEngine:
    estimator = ResidualUncertaintyEstimator.from_walk_forward_predictions("experiments/phase8/predictions.csv")
    return CharterDecisionEngine(
        ports_config_path="configs/ports.yaml",
        uncertainty_estimator=estimator,
        threshold_cost_saving_pct=2.0,
        threshold_price_rise_pct=1.5,
    )


def test_decision_rising_freight_triggers_charter_now(decision_engine):
    """Verify strong expected rise triggers CHARTER NOW."""
    req = CharterDecisionRequest(
        cargo_type="Coking Coal",
        cargo_quantity_mt=75000.0,
        destination_port="paradip",
        laycan_days_allowed=7,
        voyage_duration_days=18.0,
    )
    rec = decision_engine.recommend_charter(
        request=req,
        current_freight_index=1200.0,
        expected_drift_pct_per_day=3.0,
        rolling_volatility=0.015,
        gpr_spike_ratio=1.0,
        weather_alert=False,
    )
    assert rec.charter_action == "CHARTER NOW"
    assert "Chartering now avoids higher future hire costs" in rec.reasons[0]


def test_decision_falling_freight_triggers_wait(decision_engine):
    """Verify projected decline yielding > 2% cost savings triggers WAIT."""
    req = CharterDecisionRequest(
        cargo_type="Coking Coal",
        cargo_quantity_mt=75000.0,
        destination_port="paradip",
        laycan_days_allowed=7,
        voyage_duration_days=18.0,
    )
    rec = decision_engine.recommend_charter(
        request=req,
        current_freight_index=1200.0,
        expected_drift_pct_per_day=-3.0,
        rolling_volatility=0.015,
        gpr_spike_ratio=1.0,
        weather_alert=False,
    )
    assert rec.charter_action == "WAIT"
    assert rec.optimal_entry_day > 0
    assert rec.estimated_savings_pct >= 2.0


def test_decision_flat_market_triggers_flexible_monitor(decision_engine):
    """Verify flat / noise level freight movement triggers FLEXIBLE / MONITOR."""
    req = CharterDecisionRequest(
        cargo_type="Coking Coal",
        cargo_quantity_mt=75000.0,
        destination_port="paradip",
        laycan_days_allowed=7,
        voyage_duration_days=18.0,
    )
    rec = decision_engine.recommend_charter(
        request=req,
        current_freight_index=1200.0,
        expected_drift_pct_per_day=0.1,  # Under 1.5% threshold over 7 days
        rolling_volatility=0.015,
        gpr_spike_ratio=1.0,
        weather_alert=False,
    )
    assert rec.charter_action == "FLEXIBLE / MONITOR"


def test_decision_high_risk_triggers_spot_lockin(decision_engine):
    """Verify high risk environment with non-falling freight triggers spot lock-in (CHARTER NOW)."""
    req = CharterDecisionRequest(
        cargo_type="Iron Ore",
        cargo_quantity_mt=170000.0,
        destination_port="gangavaram",
        laycan_days_allowed=7,
        voyage_duration_days=28.0,
    )
    rec = decision_engine.recommend_charter(
        request=req,
        current_freight_index=2000.0,
        expected_drift_pct_per_day=0.0,
        rolling_volatility=0.045,  # High volatility
        gpr_spike_ratio=1.40,     # GPR spike
        weather_alert=True,        # Weather alert
    )
    assert rec.risk_level == "HIGH"
    assert rec.charter_action == "CHARTER NOW"
    assert "Locking in current spot rate is recommended" in rec.reasons[0]


def test_all_seven_ports_feasibility(decision_engine):
    """Verify feasibility rules across all 7 East Coast Indian ports."""
    # 1. Haldia: max draft 8.5m -> only Handysize
    feas_haldia, _ = decision_engine.evaluate_vessel_feasibility(30000, "haldia")
    assert feas_haldia == ["Handysize"]

    # 2. Gopalpur: max draft 12.5m -> Handysize, Supramax
    feas_gopalpur, _ = decision_engine.evaluate_vessel_feasibility(50000, "gopalpur")
    assert set(feas_gopalpur) == {"Handysize", "Supramax"}

    # 3. Paradip: max draft 14.5m -> Handysize, Supramax, Panamax (no Capesize)
    feas_paradip, _ = decision_engine.evaluate_vessel_feasibility(75000, "paradip")
    assert set(feas_paradip) == {"Handysize", "Supramax", "Panamax"}
    assert "Capesize" not in feas_paradip

    # 4. Vizag: max draft 16.5m, max DWT 150k -> Handysize, Supramax, Panamax (Capesize draft 18.2m > 16.5m is filtered)
    feas_vizag, _ = decision_engine.evaluate_vessel_feasibility(75000, "vizag")
    assert set(feas_vizag) == {"Handysize", "Supramax", "Panamax"}
    assert "Capesize" not in feas_vizag

    # 5. Gangavaram: deepwater max draft 19.5m, max DWT 200k -> Capesize is fully feasible!
    feas_gangavaram, _ = decision_engine.evaluate_vessel_feasibility(175000, "gangavaram")
    assert "Capesize" in feas_gangavaram
    assert set(feas_gangavaram) == {"Handysize", "Supramax", "Panamax", "Capesize"}

    # 6. Dhamra: max draft 18.0m -> Handysize, Supramax, Panamax feasible
    feas_dhamra, _ = decision_engine.evaluate_vessel_feasibility(75000, "dhamra")
    assert set(feas_dhamra) == {"Handysize", "Supramax", "Panamax"}

    # 7. Sagar-Sandheads: max draft 16.0m -> Handysize, Supramax, Panamax feasible
    feas_sagar1, _ = decision_engine.evaluate_vessel_feasibility(75000, "sagar_sandheads")
    feas_sagar2, _ = decision_engine.evaluate_vessel_feasibility(75000, "sagar-sandheads")
    assert feas_sagar1 == feas_sagar2
    assert "Panamax" in feas_sagar1


def test_vessel_capacity_selection_optimality(decision_engine):
    """Verify closest-capacity parcel matching."""
    vessels = ["Handysize", "Supramax", "Panamax", "Capesize"]
    assert decision_engine.select_optimal_vessel(vessels, 28000) == "Handysize"
    assert decision_engine.select_optimal_vessel(vessels, 52000) == "Supramax"
    assert decision_engine.select_optimal_vessel(vessels, 78000) == "Panamax"
    assert decision_engine.select_optimal_vessel(vessels, 165000) == "Capesize"
