"""Unit tests for Phase 9 forecast uncertainty, charter decision engine, port constraints, and risk logic."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.decision.uncertainty import ResidualUncertaintyEstimator
from src.decision.charter import (
    CharterDecisionEngine,
    CharterDecisionRequest,
    CharterRecommendation,
)


@pytest.fixture
def mock_uncertainty_estimator() -> ResidualUncertaintyEstimator:
    """Fixture providing an estimator with calibrated synthetic residuals."""
    residuals = {
        "bdi_hsi": np.array([-2.5, -1.0, 0.0, 1.2, 2.8]),
        "bdi_si": np.array([-5.0, -2.0, 0.0, 2.5, 5.5]),
        "bdi_pi": np.array([-12.0, -4.0, 0.0, 4.5, 13.0]),
        "bdi_ci": np.array([-65.0, -20.0, 0.0, 25.0, 70.0]),
    }
    return ResidualUncertaintyEstimator(residuals_by_target=residuals)


@pytest.fixture
def decision_engine(mock_uncertainty_estimator) -> CharterDecisionEngine:
    """Fixture providing initialized charter decision engine."""
    return CharterDecisionEngine(
        ports_config_path="configs/ports.yaml",
        uncertainty_estimator=mock_uncertainty_estimator,
        threshold_cost_saving_pct=2.0,
        threshold_price_rise_pct=1.5,
    )


def test_uncertainty_interval_construction(mock_uncertainty_estimator):
    """Verify empirical prediction interval calculations and horizon expansion."""
    interval_h1 = mock_uncertainty_estimator.construct_prediction_interval(
        target_key="bdi_pi", point_forecast=1200.0, horizon_step=1
    )

    assert interval_h1["point_p50"] == 1200.0
    assert interval_h1["lower_p10"] < 1200.0
    assert interval_h1["upper_p90"] > 1200.0
    assert interval_h1["interval_width"] > 0

    # Horizon step 4 must have wider uncertainty interval than step 1 (sqrt(h) scaling)
    interval_h4 = mock_uncertainty_estimator.construct_prediction_interval(
        target_key="bdi_pi", point_forecast=1200.0, horizon_step=4
    )
    assert interval_h4["interval_width"] > interval_h1["interval_width"]


def test_vessel_feasibility_and_port_constraints(decision_engine):
    """Verify that port draft and navigational restrictions filter vessel classes properly."""
    # Haldia port: shallow draft (max 8.5m) -> must restrict to Handysize
    feasible_haldia, port_info = decision_engine.evaluate_vessel_feasibility(
        cargo_quantity_mt=30000.0, destination_port="haldia"
    )
    assert "Handysize" in feasible_haldia
    assert "Capesize" not in feasible_haldia
    assert "Panamax" not in feasible_haldia

    # Gangavaram port: deep water (max 19.5m) -> Capesize must be feasible
    feasible_gangavaram, _ = decision_engine.evaluate_vessel_feasibility(
        cargo_quantity_mt=170000.0, destination_port="gangavaram"
    )
    assert "Capesize" in feasible_gangavaram


def test_vessel_selection_optimality(decision_engine):
    """Verify selection of optimal vessel for parcel size."""
    # 75,000 MT coal parcel to Paradip -> Panamax
    vessels = ["Handysize", "Supramax", "Panamax"]
    best_vessel = decision_engine.select_optimal_vessel(vessels, cargo_quantity_mt=75000.0)
    assert best_vessel == "Panamax"

    # 30,000 MT parcel -> Handysize
    best_vessel_small = decision_engine.select_optimal_vessel(vessels, cargo_quantity_mt=30000.0)
    assert best_vessel_small == "Handysize"


def test_cost_calculation_breakdown(decision_engine):
    """Verify transparent voyage and port stay cost components."""
    cost = decision_engine.compute_voyage_cost(
        freight_index=1000.0,
        vessel_name="Panamax",
        cargo_quantity_mt=75000.0,
        voyage_duration_days=18.0,
        turnaround_days=3.0,
        demurrage_usd_per_day=15000.0,
        delay_days_waited=2,
        daily_holding_cost_usd=1500.0,
    )

    # Daily hire = 1000 * 12.5 = 12,500. Freight = 12,500 * 18 = 225,000.
    assert cost["freight_cost_usd"] == 225000.0
    # Port stay = 3 * 15,000 = 45,000
    assert cost["port_stay_cost_usd"] == 45000.0
    # Delay holding = 2 * 1,500 = 3,000
    assert cost["holding_delay_cost_usd"] == 3000.0
    assert cost["total_cost_usd"] == 273000.0
    assert cost["cost_per_mt_usd"] == round(273000.0 / 75000.0, 2)


def test_market_entry_decision_rules(decision_engine):
    """Verify rule-based market entry actions: CHARTER NOW vs WAIT vs FLEXIBLE."""
    req = CharterDecisionRequest(
        cargo_type="Coking Coal",
        cargo_quantity_mt=75000.0,
        destination_port="paradip",
        laycan_days_allowed=7,
        voyage_duration_days=18.0,
    )

    # 1. Rising market (+3% expected drift per day) -> CHARTER NOW
    rec_rising = decision_engine.recommend_charter(
        request=req,
        current_freight_index=1200.0,
        expected_drift_pct_per_day=3.0,
        rolling_volatility=0.01,
        gpr_spike_ratio=1.0,
    )
    assert rec_rising.charter_action == "CHARTER NOW"

    # 2. Falling market (-3% drift per day) -> WAIT
    rec_falling = decision_engine.recommend_charter(
        request=req,
        current_freight_index=1200.0,
        expected_drift_pct_per_day=-3.0,
        rolling_volatility=0.01,
        gpr_spike_ratio=1.0,
    )
    assert rec_falling.charter_action == "WAIT"
    assert rec_falling.optimal_entry_day > 0

    # 3. Flat market (0% drift) -> FLEXIBLE / MONITOR
    rec_flat = decision_engine.recommend_charter(
        request=req,
        current_freight_index=1200.0,
        expected_drift_pct_per_day=0.0,
        rolling_volatility=0.01,
        gpr_spike_ratio=1.0,
    )
    assert rec_flat.charter_action == "FLEXIBLE / MONITOR"


def test_risk_classification(decision_engine):
    """Verify risk classification flags (volatility, GPR, uncertainty)."""
    traj = [{"day_ahead": i, "interval_width": 50.0} for i in range(1, 8)]

    # Low risk
    level_low, _ = decision_engine.evaluate_risk(
        target_key="bdi_hsi",
        trajectory=traj,
        current_level=800.0,
        rolling_volatility=0.01,
        gpr_spike_ratio=1.0,
    )
    assert level_low in ["LOW", "MEDIUM"]

    # High risk with GPR spike and high volatility
    level_high, reasons_high = decision_engine.evaluate_risk(
        target_key="bdi_ci",
        trajectory=traj,
        current_level=500.0,
        rolling_volatility=0.05,
        gpr_spike_ratio=1.45,
        weather_alert=True,
    )
    assert level_high == "HIGH"
    assert any("Geopolitical" in r for r in reasons_high)


def test_deterministic_recommendation_output(decision_engine):
    """Verify deterministic outputs for fixed inputs."""
    req = CharterDecisionRequest(
        cargo_type="Iron Ore",
        cargo_quantity_mt=170000.0,
        destination_port="gangavaram",
    )
    rec1 = decision_engine.recommend_charter(req, current_freight_index=2000.0, expected_drift_pct_per_day=2.0)
    rec2 = decision_engine.recommend_charter(req, current_freight_index=2000.0, expected_drift_pct_per_day=2.0)

    assert rec1.charter_action == rec2.charter_action
    assert rec1.expected_cost_now_usd == rec2.expected_cost_now_usd
    assert rec1.recommended_vessel == rec2.recommended_vessel


def test_phase9_experiment_artifacts_exist():
    """Verify that Phase 9 scenario replay CSVs and figures exist on disk."""
    out_dir = Path("experiments/phase9")
    scenarios_path = out_dir / "scenarios.csv"
    recs_path = out_dir / "recommendations.csv"
    cost_path = out_dir / "cost_comparison.csv"
    fig_dir = out_dir / "figures"

    assert scenarios_path.exists()
    assert recs_path.exists()
    assert cost_path.exists()
    assert (fig_dir / "01_decision_actions_and_risk.png").exists()
    assert (fig_dir / "02_charter_cost_comparison.png").exists()
