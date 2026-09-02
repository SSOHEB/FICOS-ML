"""Phase 10.5 Cost Model Arithmetic and Risk Engine Boundary Tests for FICOS.

Verifies:
- Exact arithmetic validation of voyage cost breakdown:
    Total = Freight Cost + Port Stay Cost + Delay Holding Cost
- Multipliers per vessel class (Handysize=15.0, Supramax=14.0, Panamax=12.5, Capesize=10.0).
- Hand-calculated expected-value test cases across multiple delays and freight levels.
- Risk score transitions and exact boundary condition evaluation for LOW, MEDIUM, HIGH.
"""

import pytest

from src.decision.charter import CharterDecisionEngine
from src.decision.uncertainty import ResidualUncertaintyEstimator


@pytest.fixture(scope="module")
def decision_engine() -> CharterDecisionEngine:
    return CharterDecisionEngine(
        ports_config_path="configs/ports.yaml",
        uncertainty_estimator=ResidualUncertaintyEstimator(),
    )


@pytest.mark.parametrize(
    "vessel_name,multiplier,index_level,voyage_days,turnaround,demurrage,delay_days,holding_cost,cargo_mt",
    [
        ("Handysize", 15.0, 800.0, 10.0, 4.0, 10000.0, 0, 1000.0, 30000.0),
        ("Supramax", 14.0, 1100.0, 14.0, 3.5, 12000.0, 1, 1200.0, 55000.0),
        ("Panamax", 12.5, 1400.0, 18.0, 3.0, 15000.0, 3, 1500.0, 75000.0),
        ("Capesize", 10.0, 2500.0, 28.0, 2.5, 22000.0, 5, 2000.0, 170000.0),
    ],
)
def test_cost_equation_exact_hand_calculated(
    decision_engine,
    vessel_name,
    multiplier,
    index_level,
    voyage_days,
    turnaround,
    demurrage,
    delay_days,
    holding_cost,
    cargo_mt,
):
    """Verify total cost matches hand-calculated arithmetic exactly."""
    cost = decision_engine.compute_voyage_cost(
        freight_index=index_level,
        vessel_name=vessel_name,
        cargo_quantity_mt=cargo_mt,
        voyage_duration_days=voyage_days,
        turnaround_days=turnaround,
        demurrage_usd_per_day=demurrage,
        delay_days_waited=delay_days,
        daily_holding_cost_usd=holding_cost,
    )

    expected_freight = index_level * multiplier * voyage_days
    expected_port = turnaround * demurrage
    expected_holding = delay_days * holding_cost
    expected_total = expected_freight + expected_port + expected_holding
    expected_per_mt = round(expected_total / cargo_mt, 2)

    assert cost["freight_cost_usd"] == round(expected_freight, 2)
    assert cost["port_stay_cost_usd"] == round(expected_port, 2)
    assert cost["holding_delay_cost_usd"] == round(expected_holding, 2)
    assert cost["total_cost_usd"] == round(expected_total, 2)
    assert cost["cost_per_mt_usd"] == expected_per_mt


def test_risk_classification_boundaries(decision_engine):
    """Verify risk classification score logic and boundary transitions."""
    traj_narrow = [{"day_ahead": i, "interval_width": 10.0} for i in range(1, 8)]
    traj_wide = [{"day_ahead": i, "interval_width": 300.0} for i in range(1, 8)]

    # 1. Baseline calm -> score 0 -> LOW
    lvl, _ = decision_engine.evaluate_risk(
        target_key="bdi_hsi",
        trajectory=traj_narrow,
        current_level=1000.0,
        rolling_volatility=0.01,
        gpr_spike_ratio=1.0,
        weather_alert=False,
    )
    assert lvl == "LOW"

    # 2. Moderate GPR (1.20x) alone -> score 1 -> MEDIUM
    lvl_med, _ = decision_engine.evaluate_risk(
        target_key="bdi_hsi",
        trajectory=traj_narrow,
        current_level=1000.0,
        rolling_volatility=0.01,
        gpr_spike_ratio=1.20,
        weather_alert=False,
    )
    assert lvl_med == "MEDIUM"

    # 3. High Volatility (4.0%) + High GPR (1.35x) -> score 4 -> HIGH
    lvl_high, reasons = decision_engine.evaluate_risk(
        target_key="bdi_hsi",
        trajectory=traj_narrow,
        current_level=1000.0,
        rolling_volatility=0.04,
        gpr_spike_ratio=1.35,
        weather_alert=False,
    )
    assert lvl_high == "HIGH"
    assert len(reasons) >= 2

    # 4. Capesize structural (+1) + Weather (+1) -> score 2 -> MEDIUM
    lvl_cape, _ = decision_engine.evaluate_risk(
        target_key="bdi_ci",
        trajectory=traj_narrow,
        current_level=1000.0,
        rolling_volatility=0.01,
        gpr_spike_ratio=1.0,
        weather_alert=True,
    )
    assert lvl_cape == "MEDIUM"
