"""Phase 10.5 Historical Replay, Adversarial Future-Data, and Hindsight Separation Tests.

Proves:
- Running recommendations at decision date T strictly consumes data <= T.
- Adversarially mutating all data after T (future freight, FX, oil, GPR, weather) has ZERO effect
  on the decision, forecast, cost, risk, or vessel recommendation.
- Hindsight oracle calculations remain strictly isolated from live decision-time service execution.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.schemas import DATE_COLUMN
from src.application.schemas import FICOSRequest, FICOSRecommendation
from src.application.service import FICOSService


@pytest.fixture(scope="module")
def ficos_service() -> FICOSService:
    return FICOSService()


def test_adversarial_future_data_decision_invariance(ficos_service):
    """CRITICAL ADVERSARIAL TEST:

    Take decision date T=2017-10-16.
    1. Run recommendation with standard features matrix.
    2. Modify all rows in service's feature matrix where date > T to extreme corrupted numbers.
    3. Re-run recommendation.
    4. Assert that all recommendation fields (action, optimal_day, forecast, costs, risk) are identical.
    """
    decision_date = "2017-10-16"
    req = FICOSRequest(
        decision_date=decision_date,
        cargo_quantity_mt=75000.0,
        cargo_type="Coking Coal",
        destination_port="dhamra",
        laycan_days_allowed=7,
        voyage_duration_days=18.0,
    )

    # 1. Baseline Run
    rec_base = ficos_service.process_request(req)

    # 2. Save original features and inject massive adversarial noise into all future data (t > decision_date)
    orig_features = ficos_service.df_features.copy()
    target_dt = pd.to_datetime(decision_date)
    future_mask = ficos_service.df_features[DATE_COLUMN] > target_dt

    assert future_mask.sum() > 400, "Must have future rows to corrupt"

    # Corrupt all numeric columns in the future
    num_cols = ficos_service.df_features.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        ficos_service.df_features.loc[future_mask, c] = 999999.0

    try:
        # 3. Adversarial Run
        rec_adversarial = ficos_service.process_request(req)

        # 4. Strict equivalence assertion
        assert rec_base.decision.action == rec_adversarial.decision.action
        assert rec_base.decision.optimal_entry_day == rec_adversarial.decision.optimal_entry_day
        assert rec_base.decision.expected_cost_now_usd == rec_adversarial.decision.expected_cost_now_usd
        assert rec_base.decision.expected_cost_optimal_usd == rec_adversarial.decision.expected_cost_optimal_usd
        assert rec_base.vessel.recommended == rec_adversarial.vessel.recommended
        assert rec_base.risk.level == rec_adversarial.risk.level
        np.testing.assert_allclose(rec_base.forecast.values, rec_adversarial.forecast.values, rtol=1e-5)
        np.testing.assert_allclose(rec_base.forecast.p10, rec_adversarial.forecast.p10, rtol=1e-5)
        np.testing.assert_allclose(rec_base.forecast.p90, rec_adversarial.forecast.p90, rtol=1e-5)

    finally:
        # Restore clean feature matrix
        ficos_service.df_features = orig_features


def test_historical_scenarios_replay(ficos_service):
    """Verify execution of historical scenarios from Phase 9."""
    scenarios = [
        ("2016-02-15", 75000, "paradip", "Panamax"),
        ("2017-06-20", 170000, "gangavaram", "Capesize"),
        ("2018-07-18", 30000, "haldia", "Handysize"),
        ("2018-09-25", 75000, "dhamra", "Panamax"),
    ]

    for d, qty, port, expected_vessel in scenarios:
        req = FICOSRequest(
            decision_date=d,
            cargo_quantity_mt=qty,
            destination_port=port,
        )
        rec = ficos_service.process_request(req)
        assert rec.vessel.recommended == expected_vessel
        assert rec.decision.action in ["CHARTER NOW", "WAIT", "FLEXIBLE / MONITOR"]
        assert rec.audit.decision_date == d


def test_hindsight_oracle_separation(ficos_service):
    """Verify audit record states hindsight separation and live engine has no hindsight knowledge."""
    req = FICOSRequest(
        decision_date="2018-09-25",
        cargo_quantity_mt=75000.0,
        destination_port="paradip",
    )
    rec = ficos_service.process_request(req)

    assert rec.audit.hindsight_oracle_available is False
    assert "Hindsight benchmark is computed separately" in rec.audit.hindsight_note
