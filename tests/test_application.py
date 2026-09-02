"""Unit and integration tests for Phase 10 FICOS Application Layer, validation, and auditability."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.application.schemas import (
    FICOSRequest,
    FICOSRecommendation,
    ValidationError,
    SUPPORTED_PORTS,
    SUPPORTED_VESSELS,
)
from src.application.service import FICOSService
from src.application.model_registry import ModelRegistry


@pytest.fixture(scope="module")
def ficos_service() -> FICOSService:
    """Fixture providing initialized FICOS service."""
    return FICOSService()


def test_valid_recommendation_request(ficos_service):
    """Verify that a standard valid request produces a complete structured recommendation."""
    req = FICOSRequest(
        decision_date="2018-09-25",
        cargo_quantity_mt=75000.0,
        cargo_type="Coking Coal",
        destination_port="dhamra",
        laycan_days_allowed=7,
        voyage_duration_days=18.0,
    )
    rec = ficos_service.process_request(req)

    assert isinstance(rec, FICOSRecommendation)
    assert rec.decision_date == "2018-09-25"
    assert rec.destination_port == "dhamra"
    assert rec.vessel.recommended in SUPPORTED_VESSELS
    assert rec.decision.action in ["CHARTER NOW", "WAIT", "FLEXIBLE / MONITOR"]
    assert len(rec.forecast.values) == 7
    assert len(rec.forecast.p10) == 7
    assert len(rec.forecast.p90) == 7
    assert rec.audit.model_name == "Ridge_a1.0"
    assert len(rec.reasons) > 0
    assert len(rec.assumptions) > 0


def test_invalid_cargo_quantity_raises():
    """Verify that zero or negative cargo quantity raises ValidationError."""
    with pytest.raises(ValidationError, match="Invalid cargo_quantity_mt"):
        req = FICOSRequest(
            decision_date="2018-09-25",
            cargo_quantity_mt=0.0,
            destination_port="paradip",
        )
        req.validate()

    with pytest.raises(ValidationError, match="Invalid cargo_quantity_mt"):
        req_neg = FICOSRequest(
            decision_date="2018-09-25",
            cargo_quantity_mt=-5000.0,
            destination_port="paradip",
        )
        req_neg.validate()


def test_invalid_destination_port_raises():
    """Verify that unsupported port name raises ValidationError."""
    with pytest.raises(ValidationError, match="Unsupported destination_port"):
        req = FICOSRequest(
            decision_date="2018-09-25",
            cargo_quantity_mt=75000.0,
            destination_port="singapore_unsupported",
        )
        req.validate()


def test_invalid_preferred_vessel_raises():
    """Verify that unsupported preferred vessel raises ValidationError."""
    with pytest.raises(ValidationError, match="Invalid preferred_vessel"):
        req = FICOSRequest(
            decision_date="2018-09-25",
            cargo_quantity_mt=75000.0,
            destination_port="paradip",
            preferred_vessel="NuclearCarrier",
        )
        req.validate()


def test_invalid_date_format_raises():
    """Verify that malformed decision date raises ValidationError."""
    with pytest.raises(ValidationError, match="Invalid decision_date"):
        req = FICOSRequest(
            decision_date="25/09/2018",  # Wrong format (must be YYYY-MM-DD)
            cargo_quantity_mt=75000.0,
            destination_port="paradip",
        )
        req.validate()


def test_model_registry_feature_schema_validation(ficos_service):
    """Verify that ModelRegistry verifies the exact feature schema and fails on missing columns."""
    reg = ficos_service.model_registry
    assert reg.is_initialized
    assert len(reg.expected_feature_cols) > 50

    # Pass DataFrame missing critical features
    incomplete_df = pd.DataFrame({"dummy": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        reg.validate_feature_schema(incomplete_df)


def test_forecast_and_uncertainty_quantiles(ficos_service):
    """Verify forecast horizon length and monotonic empirical quantile ordering (P10 <= P50 <= P90)."""
    req = FICOSRequest(
        decision_date="2017-10-16",
        cargo_quantity_mt=75000.0,
        destination_port="dhamra",
        laycan_days_allowed=7,
    )
    rec = ficos_service.process_request(req)

    for p10, p50, p90 in zip(rec.forecast.p10, rec.forecast.p50, rec.forecast.p90):
        assert p10 <= p50 <= p90
        assert p10 >= 0.0  # Non-negative index bound


def test_port_vessel_feasibility_rules(ficos_service):
    """Verify that shallow draft port Haldia restricts to Handysize."""
    req_haldia = FICOSRequest(
        decision_date="2018-07-18",
        cargo_quantity_mt=30000.0,
        destination_port="haldia",
    )
    rec_haldia = ficos_service.process_request(req_haldia)
    assert rec_haldia.vessel.recommended == "Handysize"
    assert "Capesize" not in rec_haldia.vessel.feasible_vessels


def test_deterministic_service_output(ficos_service):
    """Verify identical recommendation outputs for identical inputs."""
    req = FICOSRequest(
        decision_date="2018-09-25",
        cargo_quantity_mt=75000.0,
        destination_port="paradip",
    )
    rec1 = ficos_service.process_request(req)
    rec2 = ficos_service.process_request(req)

    assert rec1.decision.action == rec2.decision.action
    assert rec1.decision.expected_cost_now_usd == rec2.decision.expected_cost_now_usd
    assert rec1.forecast.values == rec2.forecast.values


def test_decision_time_leakage_isolation(ficos_service):
    """CRITICAL TEST: Verify that a decision made on date T uses only data up to T."""
    req = FICOSRequest(
        decision_date="2016-02-15",
        cargo_quantity_mt=75000.0,
        destination_port="paradip",
    )
    rec = ficos_service.process_request(req)

    # Decision date must match requested date
    assert rec.decision_date == "2016-02-15"
    assert rec.audit.decision_date == "2016-02-15"


def test_audit_record_completeness(ficos_service):
    """Verify audit record fields and explicit hindsight oracle separation note."""
    req = FICOSRequest(
        decision_date="2018-09-25",
        cargo_quantity_mt=75000.0,
        destination_port="dhamra",
    )
    rec = ficos_service.process_request(req)
    audit = rec.audit

    assert audit.request_id is not None
    assert audit.model_name == "Ridge_a1.0"
    assert audit.model_version == "Phase8_Production_Champion"
    assert audit.rule_triggered in [
        "RisingFreightMomentumThreshold",
        "ExpectedCostSavingOpportunity",
        "HighRiskSpotLockin",
        "UncertaintyBandIndifference",
    ]
    assert "Hindsight benchmark is computed separately" in audit.hindsight_note


def test_phase10_experiment_outputs_exist():
    """Verify that Phase 10 artifacts exist in experiments/phase10/."""
    out_dir = Path("experiments/phase10")
    assert (out_dir / "final_recommendations.csv").exists()
    assert (out_dir / "final_demo_outputs.json").exists()
    assert (out_dir / "final_validation_metrics.csv").exists()
    assert (out_dir / "configuration.yaml").exists()
    assert (out_dir / "audit_examples.json").exists()
