"""FICOS core orchestration service coordinating forecasting, decision logic, and auditability."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import uuid
import yaml
import numpy as np
import pandas as pd

from src.data.schemas import DATE_COLUMN
from src.application.schemas import (
    FICOSRequest,
    FICOSRecommendation,
    ForecastOutput,
    VesselOutput,
    DecisionOutput,
    RiskOutput,
    AuditRecord,
    ValidationError,
)
from src.application.model_registry import ModelRegistry
from src.decision.uncertainty import ResidualUncertaintyEstimator
from src.decision.charter import CharterDecisionEngine, CharterDecisionRequest


class FICOSService:
    """Production application service for FICOS chartering decision support."""

    def __init__(
        self,
        models_config_path: Union[str, Path] = "configs/models.yaml",
        ports_config_path: Union[str, Path] = "configs/ports.yaml",
        features_path: Union[str, Path] = "data/features/freight_features.csv",
    ):
        self.models_config_path = Path(models_config_path)
        self.ports_config_path = Path(ports_config_path)
        self.features_path = Path(features_path)

        # 1. Initialize Model Registry
        self.model_registry = ModelRegistry(config_path=self.models_config_path)
        self.model_registry.initialize()

        # 2. Initialize Uncertainty Estimator
        self.uncertainty_estimator = ResidualUncertaintyEstimator.from_walk_forward_predictions(
            "experiments/phase8/predictions.csv"
        )

        # 3. Initialize Decision Engine
        self.decision_engine = CharterDecisionEngine(
            ports_config_path=self.ports_config_path,
            uncertainty_estimator=self.uncertainty_estimator,
            threshold_cost_saving_pct=2.0,
            threshold_price_rise_pct=1.5,
        )

        # 4. Load Features Matrix
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features matrix not found at: {self.features_path.resolve()}")
        self.df_features = pd.read_csv(self.features_path)
        self.df_features[DATE_COLUMN] = pd.to_datetime(self.df_features[DATE_COLUMN])
        self.df_features = self.df_features.sort_values(by=DATE_COLUMN).reset_index(drop=True)

    def process_request(self, request: FICOSRequest) -> FICOSRecommendation:
        """Execute full end-to-end decision-support pipeline."""
        # Step 1: Input Validation
        request.validate()

        # Step 2: Contemporaneous Historical Data Extraction (Zero future data leakage)
        target_date = pd.to_datetime(request.decision_date)
        contemporaneous_df = self.df_features[self.df_features[DATE_COLUMN] <= target_date]
        if contemporaneous_df.empty:
            raise ValidationError(
                f"No historical market data available on or before decision_date '{request.decision_date}'. "
                f"Earliest available date is {self.df_features[DATE_COLUMN].min().strftime('%Y-%m-%d')}."
            )

        current_row = contemporaneous_df.iloc[-1]
        historical_decision_date = current_row[DATE_COLUMN].strftime("%Y-%m-%d")

        # Step 3: Vessel Feasibility & Port Navigational Evaluation
        feasible_vessels, port_info = self.decision_engine.evaluate_vessel_feasibility(
            cargo_quantity_mt=request.cargo_quantity_mt,
            destination_port=request.destination_port,
        )

        if request.preferred_vessel and request.preferred_vessel in feasible_vessels:
            selected_vessel = request.preferred_vessel
            vessel_reason = f"User preferred vessel '{selected_vessel}' is physically feasible at {port_info.get('name')}."
        else:
            selected_vessel = self.decision_engine.select_optimal_vessel(feasible_vessels, request.cargo_quantity_mt)
            vessel_reason = (
                f"Recommended '{selected_vessel}' as optimal capacity utilization for {request.cargo_quantity_mt:,.0f} MT "
                f"satisfying {port_info.get('name')} max draft limit of {port_info.get('max_draft_m')}m."
            )

        target_key = self.decision_engine.vessels[selected_vessel]["target_key"]

        # Step 4: Current Freight Spot Level
        if request.current_freight is not None and request.current_freight > 0:
            current_freight = float(request.current_freight)
        else:
            current_freight = float(current_row[f"{target_key}_level"])

        # Step 5: Multi-Step Freight Forecasting & Empirical Uncertainty Bounds
        horizon_days = min(7, max(1, request.laycan_days_allowed))

        # Contemporaneous drift estimation from trailing 5-day difference
        diff_col = f"{target_key}_diff_5"
        diff_5 = float(current_row[diff_col]) if diff_col in current_row else 0.0
        drift_pct = (diff_5 / (5.0 * max(1.0, current_freight))) * 100.0

        trajectory = self.decision_engine.generate_horizon_forecast(
            current_level=current_freight,
            target_key=target_key,
            horizon_days=horizon_days,
            expected_drift_pct_per_day=drift_pct,
        )

        forecast_vals = [pt["point_forecast"] for pt in trajectory]
        p10_vals = [pt["lower_p10"] for pt in trajectory]
        p50_vals = [pt["point_forecast"] for pt in trajectory]
        p90_vals = [pt["upper_p90"] for pt in trajectory]

        forecast_output = ForecastOutput(
            vessel_class=selected_vessel,
            target_index=target_key,
            horizon_days=horizon_days,
            values=forecast_vals,
            p10=p10_vals,
            p50=p50_vals,
            p90=p90_vals,
        )

        # Step 6: Risk Evaluation
        vol_col = f"{target_key}_return_vol_7"
        rolling_vol = float(current_row[vol_col]) if vol_col in current_row and not np.isnan(current_row[vol_col]) else 0.02

        gpr_col = "gpr_spike_ratio_ma30"
        gpr_ratio = float(current_row[gpr_col]) if gpr_col in current_row and not np.isnan(current_row[gpr_col]) else 1.0

        precip_col = "precip_mm_lag_1"
        precip = float(current_row[precip_col]) if precip_col in current_row and not np.isnan(current_row[precip_col]) else 0.0
        weather_alert = precip > 25.0

        risk_level, risk_reasons = self.decision_engine.evaluate_risk(
            target_key=target_key,
            trajectory=trajectory,
            current_level=current_freight,
            rolling_volatility=rolling_vol,
            gpr_spike_ratio=gpr_ratio,
            weather_alert=weather_alert,
        )
        risk_output = RiskOutput(level=risk_level, reasons=risk_reasons)

        # Step 7: Voyage Cost Calculation across Days
        turnaround = port_info.get("turnaround_days_assumption", 3.0)
        demurrage = port_info.get("demurrage_usd_per_day", 15000.0)

        cost_now = self.decision_engine.compute_voyage_cost(
            freight_index=current_freight,
            vessel_name=selected_vessel,
            cargo_quantity_mt=request.cargo_quantity_mt,
            voyage_duration_days=request.voyage_duration_days,
            turnaround_days=turnaround,
            demurrage_usd_per_day=demurrage,
            delay_days_waited=0,
            daily_holding_cost_usd=request.daily_holding_cost_usd,
        )["total_cost_usd"]

        cost_trajectory = [cost_now]
        for pt in trajectory:
            c = self.decision_engine.compute_voyage_cost(
                freight_index=pt["point_forecast"],
                vessel_name=selected_vessel,
                cargo_quantity_mt=request.cargo_quantity_mt,
                voyage_duration_days=request.voyage_duration_days,
                turnaround_days=turnaround,
                demurrage_usd_per_day=demurrage,
                delay_days_waited=pt["day_ahead"],
                daily_holding_cost_usd=request.daily_holding_cost_usd,
            )["total_cost_usd"]
            cost_trajectory.append(c)

        min_cost = min(cost_trajectory)
        optimal_day = int(np.argmin(cost_trajectory))
        savings_usd = cost_now - min_cost
        savings_pct = (savings_usd / cost_now) * 100.0 if cost_now > 0 else 0.0

        # Step 8: Rule-Based Charter Decision Logic
        reasons = []
        action = "FLEXIBLE / MONITOR"
        expected_total_freight_change_pct = ((trajectory[-1]["point_forecast"] - current_freight) / current_freight) * 100.0

        if expected_total_freight_change_pct >= self.decision_engine.threshold_rise_pct:
            action = "CHARTER NOW"
            reasons.append(
                f"CHARTER NOW because {selected_vessel} freight rate is forecast to rise by "
                f"{expected_total_freight_change_pct:+.1f}% over the {horizon_days}-day horizon. "
                f"Fixing now avoids higher future hire rates."
            )
            rule_triggered = "RisingFreightMomentumThreshold"
        elif optimal_day > 0 and savings_pct >= self.decision_engine.threshold_saving_pct and risk_level != "HIGH":
            action = "WAIT"
            reasons.append(
                f"WAIT because projected freight rate declines sufficiently to yield an estimated net cost saving of "
                f"${savings_usd:,.0f} ({savings_pct:.1f}%) net of daily cargo holding costs."
            )
            rule_triggered = "ExpectedCostSavingOpportunity"
        elif risk_level == "HIGH" and expected_total_freight_change_pct >= 0:
            action = "CHARTER NOW"
            reasons.append(
                f"CHARTER NOW because elevated market risk was detected. Locking in current spot rate eliminates upside rate exposure."
            )
            rule_triggered = "HighRiskSpotLockin"
        else:
            action = "FLEXIBLE / MONITOR"
            reasons.append(
                f"FLEXIBLE / MONITOR because projected freight movement ({expected_total_freight_change_pct:+.1f}%) "
                f"remains within the model's uncertainty range and daily market noise."
            )
            rule_triggered = "UncertaintyBandIndifference"

        reasons.append(vessel_reason)

        decision_output = DecisionOutput(
            action=action,
            optimal_entry_day=optimal_day,
            expected_cost_now_usd=round(cost_now, 2),
            expected_cost_optimal_usd=round(min_cost, 2),
            estimated_savings_usd=round(savings_usd, 2),
            estimated_savings_pct=round(savings_pct, 2),
        )

        vessel_output = VesselOutput(
            recommended=selected_vessel,
            feasible_vessels=feasible_vessels,
            reason=vessel_reason,
        )

        assumptions = [
            "Freight forecasts derived from production Ridge regression model (alpha=1.0).",
            "Uncertainty intervals constructed from empirical walk-forward out-of-sample residual quantiles (P10/P90).",
            f"Port operational parameters for {port_info.get('name')} are prototype assumptions for decision support.",
            "Decision is generated strictly using contemporaneous information available up to decision date.",
        ]

        audit_record = AuditRecord(
            request_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            decision_date=historical_decision_date,
            model_name="Ridge_a1.0",
            model_version="Phase8_Production_Champion",
            features_count=len(self.model_registry.expected_feature_cols),
            selected_vessel=selected_vessel,
            rule_triggered=rule_triggered,
            final_action=action,
            hindsight_oracle_available=False,
        )

        return FICOSRecommendation(
            decision_date=historical_decision_date,
            destination_port=request.destination_port.lower().strip(),
            cargo_quantity_mt=request.cargo_quantity_mt,
            cargo_type=request.cargo_type,
            forecast=forecast_output,
            vessel=vessel_output,
            decision=decision_output,
            risk=risk_output,
            reasons=reasons,
            assumptions=assumptions,
            audit=audit_record,
        )
