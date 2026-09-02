"""Chartering decision-support engine, vessel suitability filtering, cost modeling, and market-entry timing logic."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
import yaml
import numpy as np
import pandas as pd

from src.decision.uncertainty import ResidualUncertaintyEstimator


@dataclass
class CharterDecisionRequest:
    """Input specification for a chartering decision inquiry."""
    cargo_type: str = "Coking Coal"
    cargo_quantity_mt: float = 75000.0
    origin: str = "Gladstone, Australia"
    destination_port: str = "paradip"
    decision_date: Optional[str] = None
    laycan_days_allowed: int = 7
    voyage_duration_days: float = 18.0
    daily_cargo_holding_cost_usd: float = 1500.0  # Cost of delaying cargo dispatch per day


@dataclass
class CharterRecommendation:
    """Structured, explainable output from the decision-support engine."""
    charter_action: str  # "CHARTER NOW", "WAIT", "FLEXIBLE / MONITOR"
    recommended_vessel: str  # "Handysize", "Supramax", "Panamax", "Capesize"
    current_freight_index: float
    forecast_trajectory: List[Dict[str, Any]]
    expected_cost_now_usd: float
    expected_cost_optimal_usd: float
    optimal_entry_day: int  # 0 = now, 1..N = wait N days
    estimated_savings_usd: float
    estimated_savings_pct: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    risk_reasons: List[str]
    reasons: List[str]
    port_constraints_evaluated: Dict[str, Any]
    feasible_vessels: List[str]
    prototype_assumption: bool = True


class CharterDecisionEngine:
    """Prototype decision-support engine for East Coast Indian dry-bulk vessel chartering."""

    def __init__(
        self,
        ports_config_path: Union[str, Path] = "configs/ports.yaml",
        uncertainty_estimator: Optional[ResidualUncertaintyEstimator] = None,
        threshold_cost_saving_pct: float = 2.0,  # Minimum 2% expected cost saving to justify waiting
        threshold_price_rise_pct: float = 1.5,   # Expected rise > 1.5% favors chartering now
    ):
        self.ports_config = self._load_ports_config(ports_config_path)
        self.ports = self.ports_config.get("ports", {})
        self.vessels = self.ports_config.get("vessel_specifications", {})
        self.uncertainty_estimator = uncertainty_estimator or ResidualUncertaintyEstimator()
        self.threshold_saving_pct = threshold_cost_saving_pct
        self.threshold_rise_pct = threshold_price_rise_pct

    @staticmethod
    def _load_ports_config(path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Ports configuration not found at: {p.resolve()}")
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def evaluate_vessel_feasibility(
        self, cargo_quantity_mt: float, destination_port: str
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Filter feasible vessel types matching cargo size and port navigational limits.

        Args:
            cargo_quantity_mt: Cargo parcel size in metric tons.
            destination_port: Key matching ports.yaml (e.g. 'paradip', 'vizag').

        Returns:
            Tuple[List[str], Dict]: (feasible_vessels, port_evaluation_metadata).
        """
        port_key = destination_port.lower().strip().replace("-", "_")
        port_info = self.ports.get(port_key)
        if not port_info:
            raise ValueError(f"Unknown destination port '{destination_port}'. Available ports: {list(self.ports.keys())}")

        allowed_port_vessels = port_info.get("allowed_vessels", [])
        max_dwt = port_info.get("max_dwt", 1000000)
        max_draft = port_info.get("max_draft_m", 50.0)

        feasible = []
        for vessel_name in allowed_port_vessels:
            if vessel_name not in self.vessels:
                continue
            specs = self.vessels[vessel_name]
            # Must satisfy physical draft and port DWT limit
            if specs["typical_draft_m"] <= max_draft and specs["typical_capacity_mt"] <= max_dwt:
                feasible.append(vessel_name)

        if not feasible and allowed_port_vessels:
            feasible = allowed_port_vessels

        return feasible, port_info

    def select_optimal_vessel(self, feasible_vessels: List[str], cargo_quantity_mt: float) -> str:
        """Select the vessel class maximizing freight economy for the cargo quantity."""
        if not feasible_vessels:
            return "Panamax"  # Default fallback

        # Choose the feasible vessel whose typical capacity is closest to or accommodates the cargo parcel
        best_vessel = feasible_vessels[0]
        min_waste = float("inf")
        for v in feasible_vessels:
            if v not in self.vessels:
                continue
            cap = self.vessels[v]["typical_capacity_mt"]
            waste = abs(cap - cargo_quantity_mt)
            if waste < min_waste:
                min_waste = waste
                best_vessel = v
        return best_vessel

    def generate_horizon_forecast(
        self,
        current_level: float,
        target_key: str,
        horizon_days: int = 7,
        expected_drift_pct_per_day: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Generate multi-step trajectory with empirical uncertainty bounds.

        Args:
            current_level: Index level at decision time t=0.
            target_key: Target index identifier ('bdi_hsi', 'bdi_si', 'bdi_pi', 'bdi_ci').
            horizon_days: Forward projection horizon (e.g. 7 trading sessions).
            expected_drift_pct_per_day: Model-derived momentum/trend drift rate.

        Returns:
            List[Dict]: List of daily trajectory points with point, lower P10, upper P90, interval width.
        """
        trajectory = []
        level = current_level

        for day in range(1, horizon_days + 1):
            # Causal drift projection
            level = level * (1.0 + expected_drift_pct_per_day / 100.0)
            interval = self.uncertainty_estimator.construct_prediction_interval(
                target_key=target_key, point_forecast=level, horizon_step=day
            )
            trajectory.append({
                "day_ahead": day,
                "point_forecast": interval["point_p50"],
                "lower_p10": interval["lower_p10"],
                "upper_p90": interval["upper_p90"],
                "interval_width": interval["interval_width"],
            })

        return trajectory

    def compute_voyage_cost(
        self,
        freight_index: float,
        vessel_name: str,
        cargo_quantity_mt: float,
        voyage_duration_days: float,
        turnaround_days: float,
        demurrage_usd_per_day: float,
        delay_days_waited: int = 0,
        daily_holding_cost_usd: float = 1500.0,
    ) -> Dict[str, float]:
        """Compute transparent voyage cost breakdown.

        Args:
            freight_index: Freight index level (e.g. 1200 points).
            vessel_name: Vessel class name.
            cargo_quantity_mt: Cargo metric tons.
            voyage_duration_days: Sea voyage duration.
            turnaround_days: Expected port discharge duration.
            demurrage_usd_per_day: Port demurrage rate.
            delay_days_waited: Days waited before charter entry.
            daily_holding_cost_usd: Cargo opportunity holding cost.

        Returns:
            Dict[str, float]: Cost breakdown dictionary.
        """
        multiplier = self.vessels[vessel_name]["daily_hire_multiplier_usd"]
        
        # Estimated Daily Hire Rate = Index Level * Multiplier
        daily_hire_rate = freight_index * multiplier

        # 1. Direct Sea Voyage Freight Cost
        freight_cost = daily_hire_rate * voyage_duration_days

        # 2. Port Stay & Expected Handling/Demurrage Cost
        port_stay_cost = turnaround_days * demurrage_usd_per_day

        # 3. Holding / Storage Delay Cost incurred if entry is postponed
        holding_cost = delay_days_waited * daily_holding_cost_usd

        # Total Voyage Expected Cost
        total_cost = freight_cost + port_stay_cost + holding_cost

        return {
            "freight_cost_usd": round(freight_cost, 2),
            "port_stay_cost_usd": round(port_stay_cost, 2),
            "holding_delay_cost_usd": round(holding_cost, 2),
            "total_cost_usd": round(total_cost, 2),
            "cost_per_mt_usd": round(total_cost / max(1.0, cargo_quantity_mt), 2),
        }

    def evaluate_risk(
        self,
        target_key: str,
        trajectory: List[Dict[str, Any]],
        current_level: float,
        rolling_volatility: float = 0.02,
        gpr_spike_ratio: float = 1.0,
        weather_alert: bool = False,
    ) -> Tuple[str, List[str]]:
        """Evaluate market and operational risk indicators."""
        reasons = []
        risk_score = 0

        # Check 1: Trajectory Uncertainty Width
        final_width = trajectory[-1]["interval_width"]
        rel_uncertainty = final_width / max(1.0, current_level)
        if rel_uncertainty > 0.20:
            risk_score += 2
            reasons.append(f"High forecast uncertainty band ({rel_uncertainty * 100:.1f}% interval width at day {len(trajectory)}).")
        elif rel_uncertainty > 0.10:
            risk_score += 1
            reasons.append(f"Moderate forecast uncertainty ({rel_uncertainty * 100:.1f}% interval width).")

        # Check 2: Market Volatility
        if target_key == "bdi_ci":
            risk_score += 1
            reasons.append("Capesize segment exhibits structural volatility and convex tail movements.")
        if rolling_volatility > 0.035:
            risk_score += 2
            reasons.append(f"Elevated historical freight return volatility ({rolling_volatility * 100:.1f}% daily vol).")

        # Check 3: Geopolitical Risk
        if gpr_spike_ratio > 1.30:
            risk_score += 2
            reasons.append(f"Geopolitical Risk spike detected ({gpr_spike_ratio:.2f}x above baseline MA30).")
        elif gpr_spike_ratio > 1.15:
            risk_score += 1
            reasons.append(f"Elevated Geopolitical Risk ratio ({gpr_spike_ratio:.2f}x).")

        # Check 4: Coastal Weather
        if weather_alert:
            risk_score += 1
            reasons.append("Coastal weather disruption alert at destination port approach.")

        if risk_score >= 3:
            level = "HIGH"
        elif risk_score >= 1:
            level = "MEDIUM"
        else:
            level = "LOW"
            reasons.append("Stable freight conditions with narrow forecast dispersion and baseline risk indicators.")

        return level, reasons

    def recommend_charter(
        self,
        request: CharterDecisionRequest,
        current_freight_index: float,
        expected_drift_pct_per_day: float = 0.0,
        rolling_volatility: float = 0.02,
        gpr_spike_ratio: float = 1.0,
        weather_alert: bool = False,
    ) -> CharterRecommendation:
        """Generate comprehensive, explainable charter recommendation.

        Args:
            request: Chartering inquiry specifications.
            current_freight_index: Current index level at decision date.
            expected_drift_pct_per_day: Expected freight daily rate of change.
            rolling_volatility: Recent freight return volatility.
            gpr_spike_ratio: Geopolitical spike ratio relative to MA30.
            weather_alert: Boolean indicator for port weather disruption.

        Returns:
            CharterRecommendation: Full typed recommendation object.
        """
        # 1. Vessel Feasibility & Selection
        feasible_vessels, port_info = self.evaluate_vessel_feasibility(
            cargo_quantity_mt=request.cargo_quantity_mt,
            destination_port=request.destination_port,
        )
        recommended_vessel = self.select_optimal_vessel(feasible_vessels, request.cargo_quantity_mt)
        target_key = self.vessels[recommended_vessel]["target_key"]

        # 2. Multi-Step Horizon Forecast & Prediction Intervals
        horizon_days = min(7, max(1, request.laycan_days_allowed))
        trajectory = self.generate_horizon_forecast(
            current_level=current_freight_index,
            target_key=target_key,
            horizon_days=horizon_days,
            expected_drift_pct_per_day=expected_drift_pct_per_day,
        )

        # 3. Cost Modeling across Timing Alternatives (Day 0 = Now, Day 1..N = Wait)
        turnaround = port_info.get("turnaround_days_assumption", 3.0)
        demurrage = port_info.get("demurrage_usd_per_day", 15000.0)

        cost_now = self.compute_voyage_cost(
            freight_index=current_freight_index,
            vessel_name=recommended_vessel,
            cargo_quantity_mt=request.cargo_quantity_mt,
            voyage_duration_days=request.voyage_duration_days,
            turnaround_days=turnaround,
            demurrage_usd_per_day=demurrage,
            delay_days_waited=0,
            daily_holding_cost_usd=request.daily_cargo_holding_cost_usd,
        )["total_cost_usd"]

        cost_by_day = [cost_now]
        for pt in trajectory:
            c = self.compute_voyage_cost(
                freight_index=pt["point_forecast"],
                vessel_name=recommended_vessel,
                cargo_quantity_mt=request.cargo_quantity_mt,
                voyage_duration_days=request.voyage_duration_days,
                turnaround_days=turnaround,
                demurrage_usd_per_day=demurrage,
                delay_days_waited=pt["day_ahead"],
                daily_holding_cost_usd=request.daily_cargo_holding_cost_usd,
            )["total_cost_usd"]
            cost_by_day.append(c)

        min_cost = min(cost_by_day)
        optimal_day = int(np.argmin(cost_by_day))
        savings_usd = cost_now - min_cost
        savings_pct = (savings_usd / cost_now) * 100.0 if cost_now > 0 else 0.0

        # 4. Risk Evaluation
        risk_level, risk_reasons = self.evaluate_risk(
            target_key=target_key,
            trajectory=trajectory,
            current_level=current_freight_index,
            rolling_volatility=rolling_volatility,
            gpr_spike_ratio=gpr_spike_ratio,
            weather_alert=weather_alert,
        )

        # 5. Transparent Rule-Based Decision Logic
        reasons = []
        action = "FLEXIBLE / MONITOR"

        # Case A: Strong Expected Price Rise -> Favor CHARTER NOW
        expected_total_freight_change_pct = ((trajectory[-1]["point_forecast"] - current_freight_index) / current_freight_index) * 100.0

        if expected_total_freight_change_pct >= self.threshold_rise_pct:
            action = "CHARTER NOW"
            reasons.append(
                f"Freight rate for {recommended_vessel} is forecasted to rise by {expected_total_freight_change_pct:+.1f}% "
                f"over the next {horizon_days} trading sessions. Chartering now avoids higher future hire costs."
            )
        # Case B: Meaningful Cost Saving by Waiting -> Favor WAIT
        elif optimal_day > 0 and savings_pct >= self.threshold_saving_pct and risk_level != "HIGH":
            action = "WAIT"
            reasons.append(
                f"Freight rate is projected to ease over the next {optimal_day} days, yielding an estimated net cost saving "
                f"of ${savings_usd:,.0f} ({savings_pct:.1f}%) net of daily cargo holding costs."
            )
        # Case C: High Risk / High Volatility -> Favor Locking in Spot Rate
        elif risk_level == "HIGH" and expected_total_freight_change_pct >= 0:
            action = "CHARTER NOW"
            reasons.append(
                f"High market uncertainty/risk detected. Locking in current spot rate is recommended to eliminate upside rate exposure."
            )
        # Case D: Minor/Flat Movement -> Flexible
        else:
            action = "FLEXIBLE / MONITOR"
            reasons.append(
                f"Projected price movement ({expected_total_freight_change_pct:+.1f}%) is within standard market noise and "
                f"uncertainty margins. Monitor rate movements across the laycan window before fixing."
            )

        # Vessel suitability reason
        reasons.append(
            f"Recommended vessel '{recommended_vessel}' satisfies {port_info.get('name')} navigational constraints "
            f"(Draft {self.vessels[recommended_vessel]['typical_draft_m']}m <= {port_info.get('max_draft_m')}m max draft) "
            f"and provides optimal capacity utilization for {request.cargo_quantity_mt:,.0f} MT of {request.cargo_type}."
        )

        return CharterRecommendation(
            charter_action=action,
            recommended_vessel=recommended_vessel,
            current_freight_index=round(current_freight_index, 2),
            forecast_trajectory=trajectory,
            expected_cost_now_usd=round(cost_now, 2),
            expected_cost_optimal_usd=round(min_cost, 2),
            optimal_entry_day=optimal_day,
            estimated_savings_usd=round(savings_usd, 2),
            estimated_savings_pct=round(savings_pct, 2),
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            reasons=reasons,
            port_constraints_evaluated={
                "port_name": port_info.get("name"),
                "max_draft_m": port_info.get("max_draft_m"),
                "max_loa_m": port_info.get("max_loa_m"),
                "max_dwt": port_info.get("max_dwt"),
                "allowed_vessels": port_info.get("allowed_vessels"),
                "handling_rate_tpd": port_info.get("handling_rate_tpd"),
            },
            feasible_vessels=feasible_vessels,
            prototype_assumption=True,
        )
