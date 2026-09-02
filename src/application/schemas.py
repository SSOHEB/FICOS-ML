"""Application request, recommendation, and audit schemas for FICOS."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import uuid


SUPPORTED_PORTS = [
    "paradip",
    "vizag",
    "gangavaram",
    "gopalpur",
    "dhamra",
    "sagar_sandheads",
    "haldia",
]

SUPPORTED_VESSELS = [
    "Handysize",
    "Supramax",
    "Panamax",
    "Capesize",
]


class ValidationError(ValueError):
    """Raised when application input fails business or schema validation."""
    pass


@dataclass
class FICOSRequest:
    """Validated input specification for a FICOS chartering inquiry."""
    decision_date: str
    cargo_quantity_mt: float
    cargo_type: str = "Coking Coal"
    destination_port: str = "paradip"
    current_freight: Optional[float] = None
    origin: str = "Gladstone, Australia"
    laycan_days_allowed: int = 7
    voyage_duration_days: float = 18.0
    preferred_vessel: Optional[str] = None
    daily_holding_cost_usd: float = 1500.0

    def validate(self) -> None:
        """Validate input parameters against business constraints."""
        # 1. Validate decision_date format
        try:
            datetime.strptime(self.decision_date, "%Y-%m-%d")
        except Exception:
            raise ValidationError(
                f"Invalid decision_date '{self.decision_date}'. Expected format YYYY-MM-DD."
            )

        # 2. Validate cargo quantity
        if self.cargo_quantity_mt is None or self.cargo_quantity_mt <= 0:
            raise ValidationError(
                f"Invalid cargo_quantity_mt ({self.cargo_quantity_mt}). Must be > 0."
            )

        # 3. Validate destination port
        port_norm = self.destination_port.lower().strip().replace("-", "_")
        supported_norm = [p.replace("-", "_") for p in SUPPORTED_PORTS]
        if port_norm not in supported_norm:
            raise ValidationError(
                f"Unsupported destination_port '{self.destination_port}'. Supported ports: {SUPPORTED_PORTS}"
            )

        # 4. Validate preferred vessel if provided
        if self.preferred_vessel is not None:
            if self.preferred_vessel not in SUPPORTED_VESSELS:
                raise ValidationError(
                    f"Invalid preferred_vessel '{self.preferred_vessel}'. Supported vessels: {SUPPORTED_VESSELS}"
                )

        # 5. Validate current freight if provided
        if self.current_freight is not None and self.current_freight <= 0:
            raise ValidationError(
                f"Invalid current_freight ({self.current_freight}). Must be strictly positive (> 0)."
            )

        # 6. Validate laycan days
        if self.laycan_days_allowed < 1 or self.laycan_days_allowed > 30:
            raise ValidationError(
                f"Invalid laycan_days_allowed ({self.laycan_days_allowed}). Must be between 1 and 30 days."
            )

        # 7. Validate voyage duration
        if self.voyage_duration_days <= 0:
            raise ValidationError(
                f"Invalid voyage_duration_days ({self.voyage_duration_days}). Must be > 0."
            )


@dataclass
class ForecastOutput:
    """Multi-step freight forecast and empirical uncertainty bounds."""
    vessel_class: str
    target_index: str
    horizon_days: int
    values: List[float]
    p10: List[float]
    p50: List[float]
    p90: List[float]


@dataclass
class VesselOutput:
    """Vessel feasibility evaluation and selection."""
    recommended: str
    feasible_vessels: List[str]
    reason: str


@dataclass
class DecisionOutput:
    """Charter market entry timing recommendation and cost evaluation."""
    action: str  # "CHARTER NOW", "WAIT", "FLEXIBLE / MONITOR"
    optimal_entry_day: int
    expected_cost_now_usd: float
    expected_cost_optimal_usd: float
    estimated_savings_usd: float
    estimated_savings_pct: float


@dataclass
class RiskOutput:
    """Risk assessment and explanatory signals."""
    level: str  # "LOW", "MEDIUM", "HIGH"
    reasons: List[str]


@dataclass
class AuditRecord:
    """Complete auditable decision record."""
    request_id: str
    timestamp: str
    decision_date: str
    model_name: str
    model_version: str
    features_count: int
    selected_vessel: str
    rule_triggered: str
    final_action: str
    hindsight_oracle_available: bool = False
    hindsight_note: str = (
        "Decision generated strictly using information available on or before decision_date. "
        "Hindsight benchmark is computed separately for retrospective evaluation."
    )


@dataclass
class FICOSRecommendation:
    """Unified, stable output schema for FICOS decision support."""
    decision_date: str
    destination_port: str
    cargo_quantity_mt: float
    cargo_type: str
    forecast: ForecastOutput
    vessel: VesselOutput
    decision: DecisionOutput
    risk: RiskOutput
    reasons: List[str]
    assumptions: List[str]
    audit: AuditRecord

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation object to serializable dictionary."""
        return asdict(self)
