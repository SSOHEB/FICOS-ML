"""Command-Line Interface (CLI) for the FICOS decision-support system."""

import argparse
import sys
from pathlib import Path
import json

from src.application.schemas import FICOSRequest, ValidationError
from src.application.service import FICOSService


def format_cli_report(rec) -> str:
    """Format structured FICOS recommendation into clean professional text report."""
    lines = []
    lines.append("=" * 60)
    lines.append("           FICOS CHARTER DECISION RECOMMENDATION")
    lines.append("=" * 60)
    lines.append(f"Decision Date:       {rec.decision_date}")
    lines.append(f"Destination Port:    {rec.destination_port.upper()}")
    lines.append(f"Cargo:               {rec.cargo_quantity_mt:,.0f} MT of {rec.cargo_type}")
    lines.append(f"Recommended Vessel:  {rec.vessel.recommended}")
    lines.append(f"Feasible Vessels:    {', '.join(rec.vessel.feasible_vessels)}")
    lines.append("-" * 60)
    lines.append("7-DAY FREIGHT RATE FORECAST & UNCERTAINTY (Points):")
    for i, (val, p10, p90) in enumerate(zip(rec.forecast.values, rec.forecast.p10, rec.forecast.p90), 1):
        lines.append(f"  Day {i}: Point={val:>7.1f} | 80% CI=[{p10:>7.1f} to {p90:>7.1f}]")
    lines.append("-" * 60)
    lines.append(f"DECISION:            >>> {rec.decision.action} <<<")
    if rec.decision.optimal_entry_day > 0:
        lines.append(f"Optimal Entry Day:   Day {rec.decision.optimal_entry_day}")
    lines.append(f"Expected Cost (Now): ${rec.decision.expected_cost_now_usd:,.2f}")
    lines.append(f"Optimal Cost:        ${rec.decision.expected_cost_optimal_usd:,.2f}")
    if rec.decision.estimated_savings_usd > 0:
        lines.append(f"Estimated Savings:   ${rec.decision.estimated_savings_usd:,.2f} ({rec.decision.estimated_savings_pct:.1f}%)")
    lines.append("-" * 60)
    lines.append(f"MARKET RISK LEVEL:   {rec.risk.level}")
    for r in rec.risk.reasons:
        lines.append(f"  * {r}")
    lines.append("-" * 60)
    lines.append("DECISION RATIONALE:")
    for r in rec.reasons:
        lines.append(f"  * {r}")
    lines.append("-" * 60)
    lines.append("OPERATIONAL ASSUMPTIONS:")
    for a in rec.assumptions:
        lines.append(f"  * {a}")
    lines.append(f"Audit Request ID:    {rec.audit.request_id} (Model: {rec.audit.model_name})")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="FICOS Freight Intelligence & Charter Optimization System")
    parser.add_argument("--quantity", "-q", type=float, default=75000.0, help="Cargo parcel quantity in metric tons")
    parser.add_argument("--cargo", "-c", type=str, default="Coking Coal", help="Cargo type name")
    parser.add_argument("--destination", "-d", type=str, default="dhamra", help="Destination port name (e.g. paradip, vizag, dhamra)")
    parser.add_argument("--date", "-t", type=str, default="2018-09-25", help="Decision date in YYYY-MM-DD format")
    parser.add_argument("--freight", "-f", type=float, default=None, help="Optional current freight index level")
    parser.add_argument("--laycan", "-l", type=int, default=7, help="Allowed laycan window days (1..30)")
    parser.add_argument("--voyage-days", "-v", type=float, default=18.0, help="Estimated sea voyage duration days")
    parser.add_argument("--json", action="store_true", help="Output recommendation as raw JSON")

    args = parser.parse_args()

    try:
        service = FICOSService()
        req = FICOSRequest(
            decision_date=args.date,
            cargo_quantity_mt=args.quantity,
            cargo_type=args.cargo,
            destination_port=args.destination,
            current_freight=args.freight,
            laycan_days_allowed=args.laycan,
            voyage_duration_days=args.voyage_days,
        )
        rec = service.process_request(req)

        if args.json:
            print(json.dumps(rec.to_dict(), indent=2))
        else:
            print(format_cli_report(rec))

    except ValidationError as e:
        print(f"\n[FICOS Validation Error] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[FICOS System Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
