"""FICOS Main Application Entry Point."""

from src.application.schemas import FICOSRequest
from src.application.service import FICOSService
from src.application.cli import format_cli_report


def main():
    print("Initializing FICOS (Freight Intelligence & Charter Optimization System)...")
    service = FICOSService()

    # Sample demo request
    sample_req = FICOSRequest(
        decision_date="2018-09-25",
        cargo_quantity_mt=75000.0,
        cargo_type="Coking Coal",
        destination_port="dhamra",
        laycan_days_allowed=7,
        voyage_duration_days=18.0,
    )

    print("\nExecuting End-to-End Decision Pipeline for Sample Request:")
    rec = service.process_request(sample_req)
    print(format_cli_report(rec))


if __name__ == "__main__":
    main()
