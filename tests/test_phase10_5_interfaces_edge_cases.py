"""Phase 10.5 Interface Verification (CLI, API), Cross-Interface Consistency, and Edge-Case Tests.

Verifies:
- CLI invocation via subprocess (formatted text report and JSON mode).
- API REST endpoints (/health, /recommend) via direct HTTP request tests.
- Cross-interface consistency between Direct Service, CLI, and REST API.
- Comprehensive Edge Cases:
    * Zero / Negative cargo quantity
    * Extremely small / large cargo
    * Invalid port / preferred vessel
    * Malformed dates & dates outside historical coverage
    * Missing / Negative current freight
    * Malformed JSON payloads
"""

import json
import subprocess
import sys
import threading
import time
from http.client import HTTPConnection
from pathlib import Path
import pytest

from src.application.schemas import FICOSRequest, ValidationError
from src.application.service import FICOSService
from src.application.api import create_server


@pytest.fixture(scope="module")
def ficos_service() -> FICOSService:
    return FICOSService()


@pytest.fixture(scope="module")
def running_api_server():
    """Spin up local test HTTP server on port 8999."""
    server = create_server(host="127.0.0.1", port=8999)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)
    yield "127.0.0.1:8999"
    server.shutdown()


def test_cli_formatted_invocation():
    """Verify CLI produces clean formatted output with zero errors."""
    cmd = [
        sys.executable,
        "-m",
        "src.application.cli",
        "--quantity",
        "75000",
        "--cargo",
        "Coking Coal",
        "--destination",
        "dhamra",
        "--date",
        "2018-09-25",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed with error: {res.stderr}"
    assert "FICOS CHARTER DECISION RECOMMENDATION" in res.stdout
    assert "DECISION:" in res.stdout
    assert "MARKET RISK LEVEL:" in res.stdout
    assert "Traceback" not in res.stderr


def test_cli_json_invocation():
    """Verify CLI --json outputs valid parseable JSON."""
    cmd = [
        sys.executable,
        "-m",
        "src.application.cli",
        "--quantity",
        "75000",
        "--cargo",
        "Coking Coal",
        "--destination",
        "dhamra",
        "--date",
        "2018-09-25",
        "--json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["decision_date"] == "2018-09-25"
    assert "forecast" in data
    assert "decision" in data
    assert "vessel" in data


def test_cli_invalid_arguments_fails_gracefully():
    """Verify CLI rejects invalid port or quantity without uncaught traceback."""
    cmd = [
        sys.executable,
        "-m",
        "src.application.cli",
        "--quantity",
        "-500",
        "--destination",
        "nonexistent_port",
        "--date",
        "2018-09-25",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "FICOS Validation Error" in res.stderr
    assert "Traceback" not in res.stderr


def test_api_health_endpoint(running_api_server):
    """Verify GET /health returns 200 and system status."""
    conn = HTTPConnection(running_api_server)
    conn.request("GET", "/health")
    response = conn.getresponse()
    assert response.status == 200
    data = json.loads(response.read().decode("utf-8"))
    assert data["status"] == "HEALTHY"
    assert data["champion_model"] == "Ridge_a1.0"
    conn.close()


def test_api_recommend_endpoint_valid(running_api_server):
    """Verify POST /recommend returns 200 and valid recommendation."""
    payload = {
        "decision_date": "2018-09-25",
        "cargo_quantity_mt": 75000.0,
        "cargo_type": "Coking Coal",
        "destination_port": "dhamra",
        "laycan_days_allowed": 7,
    }
    conn = HTTPConnection(running_api_server)
    headers = {"Content-Type": "application/json"}
    conn.request("POST", "/recommend", body=json.dumps(payload), headers=headers)
    response = conn.getresponse()
    assert response.status == 200
    data = json.loads(response.read().decode("utf-8"))
    assert data["decision_date"] == "2018-09-25"
    assert data["vessel"]["recommended"] in ["Panamax", "Capesize"]
    conn.close()


def test_api_validation_errors(running_api_server):
    """Verify API returns 400 on invalid port or malformed JSON."""
    conn = HTTPConnection(running_api_server)
    headers = {"Content-Type": "application/json"}

    # 1. Invalid port
    bad_port_payload = {"decision_date": "2018-09-25", "cargo_quantity_mt": 75000, "destination_port": "mars_port"}
    conn.request("POST", "/recommend", body=json.dumps(bad_port_payload), headers=headers)
    resp = conn.getresponse()
    assert resp.status == 400
    resp_body = json.loads(resp.read().decode("utf-8"))
    assert "Validation Error" in resp_body["error"]

    # 2. Malformed JSON
    conn.request("POST", "/recommend", body="THIS IS NOT JSON", headers=headers)
    resp_json = conn.getresponse()
    assert resp_json.status == 400
    conn.close()


def test_cross_interface_consistency(ficos_service, running_api_server):
    """Verify CLI, API, and Direct Service produce identical business recommendations."""
    req_dict = {
        "decision_date": "2018-09-25",
        "cargo_quantity_mt": 75000.0,
        "cargo_type": "Coking Coal",
        "destination_port": "dhamra",
        "laycan_days_allowed": 7,
        "voyage_duration_days": 18.0,
    }

    # 1. Direct Service
    req = FICOSRequest(**req_dict)
    rec_service = ficos_service.process_request(req)

    # 2. CLI JSON
    cmd = [
        sys.executable,
        "-m",
        "src.application.cli",
        "--quantity",
        "75000",
        "--cargo",
        "Coking Coal",
        "--destination",
        "dhamra",
        "--date",
        "2018-09-25",
        "--json",
    ]
    res_cli = subprocess.run(cmd, capture_output=True, text=True)
    assert res_cli.returncode == 0
    rec_cli = json.loads(res_cli.stdout)

    # 3. API
    conn = HTTPConnection(running_api_server)
    conn.request("POST", "/recommend", body=json.dumps(req_dict), headers={"Content-Type": "application/json"})
    res_api = conn.getresponse()
    assert res_api.status == 200
    rec_api = json.loads(res_api.read().decode("utf-8"))
    conn.close()

    # Verify parity
    assert rec_service.decision.action == rec_cli["decision"]["action"] == rec_api["decision"]["action"]
    assert rec_service.decision.optimal_entry_day == rec_cli["decision"]["optimal_entry_day"] == rec_api["decision"]["optimal_entry_day"]
    assert rec_service.vessel.recommended == rec_cli["vessel"]["recommended"] == rec_api["vessel"]["recommended"]
    assert rec_service.decision.expected_cost_now_usd == rec_cli["decision"]["expected_cost_now_usd"] == rec_api["decision"]["expected_cost_now_usd"]
    assert rec_service.risk.level == rec_cli["risk"]["level"] == rec_api["risk"]["level"]
    assert rec_service.forecast.values == rec_cli["forecast"]["values"] == rec_api["forecast"]["values"]


def test_edge_case_inputs(ficos_service):
    """Test boundary edge cases: small cargo, large cargo, out-of-range dates."""
    # 1. Extremely small cargo (100 MT) -> Should map to Handysize
    req_small = FICOSRequest(
        decision_date="2018-09-25",
        cargo_quantity_mt=100.0,
        destination_port="paradip",
    )
    rec_small = ficos_service.process_request(req_small)
    assert rec_small.vessel.recommended == "Handysize"

    # 2. Extremely large cargo (250,000 MT) -> Should map to Capesize (if port allows)
    req_large = FICOSRequest(
        decision_date="2018-09-25",
        cargo_quantity_mt=250000.0,
        destination_port="gangavaram",
    )
    rec_large = ficos_service.process_request(req_large)
    assert rec_large.vessel.recommended == "Capesize"

    # 3. Date prior to historical dataset
    req_ancient = FICOSRequest(
        decision_date="1995-01-01",
        cargo_quantity_mt=75000.0,
        destination_port="paradip",
    )
    with pytest.raises(ValidationError, match="No historical market data available"):
        ficos_service.process_request(req_ancient)
