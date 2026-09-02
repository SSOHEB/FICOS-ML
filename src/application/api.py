"""Lightweight HTTP API service for FICOS charter decision support."""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from typing import Dict, Any

from src.application.schemas import FICOSRequest, ValidationError
from src.application.service import FICOSService


class FICOSRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler providing /health and /recommend REST endpoints."""

    service = None

    @classmethod
    def get_service(cls) -> FICOSService:
        if cls.service is None:
            cls.service = FICOSService()
        return cls.service

    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        response_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            self._send_json_response(200, {
                "status": "HEALTHY",
                "system": "FICOS ML Charter Decision Support Engine",
                "version": "1.0.0",
                "champion_model": "Ridge_a1.0",
            })
        else:
            self._send_json_response(404, {"error": f"Endpoint '{parsed.path}' not found."})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/recommend":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len)

            try:
                payload = json.loads(post_body.decode("utf-8"))
                req = FICOSRequest(
                    decision_date=payload.get("decision_date"),
                    cargo_quantity_mt=float(payload.get("cargo_quantity_mt", 0.0)),
                    cargo_type=payload.get("cargo_type", "Coking Coal"),
                    destination_port=payload.get("destination_port", "paradip"),
                    current_freight=payload.get("current_freight"),
                    origin=payload.get("origin", "Gladstone, Australia"),
                    laycan_days_allowed=int(payload.get("laycan_days_allowed", 7)),
                    voyage_duration_days=float(payload.get("voyage_duration_days", 18.0)),
                    preferred_vessel=payload.get("preferred_vessel"),
                )

                service = self.get_service()
                rec = service.process_request(req)
                self._send_json_response(200, rec.to_dict())

            except json.JSONDecodeError as e:
                self._send_json_response(400, {"error": "Malformed JSON", "detail": str(e)})
            except (ValidationError, ValueError, TypeError) as e:
                self._send_json_response(400, {"error": "Validation Error", "detail": str(e)})
            except Exception as e:
                self._send_json_response(500, {"error": "Internal Processing Error", "detail": str(e)})
        else:
            self._send_json_response(404, {"error": f"Endpoint '{parsed.path}' not found."})


def create_server(host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    """Instantiate HTTP REST server for FICOS."""
    server = HTTPServer((host, port), FICOSRequestHandler)
    return server
