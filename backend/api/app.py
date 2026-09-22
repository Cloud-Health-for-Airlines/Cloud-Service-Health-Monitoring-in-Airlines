#!/usr/bin/env python3
"""REST API server for BACCP monitoring dashboard and cloud integration.

Exposes generation-typed dependency graph, telemetry, boundary health score,
cascade predictions, circuit-breaker mitigation controls, and AWS cloud status.
Runs with zero external dependencies (Python 3 standard library).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

# Ensure root directory is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.cloud.cloudwatch import CloudWatchPublisher
from backend.cloud.config import config
from backend.cloud.lambda_handler import LambdaMitigationClient, breaker_manager, lambda_handler
from backend.cloud.orchestrator import orchestrator
from backend.cloud.sagemaker import SageMakerCascadePredictor
from backend.cloud.sns import SNSPublisher
from backend.cloud.xray import XRayTraceRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("baccp.api")

# Singletons
cloudwatch = CloudWatchPublisher()
xray = XRayTraceRecorder()
sagemaker = SageMakerCascadePredictor()
lambda_client = LambdaMitigationClient()
sns = SNSPublisher()

# Simulated state for interactive demo & chaos telemetry
simulated_state = {
    "active_fault": None,
    "fault_level": None,
    "fault_start_time": None,
    "base_sync_drift": 12.4,  # baseline healthy drift %
    "current_sync_drift": 12.4,
}


def load_canonical_graph() -> Dict[str, Any]:
    """Load graph from testbed output file or return canonical baseline schema."""
    with xray.trace_operation("graph_retrieval") as _:
        discovery_graph = ROOT / "testbed/results/dependency-graph.json"
        if discovery_graph.is_file():
            try:
                return json.loads(discovery_graph.read_text())
            except Exception as exc:
                logger.warning(f"Could not read {discovery_graph}: {exc}")

        # Canonical 5-node generation-typed graph schema
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "nodes": [
                {"id": "reservations", "node_type": "cloud-native", "domain": "reservations", "port": 8081, "status": "nominal"},
                {"id": "crew", "node_type": "cloud-native", "domain": "crew-scheduling", "port": 8082, "status": "nominal"},
                {"id": "baggage", "node_type": "cloud-native", "domain": "baggage-handling", "port": 8083, "status": "nominal"},
                {"id": "boundary-gateway", "node_type": "boundary-gateway", "domain": "integration", "port": 8084, "status": "monitoring"},
                {"id": "legacy-core", "node_type": "legacy", "domain": "mainframe-cics", "port": 9090, "status": "uninstrumented"},
            ],
            "edges": [
                {
                    "source": "reservations", "destination": "boundary-gateway",
                    "protocol": "tcp", "destination_port": 8080, "observation_count": 1420,
                    "first_seen": now, "last_seen": now, "generation_transition": "cloud-to-gateway",
                    "avg_latency_ms": 18.5,
                },
                {
                    "source": "crew", "destination": "boundary-gateway",
                    "protocol": "tcp", "destination_port": 8080, "observation_count": 890,
                    "first_seen": now, "last_seen": now, "generation_transition": "cloud-to-gateway",
                    "avg_latency_ms": 22.1,
                },
                {
                    "source": "baggage", "destination": "boundary-gateway",
                    "protocol": "tcp", "destination_port": 8080, "observation_count": 1105,
                    "first_seen": now, "last_seen": now, "generation_transition": "cloud-to-gateway",
                    "avg_latency_ms": 15.4,
                },
                {
                    "source": "boundary-gateway", "destination": "legacy-core",
                    "protocol": "tcp", "destination_port": 9090, "observation_count": 3415,
                    "first_seen": now, "last_seen": now, "generation_transition": "gateway-to-legacy",
                    "avg_latency_ms": 68.2,
                },
            ],
        }


def compute_boundary_health() -> Dict[str, Any]:
    """Compute digital-twin sync drift score ε(t) and status."""
    fault = simulated_state["active_fault"]
    level = simulated_state["fault_level"]

    if fault == "network-delay":
        multipliers = {"low": 35.0, "medium": 65.0, "high": 88.0}
        drift = multipliers.get(level, 40.0)
    elif fault == "connection-drop":
        multipliers = {"low": 42.0, "medium": 74.0, "high": 96.0}
        drift = multipliers.get(level, 50.0)
    elif fault == "batch-job-stall":
        multipliers = {"low": 30.0, "medium": 60.0, "high": 92.0}
        drift = multipliers.get(level, 50.0)
    else:
        # Subtle organic baseline
        drift = simulated_state["base_sync_drift"]

    simulated_state["current_sync_drift"] = drift
    cloudwatch.publish_boundary_sync_drift(drift)

    threshold = config.sync_drift_threshold
    status = "HEALTHY"
    if drift >= 70.0:
        status = "CRITICAL"
    elif drift >= threshold:
        status = "DEGRADED"

    return {
        "sync_drift_score": round(drift, 2),
        "threshold": threshold,
        "status": status,
        "active_fault": fault,
        "fault_level": level,
        "formula": "ε(t) = ||Φ(t) - Ψ(t)|| / ||Φ(t)|| * 100",
        "description": "Divergence between cloud gateway transaction rate Φ(t) and legacy mainframe completion rate Ψ(t).",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


class BACCPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler providing REST endpoints and CORS headers."""

    def _set_cors_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_cors_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        with xray.trace_operation("incoming_api_request", {"path": path, "method": "GET"}):
            if path in ("/api/health", "/api/health/boundary"):
                health = compute_boundary_health()
                self._set_cors_headers(200)
                self.wfile.write(json.dumps(health, indent=2).encode("utf-8"))

            elif path == "/api/graph":
                graph = load_canonical_graph()
                self._set_cors_headers(200)
                self.wfile.write(json.dumps(graph, indent=2).encode("utf-8"))

            elif path == "/api/alerts":
                with xray.trace_operation("alert_generation"):
                    health = compute_boundary_health()
                    graph = load_canonical_graph()
                    prediction = sagemaker.predict_cascade(
                        graph_data=graph,
                        sync_drift_score=health["sync_drift_score"],
                        active_fault=simulated_state["active_fault"],
                        fault_level=simulated_state["fault_level"],
                    )
                    cloudwatch.publish_cascade_probability(
                        probability=prediction["cascade_probability"],
                        lead_time_sec=prediction["estimated_lead_time_seconds"],
                        root_node=prediction["predicted_root_cause_node"],
                    )
                    self._set_cors_headers(200)
                    self.wfile.write(json.dumps(prediction, indent=2).encode("utf-8"))

            elif path == "/api/circuit-breaker":
                status = breaker_manager.get_status()
                self._set_cors_headers(200)
                self.wfile.write(json.dumps(status, indent=2).encode("utf-8"))

            elif path == "/api/telemetry":
                telemetry_data = {
                    "window_seconds": 60,
                    "metrics": [
                        {"name": "tcp_connection_attempt", "node": "reservations", "rate_per_sec": 24.5, "unit": "count"},
                        {"name": "tcp_connection_attempt", "node": "crew", "rate_per_sec": 14.8, "unit": "count"},
                        {"name": "tcp_connection_attempt", "node": "baggage", "rate_per_sec": 18.2, "unit": "count"},
                        {"name": "tcp_connection_attempt", "node": "boundary-gateway", "rate_per_sec": 57.5, "unit": "count"},
                    ],
                    "recent_cloudwatch_metrics": cloudwatch.get_recent_metrics(count=10),
                    "recent_xray_traces": xray.get_recent_traces(count=5),
                }
                self._set_cors_headers(200)
                self.wfile.write(json.dumps(telemetry_data, indent=2, default=str).encode("utf-8"))

            elif path == "/api/cloud/status":
                cloud_status = {
                    "mode": config.aws.mode,
                    "cloud_mode": config.aws.cloud_mode,
                    "region": config.aws.region_name,
                    "is_live": config.aws.is_live,
                    "cloudwatch": {
                        "namespace": config.aws.cloudwatch_namespace,
                        "is_live": cloudwatch.is_live,
                        "buffered_metrics_count": len(cloudwatch.published_buffer),
                    },
                    "xray": {
                        "enabled": config.aws.xray_enabled,
                        "daemon_address": config.aws.xray_daemon_address,
                        "is_live": xray.is_live,
                        "recorded_traces_count": len(xray.recorded_traces),
                    },
                    "sagemaker": {
                        "endpoint_name": config.aws.sagemaker_endpoint_name,
                        "is_live": sagemaker.is_live,
                    },
                    "sns": {
                        "topic_arn": config.aws.sns_topic_arn,
                        "is_live": sns.is_live,
                        "dispatched_alerts_count": len(sns.published_alerts),
                    },
                    "lambda": {
                        "function_name": config.aws.lambda_breaker_function,
                        "is_live": lambda_client.is_live,
                        "invocations_count": len(lambda_client.invocation_log),
                    },
                }
                self._set_cors_headers(200)
                self.wfile.write(json.dumps(cloud_status, indent=2).encode("utf-8"))

            elif path == "/metrics":
                health = compute_boundary_health()
                drift = health.get("sync_drift_score", 12.4)
                prob = 0.05
                if simulated_state["active_fault"]:
                    prob = 0.85 if simulated_state["fault_level"] == "critical" else 0.65
                
                prom_lines = [
                    "# HELP baccp_sync_drift_score Digital twin boundary synchronization drift percentage",
                    "# TYPE baccp_sync_drift_score gauge",
                    f"baccp_sync_drift_score {drift:.2f}",
                    "# HELP baccp_cascade_probability Current boundary cascade failure probability",
                    "# TYPE baccp_cascade_probability gauge",
                    f"baccp_cascade_probability {prob:.4f}",
                    "# HELP baccp_circuit_breaker_throttle_rate Gateway load shedding throttle percentage",
                    "# TYPE baccp_circuit_breaker_throttle_rate gauge",
                    f"baccp_circuit_breaker_throttle_rate {breaker_manager.throttle_rate:.2f}",
                    "# HELP baccp_circuit_breaker_state Current state: 0=CLOSED, 1=THROTTLED, 2=OPEN",
                    "# TYPE baccp_circuit_breaker_state gauge",
                    f"baccp_circuit_breaker_state {0 if breaker_manager.state == 'CLOSED' else (1 if breaker_manager.state == 'THROTTLED' else 2)}",
                    "# HELP baccp_telemetry_requests_total Total API requests received",
                    "# TYPE baccp_telemetry_requests_total counter",
                    "baccp_telemetry_requests_total 42",
                ]
                prom_body = "\n".join(prom_lines) + "\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(prom_body.encode("utf-8"))

            else:
                self._set_cors_headers(404)
                self.wfile.write(json.dumps({"error": f"Endpoint '{path}' not found"}).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(post_data.decode("utf-8"))
        except Exception:
            body = {}

        with xray.trace_operation("incoming_api_request", {"path": path, "method": "POST"}):
            if path in ("/api/circuit-breaker", "/api/circuit-breaker/action"):
                action = body.get("action", "THROTTLE").upper()
                throttle_rate = float(body.get("throttle_rate", 0.5))
                reason = body.get("reason", "Operator mitigation request from dashboard")
                affected_svc = body.get("affected_service", "reservations")
                gateway = body.get("gateway", breaker_manager.target_node)

                # Invoke mitigation through Lambda integration client
                result = lambda_client.invoke_mitigation(
                    affected_service=affected_svc,
                    gateway=gateway,
                    cascade_probability=body.get("cascade_probability", 0.65),
                    severity=body.get("severity", "HIGH"),
                    recommended_action=action,
                    sync_drift_score=float(body.get("boundary_sync_drift", simulated_state["current_sync_drift"])),
                    lead_time=float(body.get("lead_time", 120.0)),
                )

                self._set_cors_headers(200)
                self.wfile.write(json.dumps({
                    "status": "applied",
                    "action_result": result,
                    "current_state": breaker_manager.get_status(),
                }, indent=2).encode("utf-8"))

            elif path == "/api/predict":
                with xray.trace_operation("prediction_request"):
                    graph = body.get("graph") or load_canonical_graph()
                    drift = float(body.get("sync_drift_score", simulated_state["current_sync_drift"]))
                    telemetry = body.get("telemetry_features")
                    boundary = body.get("boundary_features")
                    
                    prediction = sagemaker.predict_cascade(
                        graph_data=graph,
                        sync_drift_score=drift,
                        telemetry_features=telemetry,
                        boundary_features=boundary,
                        active_fault=body.get("active_fault", simulated_state["active_fault"]),
                        fault_level=body.get("fault_level", simulated_state["fault_level"]),
                    )
                    self._set_cors_headers(200)
                    self.wfile.write(json.dumps(prediction, indent=2).encode("utf-8"))

            elif path in ("/api/pipeline/run", "/api/orchestrate"):
                trace_report = orchestrator.run_pipeline(
                    telemetry=body.get("telemetry"),
                    graph=body.get("graph") or load_canonical_graph(),
                    sync_drift_score=body.get("sync_drift_score", simulated_state["current_sync_drift"]),
                    active_fault=body.get("active_fault", simulated_state["active_fault"]),
                    fault_level=body.get("fault_level", simulated_state["fault_level"]),
                )
                self._set_cors_headers(200)
                self.wfile.write(json.dumps(trace_report, indent=2).encode("utf-8"))

            elif path == "/api/simulate/chaos":
                fault = body.get("fault")
                level = body.get("level", "medium")
                
                if fault in ("network-delay", "connection-drop", "batch-job-stall"):
                    simulated_state["active_fault"] = fault
                    simulated_state["fault_level"] = level
                    simulated_state["fault_start_time"] = time.time()
                    message = f"Simulated {fault} ({level}) applied to gateway."
                elif fault == "clear":
                    simulated_state["active_fault"] = None
                    simulated_state["fault_level"] = None
                    simulated_state["fault_start_time"] = None
                    message = "Simulated fault cleared. Returning to nominal baseline."
                else:
                    self._set_cors_headers(400)
                    self.wfile.write(json.dumps({"error": "Invalid fault type. Use network-delay, connection-drop, batch-job-stall, or clear."}).encode("utf-8"))
                    return

                health = compute_boundary_health()
                self._set_cors_headers(200)
                self.wfile.write(json.dumps({
                    "message": message,
                    "state": simulated_state,
                    "updated_health": health,
                }, indent=2).encode("utf-8"))

            elif path == "/api/simulate/trace":
                service = body.get("service", "reservations")
                duration = float(body.get("duration_ms", 35.0))
                trace = xray.record_cross_boundary_trace(
                    caller_service=service,
                    duration_ms=duration,
                    fault_injected=simulated_state["active_fault"],
                    status_code=502 if simulated_state["active_fault"] else 200,
                )
                self._set_cors_headers(200)
                self.wfile.write(json.dumps(trace, indent=2).encode("utf-8"))

            elif path in ("/api/circuit-breaker/multi-agent", "/api/circuit-breaker/mappo"):
                try:
                    from ai_models.circuit_breaker import MAPPOAgent
                    mappo = MAPPOAgent(num_gateways=2)
                    gw_states = body.get("gateways", {
                        "boundary-gateway-jfk": {
                            "cascade_probability": body.get("cascade_probability", 0.75),
                            "boundary_sync_drift": simulated_state["current_sync_drift"],
                            "gateway_latency_ms": 250.0,
                            "gateway_error_rate": 0.05,
                            "current_throttle_rate": 0.0,
                        },
                        "boundary-gateway-lhr": {
                            "cascade_probability": 0.40,
                            "boundary_sync_drift": 22.0,
                            "gateway_latency_ms": 45.0,
                            "gateway_error_rate": 0.0,
                            "current_throttle_rate": 0.0,
                        },
                    })
                    result = mappo.coordinate_mitigation(
                        gateway_states=gw_states,
                        shared_mainframe_queue=float(body.get("mainframe_queue", 45.0)),
                        shared_mainframe_cpu=float(body.get("mainframe_cpu", 68.0)),
                    )
                    self._set_cors_headers(200)
                    self.wfile.write(json.dumps({
                        "mode": "Tier-2A-MAPPO-Multi-Gateway",
                        "coordinated_actions": result,
                    }, indent=2).encode("utf-8"))
                except Exception as exc:
                    self._set_cors_headers(500)
                    self.wfile.write(json.dumps({"error": f"MAPPO coordinator error: {exc}"}).encode("utf-8"))

            else:
                self._set_cors_headers(404)
                self.wfile.write(json.dumps({"error": f"Endpoint '{path}' not found"}).encode("utf-8"))

    def log_message(self, format, *args):
        logger.debug("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))


def run_server(host: str = "0.0.0.0", port: int = 8000):
    server_address = (host, port)
    httpd = HTTPServer(server_address, BACCPRequestHandler)
    print(f"BACCP API Server running on http://{host}:{port} (Cloud mode: {config.aws.cloud_mode})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BACCP API Server...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BACCP Backend API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(args.host, args.port)
