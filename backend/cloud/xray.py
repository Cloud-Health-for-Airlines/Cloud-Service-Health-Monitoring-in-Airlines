"""AWS X-Ray tracing integration for cross-generation requests.

Traces distributed requests traversing from cloud-native microservices through
the integration boundary gateway down to the un-instrumented legacy core.
Provides context managers and trace capture around key backend operations:
- incoming API request
- telemetry processing
- graph retrieval
- prediction request
- alert generation
- circuit-breaker action

Operates in safe optional mode locally and when the X-Ray daemon is unavailable.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import random
import socket
import time
from typing import Any, Dict, Generator, List, Optional

from .config import AWSConfig, config

logger = logging.getLogger("baccp.xray")


def generate_trace_id() -> str:
    """Generate AWS X-Ray compliant trace ID: 1-{8 hex digits time}-{24 hex digits random}."""
    epoch_hex = hex(int(time.time()))[2:].zfill(8)
    random_hex = hex(random.getrandbits(96))[2:].zfill(24)
    return f"1-{epoch_hex}-{random_hex}"


def generate_segment_id() -> str:
    """Generate 16-hex digit segment ID."""
    return hex(random.getrandbits(64))[2:].zfill(16)


class XRayTraceRecorder:
    """Records distributed trace segments across the legacy/cloud boundary."""

    VALID_OPERATIONS = {
        "incoming_api_request",
        "telemetry_processing",
        "graph_retrieval",
        "prediction_request",
        "alert_generation",
        "circuit_breaker_action",
    }

    def __init__(self, aws_config: Optional[AWSConfig] = None):
        self.config = aws_config or config.aws
        self.enabled = self.config.xray_enabled
        self.daemon_address = self.config.xray_daemon_address
        self._sock = None
        self.recorded_traces: List[Dict[str, Any]] = []

        if self.enabled and self.config.is_live:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                logger.info(f"Initialized UDP socket for X-Ray daemon at {self.daemon_address}")
            except Exception as exc:
                logger.warning(f"Could not initialize X-Ray UDP socket: {exc}. Operating in local buffer mode.")
                self._sock = None

    @property
    def is_live(self) -> bool:
        return self._sock is not None

    def emit_segment(self, segment: Dict[str, Any]) -> bool:
        """Send segment JSON to X-Ray daemon over UDP (live) or record in buffer (local)."""
        self.recorded_traces.append(segment)
        if len(self.recorded_traces) > 200:
            self.recorded_traces = self.recorded_traces[-200:]

        if self.is_live and self._sock:
            try:
                host, port_str = self.daemon_address.split(":")
                payload = '{"format": "json", "version": 1}\n' + json.dumps(segment)
                self._sock.sendto(payload.encode("utf-8"), (host, int(port_str)))
                return True
            except Exception as exc:
                logger.debug(f"Failed to emit segment to X-Ray daemon: {exc}")
                return False
        return True

    @contextlib.contextmanager
    def trace_operation(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Context manager tracing execution of a named pipeline or API operation.
        
        Supported operations include:
        - incoming_api_request
        - telemetry_processing
        - graph_retrieval
        - prediction_request
        - alert_generation
        - circuit_breaker_action
        """
        start_time = time.time()
        trace_id = generate_trace_id()
        segment_id = generate_segment_id()
        segment_data: Dict[str, Any] = {
            "name": f"baccp:{operation_name}",
            "id": segment_id,
            "trace_id": trace_id,
            "start_time": start_time,
            "annotations": {
                "operation": operation_name,
                "environment": "local" if not self.is_live else "aws",
                **(annotations or {}),
            },
            "metadata": metadata or {},
            "status": "in_progress",
        }

        try:
            yield segment_data
            segment_data["status"] = "ok"
        except Exception as exc:
            segment_data["status"] = "error"
            segment_data["error"] = True
            segment_data["cause"] = {"exceptions": [{"message": str(exc), "type": type(exc).__name__}]}
            raise
        finally:
            end_time = time.time()
            segment_data["end_time"] = end_time
            segment_data["duration_ms"] = round((end_time - start_time) * 1000.0, 2)
            self.emit_segment(segment_data)

    def record_cross_boundary_trace(
        self,
        caller_service: str,
        gateway_service: str = "boundary-gateway",
        legacy_service: str = "legacy-core",
        duration_ms: float = 45.0,
        fault_injected: Optional[str] = None,
        status_code: int = 200,
    ) -> Dict[str, Any]:
        """Synthesizes a complete cross-generation end-to-end trace spanning cloud to legacy."""
        trace_id = generate_trace_id()
        root_segment_id = generate_segment_id()
        now = time.time()
        start_time = now - (duration_ms / 1000.0)

        segment = {
            "name": caller_service,
            "id": root_segment_id,
            "trace_id": trace_id,
            "start_time": start_time,
            "end_time": now,
            "annotations": {
                "generation_type": "cloud-native",
                "service_domain": caller_service,
                "environment": "local-testbed" if not self.is_live else "aws-ecs",
            },
            "subsegments": [
                {
                    "name": gateway_service,
                    "id": generate_segment_id(),
                    "start_time": start_time + 0.005,
                    "end_time": now,
                    "annotations": {
                        "generation_type": "boundary-gateway",
                        "protocol_transition": "HTTP-to-TCP",
                        "fault_active": bool(fault_injected),
                    },
                    "metadata": {
                        "fault": fault_injected,
                        "status_code": status_code,
                    },
                    "subsegments": [
                        {
                            "name": legacy_service,
                            "id": generate_segment_id(),
                            "start_time": start_time + 0.015,
                            "end_time": now - 0.002,
                            "annotations": {
                                "generation_type": "legacy",
                                "port": 9090,
                                "uninstrumentable": True,
                                "observed_via": "eBPF",
                            },
                        }
                    ],
                }
            ],
            "http": {
                "response": {
                    "status": status_code,
                }
            },
        }

        self.emit_segment(segment)
        return segment

    def get_recent_traces(self, count: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent traces."""
        return self.recorded_traces[-count:]
