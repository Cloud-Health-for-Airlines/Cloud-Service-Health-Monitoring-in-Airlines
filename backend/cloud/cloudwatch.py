"""Amazon CloudWatch telemetry metrics publisher for BACCP.

Publishes boundary health metrics, cascade probabilities, latency, error rates,
circuit breaker actuation rates, and active alert counts to CloudWatch Metric Data.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

from .config import AWSConfig, config

logger = logging.getLogger("baccp.cloudwatch")


class CloudWatchPublisher:
    """Publishes operational and predictive metrics to Amazon CloudWatch.
    
    Provides a decoupled provider interface so caller services do not depend on the boto3 SDK.
    """

    METRIC_UNIT_MAP = {
        "boundary_health_score": "Percent",
        "cascade_probability": "None",
        "prediction_lead_time": "Seconds",
        "service_latency": "Milliseconds",
        "error_rate": "Percent",
        "request_rate": "Count/Second",
        "circuit_breaker_actions": "Count",
        "active_alerts": "Count",
        # Legacy/alternate naming
        "BoundarySyncDrift": "Percent",
        "CascadeProbability": "None",
        "EstimatedLeadTime": "Seconds",
        "CircuitBreakerThrottleRate": "Percent",
    }

    def __init__(self, aws_config: Optional[AWSConfig] = None):
        self.config = aws_config or config.aws
        self.namespace = self.config.cloudwatch_namespace
        self._client = None
        self.published_buffer: List[Dict[str, Any]] = []

        if self.config.is_live:
            try:
                import boto3  # type: ignore
                kwargs = {"region_name": self.config.region_name}
                if self.config.access_key_id and self.config.secret_access_key:
                    kwargs["aws_access_key_id"] = self.config.access_key_id
                    kwargs["aws_secret_access_key"] = self.config.secret_access_key
                self._client = boto3.client("cloudwatch", **kwargs)
                logger.info(f"Initialized live CloudWatch client for region '{self.config.region_name}'")
            except Exception as exc:
                logger.warning(f"Could not connect to live CloudWatch: {exc}. Operating in local mock mode.")
                self._client = None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def put_metric_data(self, metrics: List[Dict[str, Any]]) -> bool:
        """Publish a batch of metric data items to CloudWatch.
        
        Each item should contain: MetricName, Value, Unit (optional), Dimensions (optional).
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        formatted_metrics = []
        for m in metrics:
            metric_item = {
                "MetricName": m["MetricName"],
                "Value": float(m["Value"]),
                "Unit": m.get("Unit", self.METRIC_UNIT_MAP.get(m["MetricName"], "None")),
                "Timestamp": m.get("Timestamp", now),
            }
            if "Dimensions" in m:
                metric_item["Dimensions"] = m["Dimensions"]
            formatted_metrics.append(metric_item)

        self.published_buffer.extend(formatted_metrics)
        # Keep buffer bounded for memory safety
        if len(self.published_buffer) > 500:
            self.published_buffer = self.published_buffer[-500:]

        if self.is_live:
            try:
                # CloudWatch accepts max 20 metrics per call
                for i in range(0, len(formatted_metrics), 20):
                    batch = formatted_metrics[i:i + 20]
                    self._client.put_metric_data(
                        Namespace=self.namespace,
                        MetricData=batch,
                    )
                logger.info(f"Published {len(formatted_metrics)} metrics to CloudWatch namespace '{self.namespace}'")
                return True
            except Exception as exc:
                logger.error(f"Error publishing metrics to CloudWatch: {exc}")
                return False
        else:
            logger.debug(f"[LOCAL CloudWatch] Buffered {len(formatted_metrics)} metrics in namespace '{self.namespace}'")
            return True

    def publish_baccp_metrics(self, metrics: Dict[str, float], dimensions: Optional[Dict[str, str]] = None) -> bool:
        """Publish a dictionary of BACCP operational and predictive metrics.
        
        Supported keys:
        - boundary_health_score
        - cascade_probability
        - prediction_lead_time
        - service_latency
        - error_rate
        - request_rate
        - circuit_breaker_actions
        - active_alerts
        """
        formatted = []
        dims_list = [{"Name": k, "Value": str(v)} for k, v in (dimensions or {}).items()]
        for metric_name, value in metrics.items():
            item = {
                "MetricName": metric_name,
                "Value": float(value),
                "Unit": self.METRIC_UNIT_MAP.get(metric_name, "None"),
            }
            if dims_list:
                item["Dimensions"] = dims_list
            formatted.append(item)

        return self.put_metric_data(formatted)

    def publish_boundary_sync_drift(self, drift_score: float, gateway_node: str = "boundary-gateway") -> bool:
        """Record the state synchronization drift metric ε(t)."""
        return self.put_metric_data([
            {
                "MetricName": "boundary_health_score",
                "Value": drift_score,
                "Unit": "Percent",
                "Dimensions": [{"Name": "GatewayNode", "Value": gateway_node}],
            },
            {
                "MetricName": "BoundarySyncDrift",
                "Value": drift_score,
                "Unit": "Percent",
                "Dimensions": [{"Name": "GatewayNode", "Value": gateway_node}],
            },
        ])

    def publish_cascade_probability(self, probability: float, lead_time_sec: float, root_node: str) -> bool:
        """Record model cascade prediction probability and lead time."""
        return self.put_metric_data([
            {
                "MetricName": "cascade_probability",
                "Value": probability,
                "Unit": "None",
                "Dimensions": [{"Name": "RootCauseNode", "Value": root_node}],
            },
            {
                "MetricName": "CascadeProbability",
                "Value": probability,
                "Unit": "None",
                "Dimensions": [{"Name": "RootCauseNode", "Value": root_node}],
            },
            {
                "MetricName": "prediction_lead_time",
                "Value": lead_time_sec,
                "Unit": "Seconds",
                "Dimensions": [{"Name": "RootCauseNode", "Value": root_node}],
            },
            {
                "MetricName": "EstimatedLeadTime",
                "Value": lead_time_sec,
                "Unit": "Seconds",
                "Dimensions": [{"Name": "RootCauseNode", "Value": root_node}],
            },
        ])

    def publish_service_telemetry(self, service: str, latency_ms: float, error_rate: float, request_rate: float) -> bool:
        """Record per-service latency, error rate, and request rate."""
        dims = [{"Name": "Service", "Value": service}]
        return self.put_metric_data([
            {"MetricName": "service_latency", "Value": latency_ms, "Unit": "Milliseconds", "Dimensions": dims},
            {"MetricName": "error_rate", "Value": error_rate, "Unit": "Percent", "Dimensions": dims},
            {"MetricName": "request_rate", "Value": request_rate, "Unit": "Count/Second", "Dimensions": dims},
        ])

    def publish_circuit_breaker_action(self, action: str, throttle_rate: float, gateway_node: str = "boundary-gateway") -> bool:
        """Record circuit breaker mitigation events."""
        dims = [
            {"Name": "Action", "Value": action},
            {"Name": "GatewayNode", "Value": gateway_node},
        ]
        return self.put_metric_data([
            {"MetricName": "circuit_breaker_actions", "Value": 1.0, "Unit": "Count", "Dimensions": dims},
            {"MetricName": "CircuitBreakerThrottleRate", "Value": throttle_rate, "Unit": "Percent", "Dimensions": dims},
        ])

    def publish_active_alerts(self, count: int) -> bool:
        """Record current number of active predictive alerts."""
        return self.put_metric_data([{
            "MetricName": "active_alerts",
            "Value": float(count),
            "Unit": "Count",
        }])

    def get_recent_metrics(self, count: int = 50) -> List[Dict[str, Any]]:
        """Return recently recorded metric events from buffer with serializable timestamps."""
        serialized = []
        for m in self.published_buffer[-count:]:
            item = dict(m)
            ts = item.get("Timestamp")
            if hasattr(ts, "isoformat"):
                item["Timestamp"] = ts.isoformat()
            elif ts is not None:
                item["Timestamp"] = str(ts)
            serialized.append(item)
        return serialized
