"""BACCP Cloud Orchestrator service coordinating cross-generation cloud telemetry and mitigation.

Decoupled pipeline flow:
Telemetry
   ↓
CloudWatch / X-Ray
   ↓
Prediction Adapter (SageMaker / Local RGCN)
   ↓
Risk Evaluation
   ↓
Lambda Mitigation (Circuit Breaker)
   ↓
SNS Alert Dispatcher
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Dict, Optional

from .cloudwatch import CloudWatchPublisher
from .config import AWSConfig, config
from .lambda_handler import LambdaMitigationClient, breaker_manager
from .sagemaker import SageMakerCascadePredictor
from .sns import SNSPublisher
from .xray import XRayTraceRecorder

logger = logging.getLogger("baccp.orchestrator")


class BACCPCloudOrchestrator:
    """Coordinates cloud observability, prediction, mitigation, and notifications."""

    def __init__(
        self,
        cloudwatch: Optional[CloudWatchPublisher] = None,
        xray: Optional[XRayTraceRecorder] = None,
        sagemaker: Optional[SageMakerCascadePredictor] = None,
        lambda_client: Optional[LambdaMitigationClient] = None,
        sns: Optional[SNSPublisher] = None,
        aws_config: Optional[AWSConfig] = None,
    ):
        self.config = aws_config or config.aws
        self.cloudwatch = cloudwatch or CloudWatchPublisher(self.config)
        self.xray = xray or XRayTraceRecorder(self.config)
        self.sagemaker = sagemaker or SageMakerCascadePredictor(self.config)
        self.lambda_client = lambda_client or LambdaMitigationClient(self.config)
        self.sns = sns or SNSPublisher(self.config)

    def run_pipeline(
        self,
        telemetry: Optional[Dict[str, Any]] = None,
        graph: Optional[Dict[str, Any]] = None,
        sync_drift_score: Optional[float] = None,
        active_fault: Optional[str] = None,
        fault_level: Optional[str] = None,
        gateway_node: str = "boundary-gateway",
    ) -> Dict[str, Any]:
        """Execute the coordinated BACCP cloud pipeline across all stages."""
        pipeline_start = time.time()
        execution_trace = {}

        # ----------------------------------------------------------------------
        # Stage 1: Telemetry Ingestion & Extraction
        # ----------------------------------------------------------------------
        with self.xray.trace_operation("telemetry_processing", {"gateway": gateway_node}) as segment:
            drift = 12.4 if sync_drift_score is None else float(sync_drift_score)
            service_telemetry = telemetry or {
                "reservations": {"latency_ms": 18.5, "error_rate": 0.2, "request_rate": 24.5},
                "crew": {"latency_ms": 22.1, "error_rate": 0.1, "request_rate": 14.8},
                "baggage": {"latency_ms": 15.4, "error_rate": 0.3, "request_rate": 18.2},
                "boundary-gateway": {"latency_ms": 45.0, "error_rate": 0.5, "request_rate": 57.5},
                "legacy-core": {"latency_ms": 68.2, "error_rate": 0.0, "request_rate": 57.5},
            }
            execution_trace["telemetry"] = {
                "sync_drift_score": drift,
                "services_reported": list(service_telemetry.keys()),
            }

        # ----------------------------------------------------------------------
        # Stage 2: CloudWatch Metrics & X-Ray Distributed Trace
        # ----------------------------------------------------------------------
        self.cloudwatch.publish_boundary_sync_drift(drift, gateway_node)
        for svc_name, metrics in service_telemetry.items():
            if isinstance(metrics, dict):
                self.cloudwatch.publish_service_telemetry(
                    service=svc_name,
                    latency_ms=float(metrics.get("latency_ms", 20.0)),
                    error_rate=float(metrics.get("error_rate", 0.0)),
                    request_rate=float(metrics.get("request_rate", 10.0)),
                )
        
        # Emit synthetic distributed trace across boundary
        trace_record = self.xray.record_cross_boundary_trace(
            caller_service="reservations",
            gateway_service=gateway_node,
            legacy_service="legacy-core",
            fault_injected=active_fault,
        )
        execution_trace["distributed_trace_id"] = trace_record.get("trace_id")

        # ----------------------------------------------------------------------
        # Stage 3: SageMaker Prediction Adapter
        # ----------------------------------------------------------------------
        with self.xray.trace_operation("prediction_request", {"sync_drift": drift}) as segment:
            prediction = self.sagemaker.predict_cascade(
                graph_data=graph,
                sync_drift_score=drift,
                telemetry_features=service_telemetry,
                boundary_features={"sync_drift_score": drift, "gateway": gateway_node},
                active_fault=active_fault,
                fault_level=fault_level,
            )
            prob = prediction["cascade_probability"]
            lead_time = prediction["estimated_lead_time_seconds"]
            root_node = prediction.get("predicted_root_cause_node", gateway_node)
            
            # Record predictions in CloudWatch
            self.cloudwatch.publish_cascade_probability(prob, lead_time, root_node)
            execution_trace["prediction"] = prediction

        # ----------------------------------------------------------------------
        # Stage 4: Risk Evaluation
        # ----------------------------------------------------------------------
        severity = prediction.get("severity", "LOW")
        is_critical = prob >= 0.75 or drift >= 70.0
        is_high = prob >= 0.55 or drift >= 45.0
        
        if is_critical:
            recommended_action = "OPEN"
        elif is_high:
            recommended_action = "THROTTLE"
        else:
            recommended_action = "CLOSED"

        execution_trace["risk_evaluation"] = {
            "severity": severity,
            "cascade_probability": prob,
            "is_critical": is_critical,
            "is_high": is_high,
            "recommended_action": recommended_action,
        }

        # ----------------------------------------------------------------------
        # Stage 5: Lambda Mitigation (Circuit Breaker)
        # ----------------------------------------------------------------------
        mitigation_result = None
        if is_high or is_critical:
            with self.xray.trace_operation("circuit_breaker_action", {"action": recommended_action}) as segment:
                mitigation_result = self.lambda_client.invoke_mitigation(
                    affected_service="reservations",
                    gateway=gateway_node,
                    cascade_probability=prob,
                    severity=severity,
                    recommended_action=recommended_action,
                    sync_drift_score=drift,
                    lead_time=lead_time,
                )
                execution_trace["mitigation"] = mitigation_result
        else:
            # Check if breaker should reset to closed
            if breaker_manager.state != "CLOSED":
                breaker_manager.apply_action("CLOSED", 0.0, "Risk subsided to nominal", caller="orchestrator")
                execution_trace["mitigation"] = {"action": "RESET", "state": "CLOSED"}

        # ----------------------------------------------------------------------
        # Stage 6: SNS Alert Dispatch
        # ----------------------------------------------------------------------
        sns_alert = None
        if is_high or is_critical:
            with self.xray.trace_operation("alert_generation", {"severity": severity}) as segment:
                sns_alert = self.sns.publish_cascade_alert(
                    severity=severity,
                    cascade_probability=prob,
                    affected_service="reservations",
                    gateway=gateway_node,
                    predicted_failure_location=root_node,
                    lead_time=lead_time,
                    mitigation_status=recommended_action,
                    extra_details=prediction.get("explanation"),
                )
                self.cloudwatch.publish_active_alerts(1)
                execution_trace["sns_alert"] = sns_alert
        else:
            self.cloudwatch.publish_active_alerts(0)

        pipeline_duration = round((time.time() - pipeline_start) * 1000.0, 2)
        execution_trace["duration_ms"] = pipeline_duration
        execution_trace["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        execution_trace["status"] = "success"

        return execution_trace


# Global default orchestrator instance
orchestrator = BACCPCloudOrchestrator()
