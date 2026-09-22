"""AWS Lambda integration and execution handler for automated boundary circuit breaking.

Provides:
1. CircuitBreakerManager: Runtime state tracker for boundary rate-limiting.
2. lambda_handler: Handler function executing mitigation policies (THROTTLE, ISOLATE, RESET).
3. LambdaMitigationClient: Client invoking the configured AWS Lambda function in AWS mode,
   or simulating execution locally without altering production airline traffic.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from typing import Any, Dict, List, Optional

from .cloudwatch import CloudWatchPublisher
from .config import AWSConfig, config
from .sns import SNSPublisher

logger = logging.getLogger("baccp.lambda")


class CircuitBreakerManager:
    """Manages the current runtime state of the boundary circuit breaker."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.state = "CLOSED"  # CLOSED (healthy), THROTTLED, OPEN (isolated)
            cls._instance.throttle_rate = 0.0  # 0.0 (no throttle) to 1.0 (full isolation)
            cls._instance.target_node = "boundary-gateway"
            cls._instance.last_updated = time.time()
            cls._instance.action_history = []
        return cls._instance

    def apply_action(self, action: str, throttle_rate: float, reason: str, caller: str = "lambda-automated") -> Dict[str, Any]:
        """Apply a mitigation action and record history."""
        self.state = action
        self.throttle_rate = max(0.0, min(1.0, throttle_rate))
        self.last_updated = time.time()
        
        record = {
            "timestamp": self.last_updated,
            "iso_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action": action,
            "throttle_rate": self.throttle_rate,
            "target_node": self.target_node,
            "reason": reason,
            "invoked_by": caller,
        }
        self.action_history.append(record)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]

        logger.info(f"Circuit breaker updated: {action} (rate={self.throttle_rate:.0%}) for {self.target_node}: {reason}")
        return record

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "throttle_rate": self.throttle_rate,
            "target_node": self.target_node,
            "last_updated": self.last_updated,
            "history_count": len(self.action_history),
            "recent_actions": self.action_history[-10:],
        }


breaker_manager = CircuitBreakerManager()


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Standard AWS Lambda entrypoint for cascade mitigation."""
    logger.info(f"Lambda circuit breaker invoked with event: {json.dumps(event)}")
    
    cascade_prob = float(event.get("cascade_probability", 0.0))
    drift_score = float(event.get("boundary_sync_drift", 0.0))
    source = event.get("event_type", event.get("source", "CloudWatchAlarm"))
    rec_action = event.get("recommended_action")

    cw = CloudWatchPublisher()
    sns = SNSPublisher()

    # Determine mitigation action
    if rec_action in ("OPEN", "ISOLATE"):
        action = "OPEN"
        throttle_rate = 1.0
        reason = f"Critical cascade risk (P={cascade_prob:.2f}, drift={drift_score:.1f}%). Boundary isolation actuated."
    elif rec_action in ("THROTTLE", "THROTTLED"):
        action = "THROTTLED"
        throttle_rate = round(min(0.75, 0.30 + (cascade_prob - 0.55) * 2.0), 2)
        reason = f"Elevated cascade risk (P={cascade_prob:.2f}). Throttling boundary requests by {throttle_rate:.0%}."
    else:
        # Attempt learned RL policy (Tier 1F PPO)
        applied_rl = False
        try:
            from pathlib import Path
            import sys
            repo_root = Path(__file__).resolve().parent.parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from importlib import import_module
            breaker_mod = import_module("ai-models.circuit-breaker.inference")
            action, throttle_rate, reason = breaker_mod.choose_action(
                cascade_probability=cascade_prob,
                boundary_sync_drift=drift_score,
                state=event,
            )
            applied_rl = True
        except Exception as exc:
            logger.debug(f"Learned RL circuit breaker unavailable ({exc}). Using threshold ladder.")

        if not applied_rl:
            if cascade_prob >= 0.80 or drift_score >= 70.0:
                action = "OPEN"
                throttle_rate = 1.0
                reason = f"Critical cascade risk (P={cascade_prob:.2f}, drift={drift_score:.1f}%). Boundary isolation actuated."
            elif cascade_prob >= 0.55 or drift_score >= 45.0:
                action = "THROTTLED"
                throttle_rate = round(min(0.75, 0.30 + (cascade_prob - 0.55) * 2.0), 2)
                reason = f"Elevated cascade risk (P={cascade_prob:.2f}). Throttling boundary requests by {throttle_rate:.0%}."
            else:
                action = "CLOSED"
                throttle_rate = 0.0
                reason = "System within nominal health boundaries. Resetting circuit breaker."

    record = breaker_manager.apply_action(action, throttle_rate, reason, caller=f"lambda:{source}")
    cw.publish_circuit_breaker_action(action, throttle_rate * 100.0, breaker_manager.target_node)

    # Trigger high-priority alert notification
    if action in ("THROTTLED", "OPEN"):
        sns.publish_cascade_alert(
            severity="CRITICAL" if action == "OPEN" else "HIGH",
            cascade_probability=cascade_prob,
            affected_service=event.get("affected_service", "reservations"),
            gateway=event.get("gateway", breaker_manager.target_node),
            predicted_failure_location=event.get("gateway", breaker_manager.target_node),
            lead_time=float(event.get("lead_time", 60.0)),
            mitigation_status=action,
        )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "status": "success",
            "action_applied": record,
            "breaker_status": breaker_manager.get_status(),
        }),
    }


class LambdaMitigationClient:
    """Client for invoking the circuit breaker Lambda in AWS or simulating locally."""

    def __init__(self, aws_config: Optional[AWSConfig] = None):
        self.config = aws_config or config.aws
        self.function_name = self.config.lambda_breaker_function
        self._client = None
        self.invocation_log: List[Dict[str, Any]] = []

        if self.config.is_live:
            try:
                import boto3  # type: ignore
                kwargs = {"region_name": self.config.region_name}
                if self.config.access_key_id and self.config.secret_access_key:
                    kwargs["aws_access_key_id"] = self.config.access_key_id
                    kwargs["aws_secret_access_key"] = self.config.secret_access_key
                self._client = boto3.client("lambda", **kwargs)
                logger.info(f"Initialized live Lambda client for function '{self.function_name}'")
            except Exception as exc:
                logger.warning(f"Could not connect to live AWS Lambda: {exc}. Operating in local simulation mode.")
                self._client = None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def invoke_mitigation(
        self,
        affected_service: str,
        gateway: str = "boundary-gateway",
        cascade_probability: float = 0.0,
        severity: str = "LOW",
        recommended_action: str = "THROTTLE",
        sync_drift_score: float = 0.0,
        lead_time: float = 300.0,
    ) -> Dict[str, Any]:
        """Invoke the mitigation Lambda function with the required structured payload.
        
        Payload format:
        {
          "event_type": "cascade_mitigation",
          "timestamp": "<iso>",
          "affected_service": "<service>",
          "gateway": "<gateway>",
          "cascade_probability": 0.0,
          "severity": "<severity>",
          "recommended_action": "<action>"
        }
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = {
            "event_type": "cascade_mitigation",
            "timestamp": now,
            "affected_service": affected_service,
            "gateway": gateway,
            "cascade_probability": round(float(cascade_probability), 4),
            "severity": severity,
            "recommended_action": recommended_action,
            "boundary_sync_drift": round(float(sync_drift_score), 2),
            "lead_time": round(float(lead_time), 1),
        }

        log_entry = {
            "timestamp": now,
            "payload": payload,
            "is_live": self.is_live,
        }

        if self.is_live and self._client:
            try:
                response = self._client.invoke(
                    FunctionName=self.function_name,
                    InvocationType="RequestResponse",
                    Payload=json.dumps(payload),
                )
                response_payload = json.loads(response["Payload"].read().decode("utf-8"))
                log_entry["response"] = response_payload
                self.invocation_log.append(log_entry)
                return {
                    "statusCode": 200,
                    "status": "invoked_live",
                    "function_name": self.function_name,
                    "response": response_payload,
                    "note": "AWS Lambda invoked. (Prototype safety guardrail active).",
                }
            except Exception as exc:
                logger.error(f"Live Lambda invocation failed: {exc}. Simulating locally.")

        # Local simulation mode
        handler_result = lambda_handler(payload)
        log_entry["response"] = handler_result
        log_entry["simulated"] = True
        self.invocation_log.append(log_entry)

        return {
            "statusCode": 200,
            "status": "simulated",
            "simulated": True,
            "function_name": self.function_name,
            "payload": payload,
            "result": handler_result,
            "current_breaker_state": breaker_manager.get_status(),
            "note": "Simulated local mitigation. Production airline traffic not altered.",
        }

    def get_invocation_history(self, count: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent mitigation invocation history."""
        return self.invocation_log[-count:]
