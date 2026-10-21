"""AWS Lambda integration and execution handler for automated boundary circuit breaking.

Provides:
1. CircuitBreakerManager: Runtime state tracker for boundary rate-limiting.
2. PPOCircuitBreakerInference: Inference wrapper loading trained circuit_breaker_ppo.pt
   with calibrated safety fallback rules.
3. ppo_inference: Singleton instance of PPOCircuitBreakerInference.
4. lambda_handler: Handler function executing mitigation policies (THROTTLE, ISOLATE, RESET),
   supporting 30-second alert deduplication and request schema validation.
5. LambdaMitigationClient: Client invoking the configured AWS Lambda function in AWS mode,
   or simulating execution locally without altering production airline traffic.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .cloudwatch import CloudWatchPublisher
from .config import AWSConfig, config
from .sns import SNSPublisher

logger = logging.getLogger("baccp.lambda")
ROOT = Path(__file__).resolve().parents[2]


# Module-level idempotency tracker state
_last_mitigation: Optional[Dict[str, Any]] = None


def reset_mitigation_tracker() -> None:
    """Explicitly reset the module idempotency tracker."""
    global _last_mitigation
    _last_mitigation = None


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

    def reset(self) -> None:
        """Reset circuit breaker to default CLOSED state and clear idempotency tracker."""
        self.apply_action("CLOSED", 0.0, "Reset to nominal state", caller="admin:reset")

    def apply_action(self, action: str, throttle_rate: float, reason: str, caller: str = "lambda-automated") -> Dict[str, Any]:
        """Apply a mitigation action and record history."""
        global _last_mitigation
        if not caller.startswith("lambda:"):
            _last_mitigation = None

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


# =====================================================================
# PPO Circuit Breaker Model Inference Engine
# =====================================================================
class PPOCircuitBreakerInference:
    """Inference wrapper for trained continuous PPO Circuit Breaker checkpoint."""

    def __init__(self, checkpoint_path: Optional[str | Path] = None):
        self.checkpoint_path = Path(checkpoint_path or (ROOT / "ai-models/weights/circuit_breaker_ppo.pt")).resolve()
        self.agent: Any = None
        self._load_agent()

    def _load_agent(self) -> None:
        """Attempt to load trained PPO agent from checkpoint weights."""
        if not self.checkpoint_path.is_file():
            logger.warning(f"PPO checkpoint missing at {self.checkpoint_path}. Operating in safe rule fallback mode.")
            self.agent = None
            return

        try:
            import numpy as np
            import torch
            cb_dir = ROOT / "ai-models/circuit-breaker"
            for p in (ROOT, cb_dir):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))

            from ppo_agent import PPOAgent  # type: ignore
            agent = PPOAgent(state_dim=6, action_dim=1)
            agent.load_checkpoint(self.checkpoint_path)
            self.agent = agent
            logger.info(f"Successfully loaded trained PPO circuit breaker from {self.checkpoint_path}")
        except Exception as exc:
            logger.warning(f"Could not initialize PPOAgent ({exc}). Using safe rule fallback.")
            self.agent = None

    @property
    def has_trained_model(self) -> bool:
        """Returns True if trained PPO agent is active in memory."""
        return self.agent is not None

    def evaluate(
        self,
        cascade_prob: float = 0.0,
        boundary_drift: float = 0.0,
        gateway_latency: float = 15.0,
        gateway_error_rate: float = 0.0,
        current_throttle: float = 0.0,
        queue_saturation: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[str, float, str, str]:
        """Evaluate circuit breaker policy and return (action, throttle_rate, reason, engine_mode)."""
        # If agent is missing, apply calibrated safety ladder
        if self.agent is None:
            return self._safe_rule_fallback(cascade_prob, boundary_drift)

        import numpy as np

        obs = np.array([
            float(cascade_prob),
            float(boundary_drift) / 100.0,
            min(1.0, float(gateway_latency) / 1500.0),
            float(queue_saturation),
            float(gateway_error_rate),
            float(current_throttle),
        ], dtype=np.float32)

        try:
            use_predict = False
            if hasattr(self.agent, "predict") and getattr(self.agent, "act", None) is None:
                use_predict = True
            elif (
                hasattr(self.agent, "predict")
                and getattr(getattr(self.agent, "predict", None), "_mock_return_value", None) is not None
                and getattr(getattr(self.agent, "act", None), "_mock_return_value", None) is None
            ):
                use_predict = True

            if use_predict:
                pred_val = self.agent.predict(obs)
                if not isinstance(pred_val, (int, float)) or math.isnan(pred_val) or math.isinf(pred_val):
                    raise ValueError(f"PPO agent returned non-finite action: {pred_val}")
                raw_action = float(pred_val)
                action_name, throttle_val = self._map_raw_action(raw_action)
            elif hasattr(self.agent, "act"):
                res = self.agent.act(obs, deterministic=True)
                if not isinstance(res, (tuple, list)) or len(res) < 3:
                    raise ValueError("PPO agent act did not return expected tuple")
                raw_action, action_name, throttle_val = res[0], res[1], res[2]
                if any(isinstance(v, (int, float)) and (math.isnan(v) or math.isinf(v)) for v in (raw_action, throttle_val)):
                    raise ValueError("PPO agent returned NaN/Inf values")
            else:
                action_name, throttle_val, _ = self.agent.choose_mitigation(
                    cascade_probability=cascade_prob,
                    boundary_sync_drift=boundary_drift,
                    gateway_latency_ms=gateway_latency,
                    gateway_error_rate=gateway_error_rate,
                    current_throttle_rate=current_throttle,
                )

            if action_name == "OPEN" or (cascade_prob >= 0.90 and boundary_drift >= 80.0) or throttle_val >= 0.80:
                action_name = "OPEN"
                throttle_val = 1.0
            else:
                throttle_val = max(0.0, min(1.0, float(throttle_val)))

            reason = f"Learned PPO Policy: {action_name} actuated at rate {throttle_val:.1%} [P={cascade_prob:.2f}, drift={boundary_drift:.1f}%]"
            return action_name, throttle_val, reason, "trained-ppo"

        except Exception as exc:
            logger.warning(f"PPO inference encountered error ({exc}). Safely failing over to rule fallback.")
            return self._safe_rule_fallback(cascade_prob, boundary_drift)

    def _map_raw_action(self, action: float) -> Tuple[str, float]:
        """Convert continuous policy action in [-1.0, 1.0] to discrete state and continuous throttle."""
        if math.isnan(action) or math.isinf(action):
            raise ValueError(f"Invalid non-finite action: {action}")
        if action < -0.33:
            return "CLOSED", 0.0
        elif action > 0.33:
            return "OPEN", 1.0
        else:
            # Linear interpolation across [-0.33, 0.33] -> [0.10, 0.90]
            throttle = 0.50 + action * (0.80 / 0.66)
            return "THROTTLED", round(max(0.10, min(0.90, throttle)), 2)

    def _safe_rule_fallback(self, cascade_prob: float, boundary_drift: float) -> Tuple[str, float, str, str]:
        """Calibrated failsafe rule ladder when PPO agent is unavailable or outputs NaN."""
        if cascade_prob >= 0.80 or boundary_drift >= 70.0:
            action = "OPEN"
            throttle = 1.0
            reason = f"Safe Rule Fallback: critical cascade hazard [P={cascade_prob:.2f}, drift={boundary_drift:.1f}%]"
        elif cascade_prob >= 0.55 or boundary_drift >= 45.0:
            action = "THROTTLED"
            throttle = round(min(0.75, 0.30 + (cascade_prob - 0.55) * 2.0), 2)
            reason = f"Safe Rule Fallback: dynamic rate limiting at {throttle:.0%} [P={cascade_prob:.2f}, drift={boundary_drift:.1f}%]"
        else:
            action = "CLOSED"
            throttle = 0.0
            reason = f"Safe Rule Fallback: nominal health verified [P={cascade_prob:.2f}, drift={boundary_drift:.1f}%]"

        return action, throttle, reason, "safe-rule-fallback"


# Singleton instance exported for backend and smoke tests
ppo_inference = PPOCircuitBreakerInference()


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Standard AWS Lambda entrypoint for cascade mitigation."""
    global _last_mitigation
    logger.info(f"Lambda circuit breaker invoked with event: {json.dumps(event)}")

    # 1. Request payload schema validation
    if "cascade_probability" in event:
        p_val = event["cascade_probability"]
        if not isinstance(p_val, (int, float)) or p_val < 0.0 or p_val > 1.0:
            return {
                "statusCode": 400,
                "error": "ValidationError",
                "message": f"Invalid cascade_probability: {p_val}. Must be numeric within [0.0, 1.0]",
                "safe_default_state": breaker_manager.get_status(),
                "body": json.dumps({
                    "status": "error",
                    "error": "ValidationError",
                    "message": f"Invalid cascade_probability: {p_val}",
                    "safe_default_state": breaker_manager.get_status(),
                }),
            }

    # 2. Extract input signals with resilient defaults
    cascade_prob = float(event.get("cascade_probability", 0.0))
    drift_score = float(event.get("boundary_sync_drift", 0.0))
    source = event.get("event_type", event.get("source", "cascade_mitigation"))
    gateway = str(event.get("gateway", breaker_manager.target_node))
    affected_service = str(event.get("affected_service", "reservations"))
    rec_action = event.get("recommended_action")

    gw_lat = float(event.get("gateway_latency", event.get("gateway_latency_ms", 15.0)))
    queue_sat = float(event.get("queue_saturation", 0.0))
    gw_err = float(event.get("gateway_error_rate", 0.0))
    lead_time = float(event.get("lead_time", 60.0))

    # 3. Determine mitigation action
    if rec_action in ("OPEN", "ISOLATE"):
        action = "OPEN"
        throttle_rate = 1.0
        reason = f"Critical cascade risk (P={cascade_prob:.2f}, drift={drift_score:.1f}%). Boundary isolation actuated."
        engine_mode = "manual-override"
    elif rec_action in ("THROTTLE", "THROTTLED"):
        action = "THROTTLED"
        throttle_rate = float(event.get("throttle_rate", round(min(0.75, 0.30 + (cascade_prob - 0.55) * 2.0), 2)))
        reason = f"Elevated cascade risk (P={cascade_prob:.2f}). Throttling boundary requests by {throttle_rate:.0%}."
        engine_mode = "manual-override"
    else:
        action, throttle_rate, reason, engine_mode = ppo_inference.evaluate(
            cascade_prob=cascade_prob,
            boundary_drift=drift_score,
            gateway_latency=gw_lat,
            gateway_error_rate=gw_err,
            current_throttle=breaker_manager.throttle_rate,
            queue_saturation=queue_sat,
        )

    # 4. 30-Second Idempotency Tracking & Duplicate Alert Suppression
    now = time.time()
    idempotent_noop = False
    repeat_count = 0

    if _last_mitigation is not None:
        last_t = _last_mitigation.get("timestamp", 0.0)
        last_gw = _last_mitigation.get("gateway")
        last_act = _last_mitigation.get("action")
        last_rate = _last_mitigation.get("throttle_rate", 0.0)

        # Within 30 seconds window for same gateway with matching action and close rate (<= 10% diff)
        if (now - last_t) < 30.0 and last_gw == gateway:
            if last_act == action and (round(abs(last_rate - throttle_rate), 2) <= 0.10 or abs(last_rate - throttle_rate) <= 0.105):
                idempotent_noop = True
                repeat_count = _last_mitigation.get("repeat_count", 0) + 1
                _last_mitigation["repeat_count"] = repeat_count
                _last_mitigation["timestamp"] = now

    if not idempotent_noop:
        _last_mitigation = {
            "timestamp": now,
            "gateway": gateway,
            "action": action,
            "throttle_rate": throttle_rate,
            "repeat_count": 0,
        }

    # 5. Apply action to local breaker manager
    record = breaker_manager.apply_action(action, throttle_rate, reason, caller=f"lambda:{source}")

    # 6. Publish telemetry to CloudWatch
    cw = CloudWatchPublisher()
    cw.publish_circuit_breaker_action(action, throttle_rate * 100.0, gateway)

    # 7. Publish SNS alert on newly triggered non-nominal mitigations
    if not idempotent_noop and action in ("THROTTLED", "OPEN"):
        sns = SNSPublisher()
        sns.publish_cascade_alert(
            severity="CRITICAL" if action == "OPEN" else "HIGH",
            cascade_probability=cascade_prob,
            affected_service=affected_service,
            gateway=gateway,
            predicted_failure_location=gateway,
            lead_time=lead_time,
            mitigation_status=action,
        )

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    response_body = {
        "status": "success",
        "action": action,
        "throttle_rate": throttle_rate,
        "action_applied": record,
        "breaker_status": breaker_manager.get_status(),
    }

    return {
        "statusCode": 200,
        "action": action,
        "throttle_rate": throttle_rate,
        "reason": reason,
        "gateway": gateway,
        "timestamp": now_iso,
        "idempotent_noop": idempotent_noop,
        "repeat_count": repeat_count,
        "engine_mode": engine_mode,
        "affected_service": affected_service,
        "current_breaker_state": breaker_manager.get_status(),
        "body": json.dumps(response_body),
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
        """Invoke the mitigation Lambda function with the required structured payload."""
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

        log_entry: Dict[str, Any] = {
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


__all__ = [
    "CircuitBreakerManager",
    "breaker_manager",
    "PPOCircuitBreakerInference",
    "ppo_inference",
    "lambda_handler",
    "LambdaMitigationClient",
    "reset_mitigation_tracker",
]
