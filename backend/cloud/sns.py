"""Amazon Simple Notification Service (SNS) alert publisher for BACCP.

Dispatches urgent cascade predictions and mitigation notifications to operational channels.
Includes all required incident attributes:
- severity
- cascade probability
- affected service
- boundary/gateway
- predicted failure location
- lead time
- timestamp
- mitigation status
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional

from .config import AWSConfig, config

logger = logging.getLogger("baccp.sns")


class SNSPublisher:
    """Dispatches alerts to Amazon SNS topic."""

    def __init__(self, aws_config: Optional[AWSConfig] = None):
        self.config = aws_config or config.aws
        self.topic_arn = self.config.sns_topic_arn
        self._client = None
        self.published_alerts: List[Dict[str, Any]] = []

        if self.config.is_live:
            try:
                import boto3  # type: ignore
                kwargs = {"region_name": self.config.region_name}
                if self.config.access_key_id and self.config.secret_access_key:
                    kwargs["aws_access_key_id"] = self.config.access_key_id
                    kwargs["aws_secret_access_key"] = self.config.secret_access_key
                self._client = boto3.client("sns", **kwargs)
                logger.info(f"Initialized live SNS client for topic '{self.topic_arn}'")
            except Exception as exc:
                logger.warning(f"Could not connect to live SNS: {exc}. Operating in local mock mode.")
                self._client = None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def publish_cascade_alert(
        self,
        severity: str,
        cascade_probability: float,
        affected_service: str,
        gateway: str = "boundary-gateway",
        predicted_failure_location: str = "boundary-gateway",
        lead_time: float = 300.0,
        mitigation_status: str = "NONE",
        extra_details: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish a structured high-severity predictive cascade alert to SNS."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Exact requested schema
        structured_alert = {
            "severity": severity,
            "cascade_probability": round(float(cascade_probability), 4),
            "affected_service": affected_service,
            "boundary/gateway": gateway,
            "gateway": gateway,
            "predicted_failure_location": predicted_failure_location,
            "lead_time": round(float(lead_time), 1),
            "timestamp": now,
            "mitigation_status": mitigation_status,
        }

        subject = f"BACCP Cascade [{severity}]: {affected_service} via {gateway}"[:100]
        message_body = (
            f"BACCP PREDICTIVE CASCADE ALERT\n"
            f"================================\n"
            f"Severity:                   {severity}\n"
            f"Cascade Probability:        {cascade_probability:.1%}\n"
            f"Affected Service:           {affected_service}\n"
            f"Boundary / Gateway:         {gateway}\n"
            f"Predicted Failure Location: {predicted_failure_location}\n"
            f"Estimated Lead Time:        {lead_time:.0f}s\n"
            f"Mitigation Status:          {mitigation_status}\n"
            f"Timestamp:                  {now}\n"
        )
        if extra_details:
            message_body += f"\nDetails: {extra_details}\n"

        self.publish_alert(
            subject=subject,
            message=message_body,
            severity=severity,
            attributes={
                "Severity": severity,
                "Probability": str(round(cascade_probability, 3)),
                "AffectedService": affected_service,
                "Gateway": gateway,
                "LeadTimeSec": str(round(lead_time)),
                "MitigationStatus": mitigation_status,
            },
        )

        return structured_alert

    def publish_alert(
        self,
        subject: str,
        message: str,
        severity: str = "HIGH",
        attributes: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Publish a notification to SNS topic."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        alert_record = {
            "timestamp": now,
            "subject": subject[:100],  # SNS subject limit is 100 chars
            "message": message,
            "severity": severity,
            "topic_arn": self.topic_arn,
            "attributes": attributes or {},
            "simulated": not self.is_live,
        }
        self.published_alerts.append(alert_record)
        if len(self.published_alerts) > 100:
            self.published_alerts = self.published_alerts[-100:]

        if self.is_live and self._client:
            try:
                message_attributes = {
                    "Severity": {"DataType": "String", "StringValue": severity},
                    "System": {"DataType": "String", "StringValue": "BACCP-AirlineHealth"},
                }
                if attributes:
                    for k, v in attributes.items():
                        message_attributes[k] = {"DataType": "String", "StringValue": str(v)}

                response = self._client.publish(
                    TopicArn=self.topic_arn,
                    Subject=subject[:100],
                    Message=message,
                    MessageAttributes=message_attributes,
                )
                logger.info(f"Published SNS alert '{subject}' (MessageId: {response.get('MessageId')})")
                return True
            except Exception as exc:
                logger.error(f"Failed to publish SNS message: {exc}")
                return False
        else:
            logger.info(f"[LOCAL SNS ALERT] [{severity}] {subject}: {message[:120]}...")
            return True

    def get_recent_alerts(self, count: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent alert notifications."""
        return self.published_alerts[-count:]
