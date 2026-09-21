"""Configuration management for AWS services and BACCP backend.

Supports modular dual-mode operation:
- 'local' (default): Local development, testbed chaos experiments, and offline demo mode.
  All cloud services are gracefully mocked with in-memory buffers and simulated behaviors.
- 'aws': Live AWS cloud integration using boto3 SDK when AWS credentials and endpoints are provided.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AWSConfig:
    # 1. Cloud Mode: 'local' (default) or 'aws' (also accepts legacy 'mock' or 'live')
    cloud_mode: str = field(default_factory=lambda: (
        os.getenv("CLOUD_MODE", os.getenv("AWS_MODE", "local")).lower()
    ))

    # 2. AWS Core Configuration
    region_name: str = field(default_factory=lambda: (
        os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    ))
    access_key_id: Optional[str] = field(default_factory=lambda: (
        os.getenv("AWS_ACCESS_KEY_ID") or None
    ))
    secret_access_key: Optional[str] = field(default_factory=lambda: (
        os.getenv("AWS_SECRET_ACCESS_KEY") or None
    ))

    # 3. CloudWatch Namespace (default: BACCP/AirlineCloudHealth)
    cloudwatch_namespace: str = field(default_factory=lambda: (
        os.getenv("CLOUDWATCH_NAMESPACE", "BACCP/AirlineCloudHealth")
    ))

    # 4. AWS X-Ray Tracing
    xray_enabled: bool = field(default_factory=lambda: (
        os.getenv("X_RAY_ENABLED", "false").lower() in ("true", "1", "yes")
    ))
    xray_daemon_address: str = field(default_factory=lambda: (
        os.getenv("AWS_XRAY_DAEMON_ADDRESS", "127.0.0.1:2000")
    ))

    # 5. Amazon SageMaker Endpoint
    sagemaker_endpoint_name: str = field(default_factory=lambda: (
        os.getenv("AWS_SAGEMAKER_ENDPOINT", os.getenv("SAGEMAKER_ENDPOINT_NAME", "baccp-cascade-predictor"))
    ))

    # 6. AWS Lambda Circuit Breaker
    lambda_breaker_function: str = field(default_factory=lambda: (
        os.getenv("AWS_LAMBDA_FUNCTION_NAME", os.getenv("LAMBDA_BREAKER_FUNCTION", "baccp-circuit-breaker-mitigator"))
    ))

    # 7. Amazon SNS Topic
    sns_topic_arn: str = field(default_factory=lambda: (
        os.getenv("AWS_SNS_TOPIC_ARN", os.getenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123456789012:baccp-cascade-alerts"))
    ))

    @property
    def is_live(self) -> bool:
        """Returns True only when cloud mode is configured for live AWS and credentials exist."""
        norm_mode = self.cloud_mode.lower()
        if norm_mode in ("aws", "live"):
            # In AWS mode, check if credentials or default provider exists
            return True
        return False

    @property
    def mode(self) -> str:
        """Backward-compatible mode property returning 'live' or 'local'/'mock'."""
        return "live" if self.is_live else "local"


@dataclass
class BackendConfig:
    host: str = field(default_factory=lambda: os.getenv("BACCP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("BACCP_PORT", "8000")))
    testbed_dir: str = field(default_factory=lambda: os.getenv("TESTBED_DIR", "testbed"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://baccp:baccp-local@127.0.0.1:5432/baccp"))
    sync_drift_threshold: float = field(default_factory=lambda: float(os.getenv("SYNC_DRIFT_THRESHOLD", "45.0")))
    cascade_risk_threshold: float = field(default_factory=lambda: float(os.getenv("CASCADE_RISK_THRESHOLD", "0.65")))
    aws: AWSConfig = field(default_factory=AWSConfig)


# Global default configuration instance
config = BackendConfig()
