"""AWS Cloud wiring package for BACCP."""

from .cloudwatch import CloudWatchPublisher
from .config import AWSConfig, BackendConfig, config
from .lambda_handler import CircuitBreakerManager, breaker_manager, lambda_handler
from .sagemaker import (
    SageMakerCascadePredictor,
    SageMakerEndpointError,
    SageMakerError,
    SageMakerResponseError,
    SageMakerTimeoutError,
    SageMakerValidationError,
)
from .sns import SNSPublisher
from .xray import XRayTraceRecorder

__all__ = [
    "config",
    "AWSConfig",
    "BackendConfig",
    "CloudWatchPublisher",
    "XRayTraceRecorder",
    "SageMakerCascadePredictor",
    "SageMakerError",
    "SageMakerValidationError",
    "SageMakerResponseError",
    "SageMakerTimeoutError",
    "SageMakerEndpointError",
    "SNSPublisher",
    "CircuitBreakerManager",
    "breaker_manager",
    "lambda_handler",
]
