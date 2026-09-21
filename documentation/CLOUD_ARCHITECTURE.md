# Cloud Architecture Specification: BACCP

**AWS Observability, Intelligence & Mitigation Layer for Airline IT Systems**

---

## 1. Architectural Overview

BACCP implements a modular provider/adapter cloud architecture connecting airline testbed workloads to five core AWS services:
1. **Amazon CloudWatch**: Telemetry aggregation and boundary drift metric streaming.
2. **AWS X-Ray**: Distributed cross-generation tracing across cloud microservices, gateway, and legacy core.
3. **Amazon SageMaker**: Real-time multi-task cascade prediction model serving.
4. **AWS Lambda**: Boundary-scoped automated circuit-breaker actuation.
5. **Amazon SNS**: Urgent incident notification dispatching to airline SREs and NOC operators.

```text
                               ┌───────────────────────────┐
                               │    BACCP Cloud Backend    │
                               └─────────────┬─────────────┘
                                             │
      ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
      ▼                  ▼                   ▼                   ▼                  ▼
┌───────────┐      ┌───────────┐       ┌───────────┐       ┌───────────┐      ┌───────────┐
│CloudWatch │      │   X-Ray   │       │ SageMaker │       │  Lambda   │      │    SNS    │
│8 Metrics  │      │ 6 Traced  │       │Cascade GNN│       │  Circuit  │      │ Incident  │
│Namespace  │      │Operations │       │Endpoint   │       │  Breaker  │      │  Alerts   │
└───────────┘      └───────────┘       └───────────┘       └───────────┘      └───────────┘
```

---

## 2. AWS Service Implementations

### A. Amazon CloudWatch (`backend/cloud/cloudwatch.py`)
- **Namespace**: `BACCP/AirlineCloudHealth`
- **Supported Metrics**:
  1. `boundary_health_score`: Digital-twin synchronization drift $\epsilon(t)$ in percent.
  2. `cascade_probability`: GNN predicted failure probability ($0.0 \dots 1.0$).
  3. `prediction_lead_time`: Estimated warning window $\hat{\tau}$ in seconds.
  4. `service_latency`: Per-service latency in milliseconds (Dimension: `Service`).
  5. `error_rate`: Per-service HTTP/TCP error rate in percent (Dimension: `Service`).
  6. `request_rate`: Ingress throughput in requests per second (Dimension: `Service`).
  7. `circuit_breaker_actions`: Mitigation counter (Dimensions: `Action`, `GatewayNode`).
  8. `active_alerts`: Number of active predictive cascade alarms.
- **Batching**: Metrics are formatted with UTC timestamps and published in batches of up to 20 metrics per `PutMetricData` call.

---

### B. AWS X-Ray (`backend/cloud/xray.py`)
- **Trace ID Standard**: Formatted to AWS X-Ray specification: `1-{8 hex digits time}-{24 hex digits random}`.
- **Context-Managed Operation Tracing**: Provides `trace_operation(operation_name)` covering 6 key execution paths:
  1. `incoming_api_request`: Client and dashboard HTTP requests.
  2. `telemetry_processing`: Extraction of socket connection latencies.
  3. `graph_retrieval`: Dependency graph loading and traversal.
  4. `prediction_request`: SageMaker inference invocation.
  5. `alert_generation`: Incident brief formatting.
  6. `circuit_breaker_action`: Mitigation decision and actuation.
- **Cross-Generation Segment Tree**:
  $$\text{Cloud Microservice (cloud-native)} \longrightarrow \text{Integration Gateway (boundary-gateway)} \longrightarrow \text{Legacy Mainframe (legacy)}$$
  Captures protocol transition from HTTP:8080 to TCP:9090.

---

### C. Amazon SageMaker (`backend/cloud/sagemaker.py`)
- **Endpoint Name**: `baccp-cascade-predictor` (Configurable via `AWS_SAGEMAKER_ENDPOINT`).
- **Logical Input Payload**:
  ```json
  {
    "graph": { "nodes": [...], "edges": [...] },
    "telemetry_features": { "reservations": {"latency_ms": 18.5, "error_rate": 0.2} },
    "boundary_features": { "sync_drift_score": 48.5, "gateway": "boundary-gateway" },
    "timestamp": 1774281600.0
  }
  ```
- **Logical Output Schema**:
  - `cascade_probability`: float ($0.0 \dots 1.0$).
  - `predicted_failure_location`: str (e.g. `"boundary-gateway"`).
  - `severity`: `"LOW"` | `"MEDIUM"` | `"HIGH"` | `"CRITICAL"`.
  - `lead_time`: float (estimated warning lead time in seconds).
  - `conformal_bounds`: coverage interval at $1 - \alpha = 0.90$ confidence.
  - `explanation`: natural-language AIOps incident brief.

---

### D. AWS Lambda (`backend/cloud/lambda_handler.py`)
- **Function Name**: `baccp-circuit-breaker-mitigator` (Configurable via `AWS_LAMBDA_FUNCTION_NAME`).
- **Mitigation Event Payload**:
  ```json
  {
    "event_type": "cascade_mitigation",
    "timestamp": "2026-09-21T16:07:31.165Z",
    "affected_service": "reservations",
    "gateway": "boundary-gateway",
    "cascade_probability": 0.88,
    "severity": "CRITICAL",
    "recommended_action": "OPEN"
  }
  ```
- **Dynamic Mitigation Equation**:
  $$\text{ThrottleRate} = \min(0.75, 0.30 + (P_{\text{cascade}} - 0.55) \times 2.0)$$
  For critical cascades ($P \ge 0.75$), throttle rate = $1.0$ (100% boundary isolation).

---

### E. Amazon SNS (`backend/cloud/sns.py`)
- **Topic ARN**: `arn:aws:sns:us-east-1:123456789012:baccp-cascade-alerts` (Configurable via `AWS_SNS_TOPIC_ARN`).
- **Structured Alert Attributes**:
  - `Severity`: `"HIGH"` | `"CRITICAL"`
  - `Probability`: str ($P_{\text{cascade}}$)
  - `AffectedService`: str (`"reservations"`)
  - `Gateway`: str (`"boundary-gateway"`)
  - `LeadTimeSec`: str ($\hat{\tau}$ in seconds)
  - `MitigationStatus`: str (`"THROTTLED"` | `"OPEN"`)

---

## 3. Separation of Architectural Modes

BACCP explicitly separates execution into three distinct deployment phases:

| Feature Dimension | 1. Local Prototype Mode | 2. AWS-Integrated Design | 3. Future Production Deployment |
| :--- | :--- | :--- | :--- |
| **Trigger / Config** | `CLOUD_MODE=local` (Default) | `CLOUD_MODE=aws` | AWS ECS / EKS Multi-Account VPC |
| **CloudWatch** | In-memory ring buffer (500 items), ISO timestamps. Zero AWS calls. | Live `boto3.client('cloudwatch')` publishing to AWS namespace. | CloudWatch Metric Streams via Kinesis Data Firehose into OpenSearch. |
| **AWS X-Ray** | In-memory buffer (200 traces); no-op UDP emission. Zero crashes. | Live UDP socket emission to local or sidecar X-Ray daemon (:2000). | AWS Distro for OpenTelemetry (ADOT) daemonset in Kubernetes cluster. |
| **SageMaker** | Calibrated analytical inference engine with 90% conformal intervals. | `boto3.client('sagemaker-runtime')` invoking live model endpoint. | SageMaker Multi-Model Endpoint with auto-scaling and GPU acceleration. |
| **AWS Lambda** | Local simulation via `lambda_handler()`, updates `CircuitBreakerManager`. | Remote invocation via `boto3.client('lambda')`. | Step Functions state machine with dead-letter queue and rollback. |
| **Amazon SNS** | Local memory buffer with formatted console logging. | Live `boto3.client('sns')` publishing to target Topic ARN. | SNS Fan-Out to PagerDuty, Slack webhooks, and airline Ops centers. |
| **Safety Guardrail** | Explicitly marked as prototype simulated action. | Live mitigation scoped to testbed gateway. | Dual-approval human-in-the-loop bypass for flight-critical paths. |

---

## 4. Environment Variables & Configuration

All settings are managed through `backend/cloud/config.py` and documented in `.env.example`:

```bash
# Cloud Mode: 'local' (simulated offline mode) or 'aws' (live AWS integration)
CLOUD_MODE=local

# AWS Core Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# CloudWatch, X-Ray, SageMaker, Lambda, and SNS
CLOUDWATCH_NAMESPACE=BACCP/AirlineCloudHealth
X_RAY_ENABLED=false
AWS_XRAY_DAEMON_ADDRESS=127.0.0.1:2000
AWS_SAGEMAKER_ENDPOINT=baccp-cascade-predictor
AWS_LAMBDA_FUNCTION_NAME=baccp-circuit-breaker-mitigator
AWS_SNS_TOPIC_ARN=arn:aws:sns:us-east-1:123456789012:baccp-cascade-alerts
```

---

## 5. Security & Isolation Controls

1. **Zero Hardcoded Credentials**: Credentials are read exclusively from environment variables or IAM instance metadata.
2. **Git Exclusion**: `.gitignore` strictly excludes `.env`, `.env.*`, `*.pem`, and `*.key` files.
3. **Internal Mainframe Isolation**: The legacy mainframe simulator runs on an isolated Docker network (`legacy-tier`) without published host ports, preventing unauthorized access.
4. **Non-Intrusive eBPF Tracing**: Kernel probes run with read-only probe privileges (`kprobe:tcp_v4_connect`), incapable of modifying network packets in transit.
