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
- **Deployment Artifacts**: Packaged standalone model archive [`ai-models/deploy/sagemaker/model.tar.gz`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/deploy/sagemaker/model.tar.gz) and full deployment guide [`documentation/SAGEMAKER_DEPLOYMENT.md`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/documentation/SAGEMAKER_DEPLOYMENT.md).
- **Dual-Mode Serving**:
  - **MODE 1 — LOCAL / MOCK (`LOCAL VERIFIED`)**: Loads genuine PyTorch weights from [`ai-models/weights/cascade_predictor_tier1a.pt`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/weights/cascade_predictor_tier1a.pt) directly via `LocalPyTorchPredictor`. Inference latency is **`1.85ms - 23.35ms`** with zero AWS network dependencies.
  - **MODE 2 — LIVE AWS (`LIVE AWS PENDING CREDENTIALS`)**: Uses `boto3.client('sagemaker-runtime')` with strict input schema validation, response schema validation, exponential backoff retries, and timeout handling.
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
  - `cascade_probability`: float ($0.0 \dots 1.0$, e.g. `0.9906` during cascade).
  - `predicted_failure_location`: str (`"boundary-gateway"`).
  - `severity`: `"NOMINAL"` | `"LOW"` | `"MEDIUM"` | `"HIGH"` | `"CRITICAL"`.
  - `lead_time`: float (estimated warning lead time in seconds, e.g. `99.1s`).
  - `conformal_bounds`: coverage interval at $1 - \alpha = 0.90$ confidence (e.g. $[0.8871, 1.0000]$).
  - `explanation`: structured diagnostic incident brief.

---

### D. AWS Lambda (`backend/cloud/lambda_handler.py`)
- **Function Name**: `baccp-circuit-breaker-mitigator` (Configurable via `AWS_LAMBDA_FUNCTION_NAME`).
- **Mitigation Engine**: Integrates the trained Proximal Policy Optimization (PPO) agent (`PPOCircuitBreakerInference`) loading [`ai-models/weights/circuit_breaker_ppo.pt`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/weights/circuit_breaker_ppo.pt) (42,720 bytes).
- **Mitigation Event Payload**:
  ```json
  {
    "event_type": "cascade_mitigation",
    "timestamp": "2026-09-22T17:36:42.341Z",
    "affected_service": "reservations",
    "gateway": "boundary-gateway",
    "cascade_probability": 0.9906,
    "severity": "CRITICAL",
    "boundary_sync_drift": 88.5,
    "lead_time": 99.1,
    "recommended_action": "OPEN",
    "throttle_rate": 1.0,
    "use_ppo": true
  }
  ```
- **Idempotency & Alert Deduplication**:
  - `CircuitBreakerManager` verifies if the new mitigation action matches the active state within a 30-second window.
  - Returns `idempotent_noop: true` and increments `repeat_count`, preventing redundant cloud alarm storms.
- **Safe Fallback Execution**:
  - If PPO weights are absent or tensors contain non-finite numbers (NaN/Inf), falls back to conservative static rules without failing the request.
- **Priority-Aware Load Shedding**:
  - PPO preserves **87.90% reservations** and **61.43% crew** throughput while shedding **38.31% baggage** throughput under heavy boundary backpressure.

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

| Feature Dimension | 1. Local Verified Mode | 2. Live AWS Design (Pending Credentials) | 3. Future Production Deployment |
| :--- | :--- | :--- | :--- |
| **Trigger / Config** | `CLOUD_MODE=local` (Default, **VERIFIED**) | `CLOUD_MODE=aws` (**PENDING CREDENTIALS**) | AWS ECS / EKS Multi-Account VPC |
| **Cascade Inference**| Genuine local PyTorch inference loading `cascade_predictor_tier1a.pt` (1.85ms - 23ms). | `boto3.client('sagemaker-runtime')` invoking live endpoint with retries & validation. | SageMaker Multi-Model Endpoint with auto-scaling and GPU acceleration. |
| **Mitigation Engine**| Genuine local PPO agent loading `circuit_breaker_ppo.pt` + idempotency manager. | Remote invocation via `boto3.client('lambda')`. | Step Functions state machine with dead-letter queue and rollback. |
| **CloudWatch** | In-memory ring buffer (500 items), ISO timestamps. Zero AWS calls. | Live `boto3.client('cloudwatch')` publishing to AWS namespace. | CloudWatch Metric Streams via Kinesis Data Firehose into OpenSearch. |
| **AWS X-Ray** | In-memory buffer (200 traces); no-op UDP emission. Zero crashes. | Live UDP socket emission to local or sidecar X-Ray daemon (:2000). | AWS Distro for OpenTelemetry (ADOT) daemonset in Kubernetes cluster. |
| **Amazon SNS** | Local memory buffer with formatted console logging. | Live `boto3.client('sns')` publishing to target Topic ARN. | SNS Fan-Out to PagerDuty, Slack webhooks, and airline Ops centers. |
| **Verification State**| **100% LOCALLY VERIFIED (76/76 Tests Pass)** | **Deployment Ready (Awaiting Credentials)** | Planned Multi-Carrier Deployment |

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
