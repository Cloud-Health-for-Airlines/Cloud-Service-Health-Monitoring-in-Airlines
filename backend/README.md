# backend/

Backend services and AWS cloud integration layer for BACCP (*Boundary-Aware Cross-Generation Cascade Predictor*).

## Modular Provider / Adapter Architecture

The cloud integration layer follows a modular provider/adapter pattern supporting both **local mode** (zero AWS credentials required, in-memory buffers, structured logging, safe simulations) and **live AWS mode** (via `boto3` SDK).

```text
       Telemetry Aggregator
                 │
                 ▼
       Cloud Orchestrator (orchestrator.py)
  ┌──────────────┴──────────────┐
  │                             │
  ▼                             ▼
CloudWatch Publisher        AWS X-Ray Tracing
(cloudwatch.py)             (xray.py)
  │                             │
  └──────────────┬──────────────┘
                 ▼
      SageMaker Model Adapter (sagemaker.py)
      [RGCN + Hawkes + Split Conformal]
                 │
                 ▼
          Risk Evaluator
                 │
                 ├──────────────────────────────┐
                 ▼                              ▼
      Lambda Circuit Breaker            Amazon SNS Alerting
      (lambda_handler.py)               (sns.py)
```

## Structure

- `cloud/config.py` — Centralized configuration manager supporting `CLOUD_MODE` (`local` default, `aws`) and environment variables without hardcoded credentials.
- `cloud/cloudwatch.py` — CloudWatch telemetry publisher supporting all 8 BACCP metrics: `boundary_health_score`, `cascade_probability`, `prediction_lead_time`, `service_latency`, `error_rate`, `request_rate`, `circuit_breaker_actions`, and `active_alerts` in namespace `BACCP/AirlineCloudHealth`.
- `cloud/xray.py` — AWS X-Ray distributed tracing module with context manager `trace_operation` across all 6 request paths (`incoming_api_request`, `telemetry_processing`, `graph_retrieval`, `prediction_request`, `alert_generation`, `circuit_breaker_action`).
- `cloud/sagemaker.py` — SageMaker multi-task cascade prediction connector accepting graph, telemetry, and boundary features, outputting calibrated probabilities with 90% conformal intervals.
- `cloud/lambda_handler.py` — Circuit breaker mitigation client and handler (`LambdaMitigationClient` and `lambda_handler`) dispatching structured cascade mitigation payloads.
- `cloud/sns.py` — SNS alert publisher dispatching high-severity predictive cascade alerts with full incident metadata.
- `cloud/orchestrator.py` — Pipeline coordinator executing the end-to-end cloud workflow.
- `api/app.py` — Zero-dependency REST API server exposing all dashboard and cloud endpoints.
- `tests/test_backend.py` — Automated unit and integration test suite (22/22 tests passing).

## REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Overall boundary health score $\epsilon(t)$ and degradation status. |
| `GET` | `/api/health/boundary` | Dedicated boundary health endpoint returning sync drift $\epsilon(t)$, threshold, and formula. |
| `GET` | `/api/graph` | Generation-typed dependency graph (`legacy`, `boundary-gateway`, `cloud-native`), edge telemetry, and latency. |
| `GET` | `/api/alerts` | Active predictive alerts with cascade probability, lead time, root cause, and 90% conformal bounds. |
| `POST` | `/api/predict` | Model prediction endpoint accepting custom graph, telemetry, and boundary features. |
| `GET` | `/api/telemetry` | Per-service telemetry, recent CloudWatch metrics, and recent X-Ray traces. |
| `GET` | `/api/circuit-breaker` | Current breaker state (`CLOSED`, `THROTTLED`, `OPEN`), throttle rate, and history. |
| `POST` | `/api/circuit-breaker` | Apply mitigation action (`THROTTLE`, `ISOLATE`, `RESET`). |
| `POST` | `/api/circuit-breaker/action` | Structured circuit breaker actuation via `LambdaMitigationClient`. |
| `POST` | `/api/pipeline/run` | Triggers the complete cloud orchestrator pipeline cycle. |
| `POST` | `/api/simulate/chaos` | Simulate testbed chaos injection (`network-delay`, `connection-drop`, `batch-job-stall`, `clear`). |
| `GET` | `/api/cloud/status` | Operational status of CloudWatch, X-Ray, SageMaker, Lambda, and SNS adapters. |

## Environment Variables

Copy `.env.example` to `.env` to configure your environment:

| Variable | Default | Description |
|---|---|---|
| `CLOUD_MODE` | `local` | `local` (simulated mock mode) or `aws` (live cloud mode). |
| `AWS_REGION` | `us-east-1` | AWS deployment region. |
| `AWS_ACCESS_KEY_ID` | *(unset)* | AWS access key (never commit to git). |
| `AWS_SECRET_ACCESS_KEY` | *(unset)* | AWS secret access key (never commit to git). |
| `CLOUDWATCH_NAMESPACE` | `BACCP/AirlineCloudHealth` | CloudWatch custom metric namespace. |
| `X_RAY_ENABLED` | `false` | Enable/disable X-Ray UDP daemon emission. |
| `AWS_SAGEMAKER_ENDPOINT` | `baccp-cascade-predictor` | SageMaker model endpoint name. |
| `AWS_LAMBDA_FUNCTION_NAME` | `baccp-circuit-breaker-mitigator` | Lambda circuit breaker function name. |
| `AWS_SNS_TOPIC_ARN` | `arn:aws:sns:...:baccp-cascade-alerts` | SNS alert notification topic ARN. |

## Running the API Server

```bash
# Start on default port (8000)
python3 backend/api/app.py

# Custom host and port
python3 backend/api/app.py --host 127.0.0.1 --port 8000
```

## Running Tests

```bash
python3 -m unittest backend/tests/test_backend.py
```
