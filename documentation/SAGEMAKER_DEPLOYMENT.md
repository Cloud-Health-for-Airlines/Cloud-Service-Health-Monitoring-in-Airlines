# Amazon SageMaker Deployment & Inference Integration Guide

**System**: BACCP — Boundary-Aware Cross-Generation Cascade Predictor  
**Domain**: Airline IT Operations (Legacy Mainframe ↔ Cloud Microservice Boundary)  
**Component**: Amazon SageMaker Real-Time Inference Integration & Local Fallback Adapter  
**Last Updated**: September 2026  

---

## 1. Executive Summary & Verification Status

The BACCP cascade prediction system utilizes a multi-task Heterogeneous Relational Graph Convolutional Network (RGCN) with a Gated Recurrent Unit (GRU) to forecast cascading service disruptions across hybrid airline architectures. The SageMaker integration adapter in [`backend/cloud/sagemaker.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/sagemaker.py) bridges the backend REST API to either a live AWS SageMaker real-time endpoint or a local PyTorch inference runtime.

### Current Verification Status Matrix

| Environment / Mode | Operational Status | Verification Evidence | Notes |
| :--- | :---: | :--- | :--- |
| **MODE 1: LOCAL / MOCK** | **LOCAL VERIFIED** | Genuine PyTorch inference via [`cascade_predictor_tier1a.pt`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/weights/cascade_predictor_tier1a.pt); latency: **1.82 ms**; 100% contract compliance. | Passed automated smoke test in [`results/sagemaker_smoke_test.json`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/results/sagemaker_smoke_test.json). |
| **MODE 2: LIVE AWS** | **PENDING CREDENTIALS** | Complete deployment package created ([`model.tar.gz`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/deploy/sagemaker/model.tar.gz), [`inference.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/deploy/sagemaker/inference.py)); retry, timeout, and validation logic verified via mocked AWS tests. | **Not deployed to live AWS** due to absence of production AWS credentials in local environment. |

> [!IMPORTANT]
> **Strict Verification Guarantee**:
> In accordance with project integrity requirements, mock and local tests are **never** reported as live AWS tests. Live AWS deployment is formally classified as **PENDING CREDENTIALS**.

---

## 2. End-to-End Architecture Flow

The prediction and mitigation pipeline operates as follows:

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Dashboard                   │
│         (React / Vite Operational Health Canvas)        │
└────────────────────────────┬────────────────────────────┘
                             │ HTTP POST /api/predict
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    Backend REST API                     │
│                  (backend/api/app.py)                   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│               SageMaker Prediction Adapter              │
│               (backend/cloud/sagemaker.py)              │
├────────────────────────────┬────────────────────────────┤
│   MODE 1: LOCAL / MOCK     │     MODE 2: LIVE AWS       │
│  (Default / Fallback)      │ (Requires AWS Credentials) │
│                            │                            │
│  • Loads PyTorch Weights   │  • Request Schema Valid.   │
│    (cascade_predictor_     │  • boto3 sagemaker-runtime │
│     tier1a.pt)             │  • Exponential Backoff     │
│  • HeteroCascadePredictor  │  • Timeout Handling        │
│  • Sub-2ms local latency   │  • Response Schema Valid.  │
│  • Calibrated analytical   │  • Graceful Local Fallback │
│    safety net              │                            │
└─────────────┬──────────────┴─────────────┬──────────────┘
              │                            │
              ▼                            ▼
┌────────────────────────────┐┌────────────────────────────┐
│  Local PyTorch Inference   ││ Live SageMaker Endpoint    │
│  (In-process CPU execution)││ (baccp-cascade-predictor)  │
└─────────────┬──────────────┘└────────────┬───────────────┘
              │                            │
              └──────────────┬─────────────┘
                             │ Multi-Task Prediction Payload
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    Prediction Result                    │
│   • Cascade Probability (P ∈ [0, 1])                    │
│   • Estimated Lead Time to Disruption (seconds)         │
│   • Predicted Root-Cause Node (e.g. boundary-gateway)   │
│   • ITIL Severity Level (NOMINAL / LOW / MED / HIGH /   │
│     CRITICAL)                                           │
│   • Conformal Prediction Bounds (90% coverage interval) │
│   • Inference Latency (ms)                              │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                 Backend Alert & Mitigation              │
│   • CloudWatch Metrics Published (CloudWatchPublisher)  │
│   • SNS Predictive Alert Dispatched (SNSPublisher)      │
│   • Circuit Breaker Triggered (LambdaMitigationClient)  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│               Frontend Predictive Banner                │
│    (Operator UI receives proactive advance warning)     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Dual-Mode Specification

### MODE 1 — LOCAL / MOCK (Offline Development & Testbed Evaluation)
- **Activation**: Automatically activated when `CLOUD_MODE=local` (default) or when AWS credentials are not present.
- **Implementation**:
  - Automatically loads the trained PyTorch checkpoint [`ai-models/weights/cascade_predictor_tier1a.pt`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/weights/cascade_predictor_tier1a.pt) into CPU memory.
  - Dynamically synthesizes the 5 temporal snapshots with current boundary synchronization drift $\epsilon(t)$ and active chaos fault parameters.
  - Performs genuine tensor forward pass through the Relational Graph Convolutional Network (RGCN) and GRU temporal aggregator.
  - Generates calibrated multi-task outputs including 90% conformal coverage intervals.
  - Measures execution latency (typically $1.5\text{ ms} - 2.5\text{ ms}$).
  - Falls back to the testbed-calibrated analytical engine if PyTorch dependencies or weights files are unreadable.

### MODE 2 — LIVE AWS (Production Real-Time Serving)
- **Activation**: Activated when `CLOUD_MODE=aws`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY` are provided.
- **Client Configuration**:
  - Utilizes `boto3.client('sagemaker-runtime')` configured via `botocore.config.Config`.
  - Configures `connect_timeout` and `read_timeout` (default: 5.0 seconds).
  - Configures zero SDK retries to give the adapter explicit control over exponential backoff, logging, and metrics recording.
- **Resilience Features**:
  1. **Request Validation**:
     - Pre-validates payloads before network egress using `validate_request_payload`.
     - Ensures $\epsilon(t) \in [0.0, 100.0]$, graph nodes list structure, and recognized fault parameters.
     - Rejects malformed requests immediately with `SageMakerValidationError`.
  2. **Response Contract Validation**:
     - Validates incoming JSON responses against the BACCP contract using `validate_response_payload`.
     - Validates $P(\text{cascade}) \in [0.0, 1.0]$, valid root-cause node string, recognized severity level, non-negative lead time, and coherent conformal intervals ($0 \le \text{lower} \le \text{upper} \le 1$).
     - Flags violations with `SageMakerResponseError`.
  3. **Exponential Backoff Retry**:
     - Retries transient errors (`ThrottlingException`, `503 ServiceUnavailable`, HTTP 429, socket reset) up to `SAGEMAKER_MAX_RETRIES` (default: 3).
     - Backoff interval: $t_{\text{sleep}} = 0.25 \times 2^{\text{attempt}-1}\text{ seconds}$.
  4. **Timeout Handling**:
     - Catches read timeouts and raises structured `SageMakerTimeoutError`.
  5. **Graceful Fallback to Local Model**:
     - If `SAGEMAKER_FALLBACK_TO_LOCAL=true` (default), network or endpoint failures seamlessly fall back to the in-memory PyTorch model without disrupting API callers.
     - Logs detailed warnings and flags `"mode": "local-trained-pytorch"` or `"mode": "local-analytical"` in the result metadata.

---

## 4. Environment Variables Configuration

All configuration settings are centralized in [`backend/cloud/config.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/config.py) and documented in [`.env.example`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/.env.example):

```bash
# Cloud Mode ('local' or 'aws')
CLOUD_MODE=local

# AWS Core Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# Amazon SageMaker Configuration
AWS_SAGEMAKER_ENDPOINT=baccp-cascade-predictor
SAGEMAKER_ENDPOINT_NAME=baccp-cascade-predictor
SAGEMAKER_TIMEOUT_SECONDS=5.0
SAGEMAKER_MAX_RETRIES=3
SAGEMAKER_FALLBACK_TO_LOCAL=true
```

---

## 5. SageMaker Serving Contract (`inference.py`)

The serving handler [`ai-models/deploy/sagemaker/inference.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/deploy/sagemaker/inference.py) implements the SageMaker PyTorch container standard:

```python
def model_fn(model_dir: str) -> HeteroCascadePredictor:
    """Loads weights into HeteroCascadePredictor on CPU/GPU."""
    ...

def input_fn(request_body: str, request_content_type: str = "application/json") -> Dict[str, Any]:
    """Deserializes and validates JSON payload."""
    ...

def predict_fn(input_data: Dict[str, Any], model: HeteroCascadePredictor) -> Dict[str, Any]:
    """Executes multi-task forward inference and computes conformal intervals."""
    ...

def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    """Serializes output to JSON."""
    ...
```

### Request JSON Schema
```json
{
  "sync_drift_score": 78.5,
  "active_fault": "network-delay",
  "fault_level": "high",
  "telemetry_features": {},
  "boundary_features": {
    "sync_drift_score": 78.5
  },
  "graph": {
    "nodes": ["reservations", "crew", "baggage", "boundary-gateway", "legacy-mainframe"]
  }
}
```

### Response JSON Schema
```json
{
  "cascade_probability": 0.9906,
  "predicted_root_cause_node": "boundary-gateway",
  "predicted_failure_location": "boundary-gateway",
  "predicted_affected_nodes": ["boundary-gateway", "reservations", "crew", "baggage"],
  "severity": "CRITICAL",
  "severity_index": 4,
  "lead_time": 98.5,
  "estimated_lead_time_seconds": 98.5,
  "confidence": 0.9,
  "conformal_bounds": {
    "confidence_level": 0.9,
    "lower": 0.8871,
    "upper": 1.0
  },
  "boundary_sync_drift": 78.5,
  "inference_latency_ms": 1.82,
  "model_metadata": {
    "engine": "HeteroRGCN-GRU-MultiTask",
    "checkpoint": "cascade_predictor_tier1a.pt",
    "mode": "local-trained-pytorch"
  }
}
```

---

## 6. Step-by-Step Live AWS Deployment Guide

When production AWS credentials become available, execute the following procedure to provision the live SageMaker endpoint:

### Step 1: Package Model Tarball
Generate the production deployment package containing weights and serving code:
```bash
python3 ai-models/deploy/sagemaker/deploy_endpoint.py --dry-run
```
This produces [`ai-models/deploy/sagemaker/model.tar.gz`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/deploy/sagemaker/model.tar.gz) containing:
- `model.pt` (Trained model weights, 362 KB)
- `code/inference.py` (Container serving handler)
- `code/model.py` (HeteroRGCN architecture definition)
- `code/requirements.txt` (`torch>=2.0.0`, `numpy<2.0.0`)

### Step 2: Configure Environment Credentials
Populate `.env` with AWS credentials:
```bash
cp .env.example .env
# Edit .env and set:
# CLOUD_MODE=aws
# AWS_ACCESS_KEY_ID=AKIA...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION=us-east-1
```

### Step 3: Deploy Endpoint via AWS CLI / Python SDK
Execute live deployment:
```bash
python3 ai-models/deploy/sagemaker/deploy_endpoint.py \
  --deploy \
  --endpoint-name baccp-cascade-predictor \
  --instance-type ml.m5.large \
  --role-arn arn:aws:iam::123456789012:role/BACCP-SageMaker-ExecutionRole
```

Or deploy via AWS CLI:
```bash
# 1. Upload model tarball to S3
aws s3 cp ai-models/deploy/sagemaker/model.tar.gz s3://baccp-models-bucket/baccp-cascade-predictor/model.tar.gz

# 2. Create SageMaker Model
aws sagemaker create-model \
  --model-name baccp-cascade-predictor-model \
  --primary-container Image=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310,ModelDataUrl=s3://baccp-models-bucket/baccp-cascade-predictor/model.tar.gz \
  --execution-role-arn arn:aws:iam::123456789012:role/BACCP-SageMaker-ExecutionRole

# 3. Create Endpoint Configuration
aws sagemaker create-endpoint-config \
  --endpoint-config-name baccp-cascade-predictor-config \
  --production-variants VariantName=AllTraffic,ModelName=baccp-cascade-predictor-model,InitialInstanceCount=1,InstanceType=ml.m5.large

# 4. Create or Update Endpoint
aws sagemaker create-endpoint \
  --endpoint-name baccp-cascade-predictor \
  --endpoint-config-name baccp-cascade-predictor-config
```

### Step 4: Run Live Smoke Test
Verify endpoint health and record live latency:
```bash
python3 backend/cloud/sagemaker_smoke_test.py
```
Upon successful invocation of the live endpoint, the status in `results/sagemaker_smoke_test.json` will update to **`LIVE_AWS_VERIFIED`**.

---

## 7. Smoke Test Results Summary

Executed via `python3 backend/cloud/sagemaker_smoke_test.py`:

```json
{
  "timestamp": "2026-09-22T17:22:38.425181+00:00",
  "overall_status": "LOCAL_VERIFIED (LIVE AWS PENDING CREDENTIALS)",
  "local_status": "LOCAL_VERIFIED",
  "live_aws_status": "PENDING_CREDENTIALS",
  "aws_region": "us-east-1",
  "endpoint_name": "baccp-cascade-predictor",
  "live_credentials_present": false,
  "local_inference_smoke_test": {
    "status": "PASS",
    "model_architecture": "HeteroRGCN-GRU-MultiTask",
    "checkpoint": "cascade_predictor_tier1a.pt",
    "sample_prediction": {
      "sync_drift_score": 78.5,
      "cascade_probability": 0.9906,
      "severity": "CRITICAL",
      "predicted_root_cause_node": "boundary-gateway",
      "estimated_lead_time_seconds": 98.5,
      "inference_latency_ms": 1.82
    }
  },
  "mocked_aws_integration_test": {
    "status": "PASS",
    "subtests": {
      "live_invocation_contract": "PASS",
      "exponential_backoff_retry": "PASS",
      "request_validation_rejection": "PASS (Correctly rejected drift=150.0)",
      "response_validation_rejection": "PASS (Correctly rejected probability=1.5)",
      "timeout_graceful_fallback": "PASS (Fell back to local model)"
    }
  }
}
```
