# AWS Lambda Boundary Circuit-Breaker Action Layer Deployment Guide

**System**: BACCP — Boundary-Aware Cross-Generation Cascade Predictor  
**Domain**: Airline IT Operations (Legacy Mainframe ↔ Cloud Microservice Boundary)  
**Component**: AWS Lambda Automated Boundary Mitigation & Circuit Breaker Layer  
**Last Updated**: September 2026  

---

## 1. Executive Summary & Verification Status

The BACCP automated mitigation layer acts as an autonomous protective circuit breaker at the integration gateway separating legacy mainframe transaction engines from modern cloud microservices. When elevated synchronization drift $\epsilon(t)$ or impending cascading outage risk is forecasted by the Cascade Predictor, the trained Proximal Policy Optimization (PPO) reinforcement learning agent determines proportional rate-limiting or boundary isolation actions. The AWS Lambda action layer in [`backend/cloud/lambda_handler.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/lambda_handler.py) executes this mitigation policy idempotently and safely.

### Current Verification Status Matrix

| Environment / Mode | Operational Status | Verification Evidence | Notes |
| :--- | :---: | :--- | :--- |
| **Local Simulation** | **LOCAL VERIFIED** | 8/8 automated smoke test scenarios passing; genuine PPO model execution via [`circuit_breaker_ppo.pt`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/weights/circuit_breaker_ppo.pt); sub-millisecond local response; full schema compliance. | Recorded in [`results/lambda_smoke_test.json`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/results/lambda_smoke_test.json). |
| **Live AWS Lambda** | **PENDING CREDENTIALS** | Complete Lambda-compatible handler implemented; boto3 invocation client implemented; structured input/output contracts, retries, and fallback verified. | **Not invoked against live AWS** due to absence of production AWS credentials in local environment. |

> [!IMPORTANT]
> **Strict Verification Guarantee**:
> In accordance with project integrity standards, local handler simulations are **never** reported as live AWS deployments. Live AWS deployment is formally classified as **PENDING CREDENTIALS**.

---

## 2. End-to-End Mitigation Flow

```
┌─────────────────────────────────────────────────────────┐
│             RGCN Multi-Task Cascade Predictor           │
│        (ai-models/cascade_predictor / SageMaker)        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    Predicted Signals                    │
│   • cascade_probability ∈ [0.0, 1.0]                    │
│   • boundary_sync_drift ε(t) ∈ [0.0, 100.0]%            │
│   • estimated_lead_time_seconds                         │
│   • predicted_root_cause_node (boundary-gateway)        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│           PPO Reinforcement Learning Mitigator          │
│       (ai-models/circuit-breaker/ppo_agent.py)          │
│                                                         │
│   State: [P_cascade, drift, latency, err, thr, queue]   │
│   Policy: Continuous Action a ∈ [-1.0, 1.0]             │
│   Priority: reservations (0.5) > crew (0.3) > baggage   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                     PPO Action State                    │
│   • a < -0.33  ──► CLOSED     (throttle_rate = 0.0)     │
│   • -0.33..0.33 ─► THROTTLED  (throttle_rate ∈ [0.1..0.9])
│   • a > 0.33   ──► OPEN       (throttle_rate = 1.0)     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│               Structured Mitigation Payload             │
│   • event_type: "cascade_mitigation"                    │
│   • affected_service: "reservations"                    │
│   • gateway: "boundary-gateway"                         │
│   • cascade_probability, severity, sync_drift           │
│   • recommended_action & throttle_rate                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                   AWS Lambda Handler                    │
│              (backend/cloud/lambda_handler.py)          │
├─────────────────────────────────────────────────────────┤
│ 1. Schema Validation (validate_lambda_event)            │
│ 2. Idempotency Check (CircuitBreakerManager)            │
│ 3. State Actuation (CLOSED / THROTTLED / OPEN)          │
│ 4. Metric Emission (CloudWatchPublisher)                │
│ 5. Alert Dispatch (SNSPublisher, suppressed if repeat)  │
│ 6. Safe Default Fallback (if model/input fails)         │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                  Structured Output Response             │
│   {                                                     │
│     "statusCode": 200,                                  │
│     "action": "THROTTLED",                              │
│     "throttle_rate": 0.3537,                            │
│     "reason": "PPO Policy action=THROTTLED ...",        │
│     "severity": "HIGH",                                 │
│     "cascade_probability": 0.68,                        │
│     "gateway": "boundary-gateway",                      │
│     "timestamp": "2026-09-22T17:32:31Z",                │
│     "idempotent_noop": false                            │
│   }                                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Handler Contract Specifications

### Input Event Schema (`validate_lambda_event`)

The handler accepts both direct dictionary payloads and JSON-serialized strings (compatible with Amazon API Gateway and AWS EventBridge):

```json
{
  "event_type": "cascade_mitigation",
  "affected_service": "reservations",
  "gateway": "boundary-gateway",
  "cascade_probability": 0.68,
  "severity": "HIGH",
  "boundary_sync_drift": 58.0,
  "lead_time": 110.0,
  "recommended_action": "THROTTLE",
  "throttle_rate": 0.35,
  "use_ppo": true
}
```

| Field Name | Type | Constraints | Description |
| :--- | :---: | :--- | :--- |
| `event_type` | string | Optional | Identifies event source (default: `cascade_mitigation`). |
| `affected_service` | string | Optional | Primary airline service tier affected (default: `reservations`). |
| `gateway` | string | Optional | Target integration boundary gateway (default: `boundary-gateway`). |
| `cascade_probability` | float | $[0.0, 1.0]$ | Forecasted cascade probability from RGCN model. |
| `boundary_sync_drift` | float | $[0.0, 100.0]$ | Synchronization drift metric $\epsilon(t)$ in percent. |
| `severity` | string | `NOMINAL`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` | Disruption severity tier. |
| `lead_time` | float | $\ge 0.0$ | Estimated warning time horizon in seconds. |
| `recommended_action` | string | `CLOSED`, `THROTTLE`, `THROTTLED`, `OPEN`, `ISOLATE`, `RESET`, `AUTO`, `PPO` | Mitigation action requested. |
| `use_ppo` | boolean | Optional | Set to `true` to force PPO evaluation if action not predetermined. |

---

### Output Response Schema

Every successful execution returns HTTP 200 with the full contract payload:

```json
{
  "statusCode": 200,
  "action": "THROTTLED",
  "throttle_rate": 0.3537,
  "reason": "PPO Policy action=THROTTLED (rate=35%) under cascade_prob=0.68, drift=58.0%.",
  "severity": "HIGH",
  "cascade_probability": 0.68,
  "gateway": "boundary-gateway",
  "timestamp": "2026-09-22T17:32:31.344123+00:00",
  "idempotent_noop": false,
  "repeat_count": 0,
  "engine_mode": "trained-ppo",
  "affected_service": "reservations",
  "current_breaker_state": {
    "state": "THROTTLED",
    "throttle_rate": 0.3537,
    "target_node": "boundary-gateway",
    "history_count": 5
  },
  "body": "{\"status\": \"success\", \"action\": \"THROTTLED\", ...}"
}
```

---

## 4. Key Resilience & Operational Features

### 1. Trained PPO Inference (`PPOCircuitBreakerInference`)
- Automatically loads the trained Actor-Critic checkpoint [`ai-models/weights/circuit_breaker_ppo.pt`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/ai-models/weights/circuit_breaker_ppo.pt).
- State normalization maps system signals into $[-1.0, 1.0]$ bounds before evaluation.
- Preserves airline service prioritization: sheds non-critical baggage traffic, buffers crew scheduling, and safeguards reservations.

### 2. Idempotent Mitigation
- Repeated mitigation events arriving within a 30-second window with compatible throttle rates ($\le 10\%$ difference) are flagged as `idempotent_noop: true`.
- State is preserved and `repeat_count` is incremented.
- **Alert Flood Prevention**: Redundant duplicate SNS notifications are suppressed to prevent operator pager fatigue during ongoing incidents.

### 3. Safe Default Behavior (Fail-Safe)
- **Validation Failure**: Malformed events immediately return HTTP 400 with a detailed error message and retain the safe existing breaker state without crashing.
- **Model Checkpoint Missing**: If PPO weights are unavailable, the handler falls back immediately to calibrated safety rules based on drift $\epsilon(t)$ and probability $P$.
- **Invalid Model Output**: Catches non-finite (NaN / Inf) action values and recovers via the safe rule fallback.

---

## 5. Automated Smoke Test Scenarios

The smoke test in [`backend/cloud/lambda_smoke_test.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/lambda_smoke_test.py) tests all 8 core operational paths:

| Test Case | Scenario Description | Expected Action | Recorded Output | Status |
| :--- | :--- | :---: | :---: | :---: |
| **1. Nominal State** | Low drift (10.5%), low cascade risk (3%) | `CLOSED` (0%) | `CLOSED` (0.0%), `trained-ppo` | **PASS** |
| **2. High-Risk State** | Medium drift (58%), elevated risk (68%) | `THROTTLED` | `THROTTLED` (35.4%), `trained-ppo` | **PASS** |
| **3. Critical Cascade** | High drift (95%), saturated queue (95%), risk (99%) | `OPEN` (100%) | `OPEN` (100.0%), `trained-ppo` | **PASS** |
| **4. Malformed Event** | Out-of-bounds probability ($P = 2.5$) | HTTP 400 | HTTP 400 (`ValidationError`) | **PASS** |
| **5. Missing Fields** | Minimal empty dictionary `{}` | HTTP 200 | Safe default closed state | **PASS** |
| **6. Repeated Mitigation** | Immediate duplicate high-risk event | Idempotent No-Op | `idempotent_noop: true`, repeat #1 | **PASS** |
| **7. Model Unavailable** | Simulated missing PPO checkpoint | Safe Rule Default | `OPEN` (`safe-rule-fallback`) | **PASS** |
| **8. Invalid PPO Output** | Simulated NaN action output | Safe Recovery | `OPEN` (`safe-rule-fallback`) | **PASS** |

Execution Command:
```bash
python3 backend/cloud/lambda_smoke_test.py
```
Output Report: [`results/lambda_smoke_test.json`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/results/lambda_smoke_test.json).

---

## 6. Step-by-Step Live AWS Lambda Deployment Guide

When production AWS credentials become available, follow these steps to provision the live Lambda function:

### Step 1: Package Function Archive
```bash
mkdir -p build/lambda
cp backend/cloud/lambda_handler.py build/lambda/
cp backend/cloud/config.py build/lambda/
cp backend/cloud/cloudwatch.py build/lambda/
cp backend/cloud/sns.py build/lambda/
cp ai-models/circuit-breaker/ppo_agent.py build/lambda/
cp ai-models/circuit-breaker/env.py build/lambda/
cp ai-models/circuit-breaker/reward.py build/lambda/
cp ai-models/weights/circuit_breaker_ppo.pt build/lambda/

cd build/lambda && zip -r ../baccp-circuit-breaker-mitigator.zip . && cd ../..
```

### Step 2: Create IAM Role & Lambda Function
```bash
# 1. Create execution role
aws iam create-role \
  --role-name BACCP-Lambda-ExecutionRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

# 2. Attach basic execution and CloudWatch policies
aws iam attach-role-policy \
  --role-name BACCP-Lambda-ExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# 3. Create function
aws lambda create-function \
  --function-name baccp-circuit-breaker-mitigator \
  --runtime python3.11 \
  --role arn:aws:iam::123456789012:role/BACCP-Lambda-ExecutionRole \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://build/baccp-circuit-breaker-mitigator.zip \
  --timeout 10 \
  --memory-size 256
```

### Step 3: Run Live Smoke Verification
```bash
python3 backend/cloud/lambda_smoke_test.py
```
Once deployed to AWS with configured credentials, the status in `results/lambda_smoke_test.json` will update to **`LIVE_AWS_VERIFIED`**.
