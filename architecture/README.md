# architecture/

System architecture, topological diagrams, dataflow specifications, and component interface contracts for **BACCP** (*Boundary-Aware Cross-Generation Cascade Predictor*).

## Documents

- [`canonical-architecture.md`](./canonical-architecture.md) — Comprehensive canonical architecture specification with 6 Mermaid diagrams, status badges, and source tree traceability.
- [`architecture-diagram.md`](./architecture-diagram.md) — GitHub-rendered native Mermaid architecture diagrams.

---

## Architectural Component Overview

The architecture models the airline IT ecosystem and is divided into distinct, loosely-coupled layers:

### 1. Technology Generations & Service Topology
- **`cloud-native` (Existing)**: Microservices (`reservations:8081`, `crew:8082`, `baggage:8083`) running in Docker bridge networks. They communicate with the gateway via standard HTTP REST requests.
- **`boundary-gateway` (Existing)**: Dual-homed reverse proxy and rate-limiter (`boundary-gateway:8084`) bridging modern cloud requests to the legacy mainframe protocol. It hosts the circuit breaker mitigation interceptor.
- **`legacy` (Existing)**: Uninstrumented mainframe core (`legacy-core:9090`) processing simulated CICS/COBOL transactions over TCP sockets in an isolated, internal network with no external host port exposure.

### 2. Kernel-Level Non-Intrusive Observability
- **`eBPF` (Existing)**: Low-overhead socket-level kernel probes (`discovery/bpftrace/tcp_v4_connect.bt`) attached to `kprobe:tcp_v4_connect`. Traces TCP connections across generation boundaries without bytecode injection or SDK overhead on the legacy core.
- **`dependency graph` (Existing)**: Discovered graph topology (`discovery/aggregate.py`) identifying caller/callee relationships, ports, request rates, and latencies, persisted to PostgreSQL (`database/schema.sql`).

### 3. Predictive Intelligence & Calibration
- **`cascade prediction` (Prototype Engine / Existing Adapter)**: Multi-task cascade prediction engine (`backend/cloud/sagemaker.py`) predicting cascade probability ($P_{\text{cascade}}$), failure location, severity, and operational lead time ($\hat{\tau}$ seconds).
- **`GNN` (Relational GNN Specification / Planned Phase-II Model)**: Relational Graph Convolutional Network (RGCN) where generation-gap edges are distinct relation matrices. Formalized in `documentation/phase1-comprehensive-report.md` and `ai-models/README.md`; currently executed via calibrated analytical engine.
- **Split Conformal Prediction (Prototype)**: $1 - \alpha = 0.90$ confidence intervals bounding predictive uncertainty before triggering mitigation.

### 4. Automated Mitigation
- **`RL/circuit breaker` (Prototype Client & Simulation / Planned Phase-II Policy)**: Rate-limiting and boundary isolation client (`backend/cloud/lambda_handler.py`). When high/critical cascades are predicted, dynamically adjusts gateway throttle rates ($30\% - 100\%$) to protect the mainframe from cascading collapse. Planned Phase-II integrates safe RL (constrained MDP) from `ai-models/`.

### 5. AWS Cloud Integration Layer
- **`CloudWatch` (Existing)**: Publishes 8 metrics (`boundary_health_score`, `cascade_probability`, `prediction_lead_time`, `service_latency`, `error_rate`, `request_rate`, `circuit_breaker_actions`, `active_alerts`) in namespace `BACCP/AirlineCloudHealth`.
- **`X-Ray` (Existing)**: Context-managed operation tracing across 6 paths (`incoming_api_request`, `telemetry_processing`, `graph_retrieval`, `prediction_request`, `alert_generation`, `circuit_breaker_action`).
- **`SageMaker` (Existing Adapter / Prototype Engine)**: Model serving connector invoking endpoint in AWS mode, or falling back to local analytical prediction in local mode.
- **`Lambda` (Existing Client / Prototype Action)**: Invokes circuit-breaker Lambda with structured cascade mitigation payload.
- **`SNS` (Existing)**: Dispatches high-severity alerts with all 8 required incident attributes to operators and SRE dashboards.

---

## Implementation Status Classification

| Category | Definition | Included Components |
| :--- | :--- | :--- |
| **`[Existing]`** | Fully implemented and operational. Tested by unit & integration suites. | `testbed/services/*`, `testbed/discovery/*`, `testbed/chaos/*`, `database/schema.sql`, `backend/api/app.py`, `backend/cloud/config.py`, `backend/cloud/cloudwatch.py`, `backend/cloud/xray.py`, `backend/cloud/sns.py`, `backend/cloud/orchestrator.py`, `frontend/src/*`. |
| **`[Prototype]`** | Operational prototype / simulated provider. Uses calibrated analytical inference or safety guardrails. | `backend/cloud/sagemaker.py` (Analytical engine with 90% conformal intervals), `backend/cloud/lambda_handler.py` (Simulated breaker client & local handler). |
| **`[Planned]`** | Architectural specification scheduled for Phase-II. | `ai-models/` (Trained PyTorch RGCN weights, continuous Hawkes point process training, online RL PPO policy training). |

---

## Verification & Consistency

All architecture diagrams and contracts have been checked against the actual repository tree:
- **Zero Unimplemented Components Displayed as Existing**: All future components in `ai-models/` are explicitly tagged `[Planned]`.
- **Dual-Mode Portability**: Architecture reflects that the system runs identically locally without AWS credentials via mock buffers, or live on AWS via `boto3`.
- **API Parity**: Every endpoint referenced in the diagrams (`/api/health/boundary`, `/api/alerts`, `/api/graph`, `/api/circuit-breaker/action`, `/api/pipeline/run`, `/api/cloud/status`) is active in `backend/api/app.py`.
