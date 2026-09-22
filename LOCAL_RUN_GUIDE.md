# BACCP: Local Execution Guide & System Manual

**Boundary-Aware Cross-Generation Cascade Predictor for Airline IT Systems**  
**AI/ML Engineering Owner:** Varad  

---

## 1. Project Architecture & How It Works

BACCP is designed to solve a critical operational vulnerability in hybrid airline enterprise IT: **cross-generation cascade failures**.

```
[ Tier 1: Mainframe Core ]        [ Tier 2: Boundary Gateway ]       [ Tier 3: Cloud Microservices ]
  IBM z/OS IMS / DB2        <--->   ActiveMQ / Envoy Proxy     <--->   Reservations (Booking Checkout)
  (Synchronous COBOL/DB2)          (Translation & Protocol)            Crew Scheduling (FAA Duty)
                                                                       Baggage Reconciliation
```

### Why Failures Cascade Across Generations
1. **Queue Saturation & Latency Asymmetry:** Legacy mainframe transactions operate on fixed-thread synchronous clocks, while cloud-native microservices operate on asynchronous sub-second events.
2. **Mainframe Batch Contention:** Nightly batch processing or network delays stall the boundary gateway's TCP connections.
3. **Queue Backlog Explosion:** As the gateway transit buffer saturates, retry storms cascade backwards and forwards, causing downstream reservation and ticketing thread pools to exhaust.

### The BACCP Solution
1. **Generation-Typed Graph Schema:** Represents services as typed nodes (`legacy`, `boundary-gateway`, `cloud-native`) and edges as typed relations (`sync_to`, `sync_from`, `route_to`, `call_gateway`).
2. **Boundary Synchronization Drift Metric $\epsilon(t)$:** Measures queuing depth, message round-trip asymmetry, and error rates at the gateway in real-time.
3. **Tier 1A Relational GNN (Hetero-RGCN):** Uses relation-specific transformation matrices $W_r$ per technology gap to predict cascade hazards.
4. **Tier 1C Multi-Task Head:** Predicts cascade probability, warning lead time (mean warning: **~42.2 seconds**), root cause node, and outage severity.
5. **Tier 1D & 2E Explainability & Conformal Calibration:** Computes attention attributions across inter-service edges and wraps probabilities in certifiably sound **90% Split Conformal coverage intervals**.
6. **Tier 1F & 1G Continuous PPO RL Circuit Breaker:** Uses a continuous Beta Actor-Critic policy to dynamically shed gateway load, optimizing an airline multi-objective reward prioritizing passenger booking revenue (`reservations: 1.5` > `crew: 1.0` > `baggage: 0.7`).
7. **Tier 2A Multi-Agent PPO (MAPPO):** Coordinates multiple regional gateways (e.g. JFK & LHR) sharing a centralized mainframe core to prevent domino oscillations.

---

## 2. Quick Start: Interactive Terminal Demo

You can run the live interactive dashboard right in your terminal without starting any servers:

```bash
# Interactive mode with live controls (press 1-6 to inject faults):
python run_demo.py

# Or automated 15-second simulation scenario:
python run_demo.py --auto
```

### Live Interactive Menu Controls
- Press `[1]` to inject **Network Delay (500ms)**: Watch boundary drift $\epsilon(t)$ climb and the PPO circuit breaker actuate `THROTTLED`.
- Press `[2]` to inject **Mainframe Batch Stall (15s)**: Watch gateway buffer saturate, hazard jump to CRITICAL, and the circuit breaker actuate `OPEN`.
- Press `[3]` to inject **Connection Drop**.
- Press `[4]` to **Clear Faults**: Watch the digital twin drift recover to nominal (11.5%) and the circuit breaker reset to `CLOSED`.
- Press `[6]` to execute the **Tier 2A Multi-Agent PPO (MAPPO)** multi-gateway test.

---

## 3. Running the Full System Locally (Backend + Frontend)

### Step 1: Start the Backend REST API Server
The backend is a zero-dependency REST server running on Python standard library:

```bash
# From repository root:
python backend/api/app.py --port 8000
```
Server runs on: **`http://localhost:8000`**

#### Key Endpoints:
- `GET  /api/health` — Boundary health score, drift $\epsilon(t)$, active faults, and circuit breaker status.
- `GET  /api/graph` — Canonical 5-node generation-typed dependency graph.
- `POST /api/predict` — Invokes the trained **Tier 1A Hetero-RGCN** model to return cascade probability, conformal bounds, and natural language explanation.
- `POST /api/simulate/chaos` — Injects chaos faults into the digital twin (`{"fault": "network-delay", "level": "high"}`).
- `POST /api/circuit-breaker/multi-agent` — Runs the **Tier 2A MAPPO** multi-gateway coordination engine.
- `GET  /metrics` — **Prometheus-compatible metrics endpoint** for production cloud monitoring.

---

### Step 2: Start the Frontend Monitoring Dashboard
The frontend is built with React 18 and Vite 5:

```bash
# In a new terminal:
cd frontend
npm install
npm run dev
```
Open your browser at: **`http://localhost:5173`**

The dashboard will connect to `http://localhost:8000` and display:
- Real-time SVG topological generation dependency graph
- Radial Boundary Synchronization Drift $\epsilon(t)$ gauge
- Live prediction lead-time warning alerts
- Interactive Chaos Injection buttons (`Network Delay`, `Batch Job Stall`, `Connection Drop`)
- Automated Circuit Breaker state & mitigation actuation log

---

### Step 3: Standalone Presentation Runner
If you want to view the slide deck in your browser without Node.js:

- Simply double-click or open [presentation/presentation.html](file:///f:/project/Cloud-Service-Health-Monitoring-in-Airlines-main/Cloud-Service-Health-Monitoring-in-Airlines-main/presentation/presentation.html) in Chrome, Edge, or Firefox.
- Navigate using keyboard arrow keys (`Left` / `Right`) or on-screen controls.

---

## 4. Machine Learning Pipeline Execution

All machine learning components are fully self-contained and reproducible:

```bash
# 1. Regenerate Dataset (60 experimental runs: 35 cascades, 25 nominal):
python ai-models/data/generate_dataset.py --mode=synthetic --runs=60

# 2. Train Cascade Predictors (Tier 0 Baseline GAT+GRU & Tier 1A Hetero-RGCN):
python ai-models/evaluation/train_cascade_predictor.py

# 3. Train Circuit Breakers (Tier 0 DQN & Tier 1F Continuous PPO):
python ai-models/evaluation/train_circuit_breaker.py

# 4. Run Full Evaluation & Generate Benchmark Reports:
python ai-models/evaluation/evaluate.py
```
Generated artifacts will be updated in:
- [results/evaluation_metrics.json](file:///f:/project/Cloud-Service-Health-Monitoring-in-Airlines-main/Cloud-Service-Health-Monitoring-in-Airlines-main/results/evaluation_metrics.json)
- [results/ai-models-evaluation.md](file:///f:/project/Cloud-Service-Health-Monitoring-in-Airlines-main/Cloud-Service-Health-Monitoring-in-Airlines-main/results/ai-models-evaluation.md)

---

## 5. Automated Test Suites

```bash
# 1. Run all Backend & Cloud Integration Tests (22/22 Passing):
python -m unittest backend/tests/test_backend.py

# 2. Run Architecture Unit Tests (4/4 Passing):
python -m unittest ai-models/tests/test_tier0_arch.py
```

---

## 6. Python Package Import Usage

You can import all components into any external script using standard Python syntax:

```python
# Standard idiomatic imports via ai_models bridge:
from ai_models.graph_builder import GenerationTypedGraph, compute_sync_drift_score
from ai_models.cascade_predictor import HeteroGNN, predict
from ai_models.circuit_breaker import PPOAgent, MAPPOAgent, LagrangianSafetyFilter, choose_action

# 1. Compute Boundary Drift:
drift = compute_sync_drift_score(queue_depth=45.0, latency_ms=180.0, error_rate=0.02)
print(f"Drift Score: {drift:.1f}%")

# 2. Run Cascade Inference:
prediction = predict(graph_data=..., sync_drift_score=drift)
print(f"Hazard: {prediction['cascade_probability']:.1%}, CI: {prediction['conformal_bounds']}")

# 3. Run Circuit Breaker:
action, rate, reason = choose_action(prediction['cascade_probability'], drift)
print(f"Action: {action} (Throttle {rate:.0%}): {reason}")
```

---

## 7. What You Can Do Next (Roadmap to Scale to Enterprise)

1. **Deploy to Physical Kubernetes & Mainframe:**
   - Package the backend as a container using the Dockerfile.
   - Attach `testbed/discovery/bpftrace/tcp_v4_connect.bt` in a privileged DaemonSet on Linux Kubernetes nodes to stream live eBPF telemetry directly into the graph builder.
2. **AWS Production Deployment:**
   - Host `weights/cascade_predictor_tier1a.pt` on an AWS SageMaker Asynchronous Multi-Model Endpoint.
   - Deploy `backend/cloud/lambda_handler.py` as a serverless Lambda function attached to an AWS EventBridge rule or CloudWatch alarm.
   - Configure SNS topic ARNs to page airline operational command centers via PagerDuty and Slack webhooks.
