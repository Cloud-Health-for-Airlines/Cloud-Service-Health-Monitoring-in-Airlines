# BACCP — Boundary-Aware Cross-Generation Cascade Predictor

### Predictive Cascading-Failure Detection & Mitigation at the Legacy-Mainframe / Cloud-Microservice Boundary in Airline IT Systems

> Suggested repo names (pick one): `baccp-airline-cascade-predictor`, `boundary-aware-cascade-monitor`, `skyboundary`. This README assumes the name **baccp-airline-cascade-predictor**.

---

## Team Members

| Name | Role |
|---|---|
| Ragghav | Data & Dependency Graph Engineering (eBPF tracing, graph construction, testbed) |
| Varad | AI/ML Engineering (cascade-prediction GNN, RL circuit breaker) |
| Bhiwanshu | Cloud Integration, Backend/Frontend & Documentation |

*(See [`WORK_DISTRIBUTION.md`](./WORK_DISTRIBUTION.md) for the full task breakdown — the above is the high-level ownership split.)*

---

## Problem Statement

Airlines run mission-critical reservation, crew-scheduling, and baggage-handling logic on a mix of decades-old mainframe systems and newer cloud-native microservices. Existing cloud/microservice monitoring and cascading-failure-prediction research treats the whole system as one uniform, "modern" dependency graph — it has no concept of the legacy/cloud technology-generation boundary, and no way to observe it, because legacy mainframe components generally cannot be instrumented with SDKs or sidecars the way cloud services can. As a result, failures that originate in or cross through this boundary (as in real incidents such as the Southwest 2022 meltdown, the 2024 Delta/CrowdStrike disruption, and the January 2026 United outage) are detected reactively, after they have already cascaded across domains, rather than predicted before they manifest.

## Objectives

1. Auto-discover the legacy-mainframe ↔ cloud-microservice integration boundary (MQ / EDI / batch-to-API gateways) using zero-instrumentation, kernel-level (eBPF) tracing — without modifying or instrumenting the legacy side.
2. Construct a **generation-typed heterogeneous dependency graph** that distinguishes legacy, boundary-gateway, and cloud-native nodes as a first-class typing dimension (not just service domain).
3. Predict the probability and estimated lead time of a cascading failure crossing the legacy/cloud boundary, before it manifests, using an attention-weighted temporal graph neural network with boundary-specific ("generation-gap") edge features.
4. Automatically mitigate predicted cascades with a reinforcement-learning circuit breaker scoped specifically to boundary/gateway nodes.
5. Validate the system on a synthetic airline testbed (reservations / crew / baggage microservices + an emulated legacy core) using systematic chaos-engineering fault injection, since real airline operational data is not publicly available.

## Proposed Architecture / Framework

See [`architecture/architecture-diagram.md`](./architecture/architecture-diagram.md) for the full diagram and component breakdown. Summary:

```
Legacy Mainframe Core → Integration Gateway (eBPF-traced) → Generation-Typed Dependency
Graph Builder → Cascade-Probability GNN → Temporal Point Process (timing) → Conformal
Calibration (confidence bound) → LLM Explanation Agent (incident brief) + Boundary-Scoped
Multi-Agent RL Circuit Breaker → Cloud Microservices Layer (Reservations / Crew / Baggage)
```

**Advanced ML components** (tiered roadmap — recommended upgrades and stretch goals — in [`ai-models/README.md`](./ai-models/README.md)):
- **Heterogeneous/relational GNN (RGCN/HGT)** — types message-passing weights by technology generation (legacy/gateway/cloud), not just as an edge feature; directly answers the literature survey's named gap.
- **Continuous-time dynamic graph modeling (TGN/DySAT)** — updates node memory per eBPF event rather than on fixed snapshots, better suited to irregular legacy batch-job timing vs. cloud request traffic.
- **Multi-task prediction head** — cascade probability, lead time, most-likely failure location, and severity from one shared model.
- **Calibrated uncertainty & explainability** — attention visualization plus temperature scaling / evidential deep learning, so alerts come with a stated confidence.
- **PPO/SAC circuit breaker with multi-objective reward** — replaces the DQN+A3C baseline; stretch goals include hierarchical multi-agent coordination across gateways and safe/constrained RL.
- Additional stretch goals: conformal prediction calibration, an LLM explanation agent, self-supervised pre-training, causal/counterfactual modeling, and lead-time-aware evaluation metrics.

## Technology Stack

| Layer | Technology |
|---|---|
| Kernel-level tracing | eBPF (socket/TCP tracepoints) |
| Dependency graph & cascade model | Python, PyTorch / PyTorch Geometric — RGCN/HGT (heterogeneous message passing) + TGN/DySAT (continuous-time) |
| Cascade timing | Neural Hawkes Process / RMTPP, or folded into the TGN continuous-time model |
| Alert calibration | Temperature scaling / evidential deep learning + split conformal prediction |
| Explanation layer | LLM agent (function-calling / retrieval-augmented) — stretch goal |
| Circuit breaker | PPO / SAC (baseline upgrade); hierarchical multi-agent (MAPPO) as stretch goal |
| Cloud observability | Amazon CloudWatch, AWS X-Ray |
| ML training/serving | Amazon SageMaker |
| Automation / alerting | AWS Lambda, Amazon SNS |
| Chaos engineering / fault injection | Chaos Mesh (or equivalent) on a Kubernetes testbed |
| Backend services | Python (FastAPI) or Node.js — microservice simulators |
| Frontend dashboard | React |
| Database | PostgreSQL (graph/metadata), time-series store (e.g., InfluxDB/Timestream) for telemetry |

## Dataset Details

No public real-world airline operational dataset exists, so the project uses a **synthetic testbed**, consistent with the evaluation approach of the reviewed literature:

- **Microservice layer:** DeathStarBench-style benchmark application, adapted to model reservation / crew-scheduling / baggage-handling domains.
- **Legacy layer:** An emulated mainframe/CICS-style stub service reachable only through the integration gateway (deliberately non-instrumentable at the application level, to force reliance on eBPF).
- **Fault injection data:** Generated via systematic chaos-engineering experiments (network delay, connection drop, batch-job stall) at multiple intensity levels, following the methodology in the reviewed chaos-engineering literature.
- **Literature-derived reference datasets** (for baseline comparison only, not direct reuse): DeathStarBench Social Network (Krasnovsky & Zorkin, 2025), CWRU Bearing dataset (Chakma & Choi, 2025 — for digital-twin sync-metric validation only).

## Repository Structure

```
.
├── README.md
├── WORK_DISTRIBUTION.md
├── architecture/          → canonical architecture, Mermaid diagrams, contracts
├── frontend/              → monitoring dashboard (React 18, Vite 5 & standalone runner)
├── backend/               → REST API server & dual-mode AWS cloud wiring (22 tests)
├── ai-models/             → cascade-prediction GNN, RL circuit breaker
├── database/              → schema.sql, synthetic data generators
├── documentation/         → project proposal, research gaps, system design, evaluation plan
├── testbed/               → chaos-engineering fault injection & eBPF discovery setup
├── results/               → evaluation outputs, metrics, plots
└── presentation/          → 18-slide Marp deck & interactive HTML presentation runner
```

## Quickstart & Verification

```bash
# 1. Run Backend Unit Tests (CloudWatch, X-Ray, SageMaker, Lambda, SNS)
python3 -m unittest backend/tests/test_backend.py

# 2. Launch the BACCP REST API & Cloud Integration Server (Port 8000)
python3 backend/api/app.py

# 3. View the Monitoring Dashboard (Zero Dependencies / Instant Browser Preview)
open frontend/dist_preview/index.html

# 4. View the Interactive Presentation Deck
open presentation/presentation.html

# 5. Read the Master Academic Technical Report
open documentation/phase1-comprehensive-report.md
```

