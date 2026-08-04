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
Graph Builder → Cascade-Probability GNN → Predictive Alerts + Boundary-Scoped RL Circuit
Breaker → Cloud Microservices Layer (Reservations / Crew / Baggage)
```

## Technology Stack

| Layer | Technology |
|---|---|
| Kernel-level tracing | eBPF (socket/TCP tracepoints) |
| Dependency graph & cascade model | Python, PyTorch / PyTorch Geometric (GAT + GRU temporal GNN) |
| Circuit breaker | DQN + A3C hybrid (PyTorch / Stable-Baselines3) |
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
├── architecture/          → architecture & framework diagram
├── frontend/               → monitoring dashboard (React)
├── backend/                → gateway simulators, API services
├── ai-models/              → cascade-prediction GNN, RL circuit breaker
├── database/                → schema, synthetic data generators
├── documentation/          → literature survey, research gaps, project proposal
├── testbed/                → chaos-engineering fault injection setup
├── results/                → evaluation outputs, metrics, plots
└── presentation/            → slide deck
```

Each folder contains its own `README.md` describing its purpose and current status.
