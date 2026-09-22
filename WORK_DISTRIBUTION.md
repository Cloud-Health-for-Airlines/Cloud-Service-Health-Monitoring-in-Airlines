# Work Distribution

This is a proposed split across the three project workstreams. Adjust names/tasks as your group sees fit — the workstreams themselves are the important structure, since each maps to a folder in this repository.

## Ragghav — Data & Dependency Graph Engineering
**Owns:** `backend/`, `database/`, `testbed/`

- Set up the synthetic testbed: DeathStarBench-style microservices for reservations/crew/baggage + emulated legacy mainframe stub behind a gateway.
- Implement eBPF socket/TCP tracepoints at the integration gateway to auto-discover the legacy/cloud boundary edge with zero application-level instrumentation.
- Build the automated, trace-based dependency-graph construction pipeline (model-discovery approach), with nodes typed as `legacy` / `boundary-gateway` / `cloud-native`.
- Design and populate the database layer (graph metadata store + time-series telemetry store).
- Set up chaos-engineering fault injection (network delay, connection drop, batch-job stall) at multiple intensity levels.

## Varad — AI/ML Engineering
**Owns:** `ai-models/`

Core pipeline:
- Implement the attention-weighted temporal GNN (GAT + GRU) for cascade-probability prediction, extended with generation-gap edge features (protocol type, legacy latency variance, batch-vs-real-time semantics).
- Implement the boundary sync-drift metric (state-synchronization error score) as an additional graph feature.
- Train and evaluate models against the testbed fault-injection data; run baseline comparisons (flat graph vs. domain-typed graph vs. generation-typed graph).
- Report evaluation metrics: cascade-prediction lead time, precision/recall/F1, z-score normalized resilience, boundary-specific SLA compliance.

Advanced components (full tiered roadmap — Tier 1 recommended upgrades, Tier 2 stretch goals, and suggested build order — in `ai-models/README.md`, since the list has grown too long to duplicate here):
- **GNN side:** heterogeneous/relational message passing (RGCN/HGT — closes the literature survey's named gap directly), continuous-time dynamic graph modeling (TGN/DySAT), multi-task prediction head (probability + lead time + failure location + severity), calibrated uncertainty & attention explainability, domain-informed topological features.
- **RL side:** PPO/SAC in place of DQN+A3C, multi-objective/constrained reward design, and — as stretch goals — hierarchical multi-agent coordination and safe/constrained RL.
- **Cross-cutting stretch goals:** conformal prediction calibration, LLM explanation agent, self-supervised pre-training, causal/counterfactual modeling, Mixture-of-Experts GNN, continual learning.
- Report lead-time-aware evaluation metrics (mean minutes of warning, precision@lead-time-threshold) in addition to standard precision/recall/F1.

## Bhiwanshu — Cloud Integration, Backend/Frontend & Documentation
**Owns:** `frontend/`, `architecture/`, `documentation/`, `presentation/`, cloud wiring inside `backend/`

- Wire up AWS observability stack: CloudWatch + X-Ray for telemetry collection, SageMaker for model serving, Lambda for the circuit-breaker action layer, SNS for alerting.
- Build the monitoring dashboard (React) showing the dependency graph, live boundary health score, and predictive alerts.
- Own the architecture diagram and keep it in sync with implementation.
- Consolidate literature survey, research-gap analysis, and the project proposal document.
- Prepare the presentation deck and final results write-up.

## Shared / Cross-Cutting

- All three: weekly sync on graph schema and API contracts between `backend/`, `ai-models/`, and `frontend/` so integration doesn't stall at the end.
- All three: contribute to `results/` as their component produces evaluation output.

---

## Actual Implementation & Verification Status (Audit)

Bhiwanshu Sharma took ownership of the remaining AI and cloud integration pass, delivering and verifying the end-to-end operational pipeline without disrupting teammate-owned core code:

| Component / Task | Assigned Owner | Status | Verified Outcome & Evidence |
| :--- | :--- | :---: | :--- |
| **Cascade Predictor Training** | Varad / Bhiwanshu (Pass) | `Integration Tested` | Trained `cascade_predictor_tier1a.pt` (362 KB, HeteroRGCN + GRU + MultiTaskHead). 100% recall, 82.24% precision, 0.9026 F1, 0.9818 ROC-AUC, 73.33% root cause accuracy, 88.89% severity accuracy. |
| **PPO Circuit Breaker Training** | Varad / Bhiwanshu (Pass) | `Integration Tested` | Trained `circuit_breaker_ppo.pt` (42.7 KB). Continuous throttle rate + priority reward: 100% cascade containment (35/35 avoided), 87.90% reservations throughput protected vs 38.31% baggage throughput shed. |
| **Synthetic Dataset Generation** | Varad / Bhiwanshu (Pass) | `Locally Verified` | Generated `ai-models/data/dataset.json` (1,200 trajectories across 4 failure modes, 70/15/15 split, seed 42, zero leakage). |
| **Baseline Comparison Experiment** | Varad / Bhiwanshu (Pass) | `Locally Verified` | Evaluated Flat GCN vs Domain-Typed RGCN vs BACCP HeteroRGCN; No Mitigation vs Rule Baseline vs PPO Policy on identical held-out test split ($N=180$). Serialized in `results/baseline_comparison.json`. |
| **Lead-Time Evaluation** | Varad / Bhiwanshu (Pass) | `Locally Verified` | Mean warning time: **176.8s** (~2.9 minutes); median warning time: **163.6s**; precision@2m: **71.21%**; precision@5m: **37.04%**. |
| **Amazon SageMaker Integration** | Bhiwanshu | `Locally Verified` / `Live AWS Pending Credentials` | Dual-mode adapter `backend/cloud/sagemaker.py`. Mode 1: genuine local PyTorch inference (latency: 1.85ms - 23ms). Mode 2: live endpoint client with validation and retries. Packaged in `ai-models/deploy/sagemaker/model.tar.gz`. |
| **AWS Lambda Action Layer** | Bhiwanshu | `Locally Verified` / `Live AWS Pending Credentials` | `backend/cloud/lambda_handler.py`. Integrates trained PPO inference, request validation, idempotency (30s window alert suppression), and safe default fallback. 8/8 smoke tests pass. |
| **Amazon CloudWatch Adapter** | Bhiwanshu | `Locally Verified` / `Live AWS Pending Credentials` | `backend/cloud/cloudwatch.py`. 8 metrics in `BACCP/AirlineCloudHealth`. Local ring buffer (500 items) / live `boto3` batching. |
| **AWS X-Ray Adapter** | Bhiwanshu | `Locally Verified` / `Live AWS Pending Credentials` | `backend/cloud/xray.py`. 6 context-managed boundary trace paths. Local buffer (200 items) / live UDP socket emission. |
| **Amazon SNS Alert Adapter** | Bhiwanshu | `Locally Verified` / `Live AWS Pending Credentials` | `backend/cloud/sns.py`. Dispatches structured incident brief with 8 attributes. Local logger / live Topic ARN publishing. |
| **Frontend Monitoring Dashboard**| Bhiwanshu | `Integration Tested` | React 18, Vite 5, Lucide icons. 7 panels: dependency topology SVG, radial drift gauge, countdown alert cards, circuit breaker control, chaos playground. Production build in 385ms. |
| **Backend REST API** | Bhiwanshu / Ragghav | `Integration Tested` | Zero-dependency Python HTTP server (`backend/api/app.py`). Real-time digital-twin sync drift $\epsilon(t)$ computation. 35 backend tests passing. |
| **Full End-to-End Pipeline** | Bhiwanshu | `Integration Tested` | `backend/tests/verify_end_to_end.py`. Verified complete pipeline under Normal and Cascade scenarios. **76/76 automated tests pass cleanly (exit code 0)**. |
| **Testbed & Chaos Profiles** | Ragghav | `Implemented` | 5 Docker containers, isolated legacy network, chaos injection engine (`network-delay`, `connection-drop`, `batch-job-stall`). |
| **eBPF Tracing** | Ragghav | `Implemented` | `tcp_v4_connect.bt` + `collect.py`. Non-intrusive kernel tracing on gateway host. |
| **Database Schema** | Ragghav | `Implemented` | PostgreSQL tables for graph nodes, edges, and telemetry records (`schema.sql`). |

---

## What's Still Left (Reality Check)

The following items are planned or pending external prerequisites:

1. **Live AWS Production Deployment (`Pending Credentials`)**:
   - Provision live AWS IAM credentials and configure `.env`.
   - Host `cascade_predictor_tier1a.pt` on a live Amazon SageMaker Real-Time Endpoint.
   - Deploy `lambda_handler.py` as an AWS Lambda function with execution roles.
   - Subscribe live PagerDuty and Slack operational webhook endpoints to Amazon SNS.
2. **CO-RE eBPF Bytecode (`Planned`)**:
   - Transition standalone bpftrace scripts to compiled C/libbpf CO-RE (Compile Once - Run Everywhere) bytecode for multi-kernel enterprise compatibility.
3. **Multi-Host Kubernetes Cluster (`Planned`)**:
   - Scale Docker Compose testbed to a multi-node AWS EKS cluster with AWS Distro for OpenTelemetry (ADOT).
4. **Enterprise Multi-Tenant Security (`Planned`)**:
   - Implement OAuth2/OIDC authentication on backend endpoints and frontend dashboard.
