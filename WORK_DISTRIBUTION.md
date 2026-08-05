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
