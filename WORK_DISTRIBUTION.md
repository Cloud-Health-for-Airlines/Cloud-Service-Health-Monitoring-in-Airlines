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

Advanced components (see `ai-models/README.md` for full detail and suggested build order):
- **Conformal prediction calibration layer** — wrap the GNN's cascade-probability output in a statistically-guaranteed confidence bound (build this first — cheapest of the four).
- **Temporal point process module (Neural Hawkes / RMTPP)** — estimate cascade *timing*, not just probability, to make the lead-time metric rigorous.
- **LLM explanation agent** — turn pipeline output into a human-readable incident brief; good demo milestone.
- **Multi-agent RL circuit breaker (MAPPO)** — replaces the single-agent DQN+A3C breaker with per-boundary-node agents coordinated via a shared critic, so a local throttle doesn't starve a dependent service. Highest effort of the four — budget the most time here, and highlight in the report that this directly closes a gap the original DRL rate-limiting paper's authors named as future work themselves.
- Report additional evaluation metrics: conformal interval empirical coverage rate, Hawkes-process time-to-cascade prediction error, and global-vs-local optimization comparison (multi-agent RL vs. single-agent baseline).

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
