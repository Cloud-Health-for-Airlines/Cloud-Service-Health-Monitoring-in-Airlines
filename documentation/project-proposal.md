# Finalized Project Idea

## Boundary-Aware Cross-Generation Cascade Prediction & Mitigation for Airline Cloud–Mainframe Infrastructure (working name: **BACCP** — *Boundary-Aware Cross-generation Cascade Predictor*)

---

## 1. The One-Sentence Pitch

A monitoring and self-protection system that treats the **legacy-mainframe-to-cloud-microservice boundary** as a first-class, explicitly-typed part of the dependency graph — auto-discovered via kernel-level (eBPF) tracing because legacy systems can't be instrumented any other way — and uses that graph to **predict cascading failures before they cross generations**, then **automatically throttles the exact boundary node** causing risk.

This is the idea all three of your documents were independently circling. The two lit reviews land on the same gap from different angles (legacy/cloud boundary monitoring, predictive vs. reactive detection); the 5-domain paper set supplies the exact mechanisms — none of which, individually, is new, but the specific combination targeting *this* problem is.

---

## 2. Why This Is the Right Idea (not just *a* idea)

Cross-referencing all three documents:

- **Airline lit review** — identifies the gap directly: no reviewed work (Krasnovsky/Zorkin, Li, Tang/I³, Barua/Kaiser, Zhang survey) treats a legacy/cloud technology-generation boundary as a distinct thing to monitor. I³ (Tang et al.) proves cross-*domain* heterogeneous graphs work for cascade prediction (electric/road/comms/building) — but never applies the heterogeneity axis to *technology generation* within one stack. Barua & Kaiser propose exactly the airline microservices architecture to monitor, but assume a clean rebuild and have zero observability layer.
- **Phase-1 lit review** — independently arrives at "predictive over reactive" and "no unified observability + ML framework for airline reservations" as the gap, and grounds it in real AWS tooling (CloudWatch, X-Ray, SageMaker, Lambda, SNS) — this becomes your implementation substrate.
- **5-domain paper set** — supplies the actual mechanisms:
  - **Paper 2 (ZeroTracer, eBPF)** → solves the *literal blocker*: you cannot inject a sidecar or SDK into a 1980s CICS/COBOL mainframe. Kernel-level, zero-instrumentation tracing at the *gateway* (MQ, CICS transaction gateway, EDI/batch-to-API adapter) is the only way to observe that edge at all.
  - **Paper 3 (DQN+A3C adaptive rate limiting)** → gives you a validated, production-tested (500M req/day) mechanism for automatically throttling a node once risk is detected — repurposed here to act *specifically* at boundary/gateway nodes rather than generically per-service.
  - **Paper 1 (6G Digital Twin sync)** → its formal state-synchronization math (Φ(t)/Ψ(t), sync error ε(t)) is directly reusable as the "legacy state ↔ cloud state" drift metric for the boundary node itself — a quantifiable "boundary health score."
  - **Paper 4 (FedMon, federated eBPF)** → optional but valuable extension: if this were shared across multiple airlines or business units without exposing PNR/PII, federated learning on encrypted model updates (not raw data) is the mechanism — matches the airline lit review's explicit PNR/PII concern.
  - **Paper 5 (Chaos engineering / K8s resilience via failure injection)** → gives you the rigorous evaluation methodology (z-score normalized resilience metrics, systematic fault-intensity sweeps) to validate the system without needing real airline outage data.
  - Krasnovsky & Zorkin's model-discovery approach (from the airline review) + Li's attention-GNN + cascade-probability model form the graph-construction and prediction backbone.

No paper in either literature set combines auto-discovery of un-instrumentable legacy boundaries + generation-typed heterogeneous graphs + cascade prediction calibrated to boundary-crossing + boundary-scoped automated mitigation. That combination is the inventive step.

---

## 3. What Makes This "Patent-Worthy" (Novelty Argument, Not Legal Advice)

*Caveat up front: I'm not a patent attorney and this isn't legal advice — a real patentability opinion needs prior-art search and a lawyer. What follows is a novelty argument you can use in your report, and genuinely worth a provisional filing if your university supports it.*

Patents protect **specific, non-obvious technical combinations that solve a concrete problem** — not the general idea of "monitor the cloud with AI." The defensible, specific claims here would be:

1. **A method for constructing a heterogeneous dependency graph in which node type is defined by technology generation** (legacy-mainframe / hybrid-gateway / cloud-native) as a distinct dimension from service domain — extending prior heterogeneous-graph cascade models (which type nodes by physical infrastructure domain, e.g., I³) into a new typing axis nobody has published.
2. **A method for discovering and instrumenting the specific edge type where a legacy, non-instrumentable system connects to a cloud-native system, using kernel-level (eBPF) socket/TCP event correlation at the integration gateway** rather than application-level tracing — because the legacy side categorically cannot carry a trace header or run a sidecar.
3. **A cascade-probability model with an explicit "generation-gap" edge feature** (protocol type, batch-vs-real-time semantics, legacy latency variance) used to predict the probability and estimated time of a failure crossing the boundary, prior to it manifesting on the cloud side (or vice versa) — an extension of attention/temporal GNN cascade models (Li 2026) with a feature nobody has added.
4. **A method for automated, RL-based circuit-breaking scoped specifically to boundary/gateway nodes**, triggered by the above cascade-probability score, as opposed to generic per-service rate limiting — a targeted application of DRL rate-limiting (Lyu et al.) to a node class it wasn't designed for.

The combination of (1)–(4) as one pipeline, applied to the airline legacy/cloud problem, is what you'd frame as the novel system in a provisional patent claim or invention disclosure — check with your university's tech transfer office if you want to pursue this for real; many universities will file a provisional for free on behalf of a strong student project.

---

## 4. System Architecture

```
Legacy Mainframe Core (CICS / COBOL / batch)
        │
        │  (no SDK, no sidecar possible)
        ▼
[ Integration Gateway Layer: MQ / EDI / batch-to-API adapters ]
        │
        │  eBPF socket + TCP tracepoints (Paper 2 mechanism)
        ▼
┌─────────────────────────────────────────────┐
│  Generation-Typed Dependency Graph Builder    │
│  (auto-discovery, no manual modeling —        │
│   Krasnovsky/Zorkin-style trace extraction)   │
│  Node types: legacy | boundary-gateway | cloud│
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  Cascade-Probability Engine                   │
│  Attention-weighted temporal GNN (Li 2026)    │
│  + generation-gap edge features               │
│  + boundary sync-drift score (Paper 1 math)   │
└─────────────────────────────────────────────┘
        │
        ├──► Predictive alert (before manifestation)
        │      → SNS / dashboard (Phase-1 AWS stack)
        │
        └──► Boundary-Scoped Adaptive Circuit Breaker
               (DQN+A3C, Paper 3 — scoped to gateway nodes only)
        │
        ▼
Cloud Microservices Layer (reservations, crew, baggage)
```

Implementation substrate (from your Phase-1 review): AWS CloudWatch + X-Ray for telemetry collection, SageMaker for the GNN/RL training and inference, Lambda for the circuit-breaker action layer, SNS for alerting.

---

## 5. Evaluation Plan (no real airline data needed)

Following Paper 5's methodology directly:

- **Testbed:** DeathStarBench-style microservices (reservations, crew, baggage domains) + a simple mainframe emulator or stubbed legacy service (COBOL/CICS emulator, or even a deliberately un-instrumentable black-box service reachable only via a gateway) to stand in for the legacy core.
- **Fault injection:** Chaos Mesh or equivalent, at the gateway layer specifically — network delay, connection drop, batch-job stall — at multiple intensity levels (25/50/75/100%, per Paper 5).
- **Metrics:** cascade-prediction lead time (seconds before manifestation — this is your key novel metric, since no reviewed paper reports it for a legacy/cloud boundary), precision/recall/F1 of predicted vs. observed cascades (comparable to Li 2026's 93.2% F1 baseline to benchmark against), z-score normalized resilience/volatility (Paper 5's method), and boundary-specific SLA compliance after circuit-breaker activation (comparable to Lyu et al.'s 98.7% SLA figure).
- **Baseline comparisons:** (a) flat/homogeneous graph with no generation typing (papers 1–2 from the airline review, as negative control), (b) domain-typed but not generation-typed heterogeneous graph (I³-style), (c) your full generation-typed + boundary-aware system.

---

## 6. Advanced ML Components (added to the AI/ML workstream)

Four additional techniques extend the core GNN + RL pipeline, each grounded in current (2025–2026) research and each adding a distinct, defensible capability rather than complexity for its own sake. Full detail and suggested build order: `ai-models/README.md`.

1. **Temporal Point Process module (Neural Hawkes Process / RMTPP)** — the GNN predicts whether a cascade will cross the boundary; a temporal point process models the self-exciting event dynamics to estimate *when*, giving the project's headline "lead time before manifestation" metric a rigorous probabilistic foundation instead of an informal threshold.
2. **Multi-Agent RL circuit breaker (MAPPO)** — replaces the single-agent DQN+A3C breaker with one coordinated agent per boundary/gateway node. This directly closes a gap the original DRL rate-limiting paper's authors (Lyu et al.) named themselves as unsolved future work: single-service optimization can locally throttle one node while starving a dependent one, which is exactly the cascading-failure scenario this project targets.
3. **LLM Explanation Agent** — a retrieval-augmented LLM agent that turns the pipeline's raw output (cascade probability, estimated lead time, affected boundary node, mitigation action taken) into a human-readable incident brief, in the style of current agentic AIOps systems (AWS DevOps Agent, Azure SRE Agent both reached general availability in March 2026; academic multi-agent RCA systems like RCAFlow follow the same pattern). This is the most demo-friendly addition.
4. **Conformal Prediction calibration layer** — a model-agnostic, distribution-free wrapper around the GNN/Hawkes output that gives every cascade alert a statistically-valid confidence bound and an explicit false-alarm-rate guarantee, without retraining the underlying model. Cheapest of the four to add, and the one that makes every other component's output trustworthy enough to act on automatically.

**Novelty rationale:** the multi-agent RL circuit breaker is the strongest of the four specifically because it closes a gap the original authors identified in their own paper — this is a very citable claim for the report. The temporal point process module is the second-strongest, since it upgrades the project's core predictive claim from a probability score to a calibrated timing estimate.

## 7. What to Say in the Report

Frame it exactly as your airline lit review's gap section already does — this project is not a new technique in isolation, it's the first system to combine validated techniques from digital twins, eBPF observability, adaptive rate limiting, and heterogeneous cascade-prediction GNNs, purpose-built for the one structural constraint none of them individually addresses: **you cannot instrument a legacy mainframe the way you instrument a cloud service, and existing cascade-prediction work quietly assumes you can.**
