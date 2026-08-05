# Architecture / Framework Diagram

This diagram is written in [Mermaid](https://mermaid.js.org/), which GitHub renders natively when viewing this file in the repository — no image export needed.

```mermaid
flowchart TB
    subgraph Legacy["Legacy Mainframe Core"]
        L1[CICS / COBOL Transaction Processing]
        L2[Batch Jobs]
    end

    subgraph Gateway["Integration Gateway Layer"]
        G1[MQ / EDI / Batch-to-API Adapter]
        G2["eBPF Socket + TCP Tracepoints\n(zero application instrumentation)"]
    end

    subgraph GraphBuilder["Generation-Typed Dependency Graph Builder"]
        D1["Automated trace-based graph discovery"]
        D2["Node typing: legacy | boundary-gateway | cloud-native"]
    end

    subgraph Predictor["Cascade-Probability Engine"]
        P1["Heterogeneous / relational GNN\n(RGCN or HGT — generation-typed message passing)"]
        P1b["Continuous-time dynamic graph memory (TGN/DySAT)"]
        P2["Generation-gap as relation type (not just edge feature)"]
        P3["Boundary sync-drift score + topological features"]
        P4["Multi-task head: probability | lead-time | next node | severity"]
    end

    subgraph Timing["Temporal Point Process Module"]
        T1["Neural Hawkes Process"]
        T2["Cascade timing / lead-time estimate"]
    end

    subgraph Calibration["Conformal Prediction Calibration"]
        CP1["Sliding-window split conformal calibration"]
        CP2["Statistically-guaranteed confidence bound"]
    end

    subgraph Explain["LLM Explanation Agent"]
        E1["Retrieval-augmented LLM agent"]
        E2["Natural-language incident brief"]
    end

    subgraph Mitigation["Boundary-Scoped RL Circuit Breaker"]
        M1["PPO / SAC agent(s) — replaces DQN+A3C baseline"]
        M2["Multi-objective / constrained reward\n(throughput, cascade risk, critical-path latency, isolation cost)"]
        M3["(stretch) Hierarchical multi-agent coordination across gateways"]
        M4["Throttle / isolate boundary-gateway node(s)"]
    end

    subgraph Cloud["Cloud Microservices Layer"]
        C1[Reservations Service]
        C2[Crew Scheduling Service]
        C3[Baggage Handling Service]
    end

    subgraph Observability["AWS Observability & Alerting"]
        O1[CloudWatch + X-Ray]
        O2[SageMaker - model serving]
        O3[Lambda - action layer]
        O4[SNS - alerts]
    end

    L1 --> G1
    L2 --> G1
    G1 --> G2
    G2 --> D1
    D1 --> D2
    D2 --> P1
    P1 --> P1b
    P1b --> P2
    P1 --> P3
    P2 --> P4
    P3 --> P4
    P4 --> T1
    T1 --> T2
    P4 --> CP1
    T2 --> CP1
    CP1 --> CP2
    CP2 --> E1
    E1 --> E2
    CP2 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> G1
    E2 --> O4
    D2 --> C1
    D2 --> C2
    D2 --> C3
    C1 --> O1
    C2 --> O1
    C3 --> O1
    P1 --> O2
    M1 --> O3
    O3 --> O4
```

## Component Notes

| Component | Purpose | Reference technique |
|---|---|---|
| eBPF tracepoints | Observe the legacy/cloud boundary without instrumenting the mainframe | Zero-instrumentation kernel tracing |
| Graph Builder | Auto-construct dependency graph from trace data, typed by technology generation | Automated model-discovery + heterogeneous graph typing |
| **Cascade-Probability Engine** | Predict cross-generation cascade probability, timing, location, and severity | Heterogeneous/relational GNN (RGCN/HGT) + continuous-time dynamic graph (TGN/DySAT) + multi-task head |
| Boundary sync-drift score | Quantify legacy/cloud state divergence at the gateway | Digital-twin state-synchronization formulation + topological centrality features |
| Temporal Point Process Module | Estimate *when* a cascade will cross the boundary, not just whether | Neural Hawkes Process / RMTPP (self-exciting event intensity) |
| Conformal Prediction Calibration | Give every alert a statistically-valid confidence bound; bound false-alarm rate | Split conformal prediction, sliding-window temporal calibration |
| LLM Explanation Agent | Turn raw graph/timing/RL output into a human-readable incident brief | Retrieval-augmented LLM agent, agentic AIOps pattern |
| **RL Circuit Breaker** | Coordinate throttling across boundary nodes so a local fix doesn't starve a dependent service | PPO/SAC + multi-objective reward design; hierarchical multi-agent coordination (stretch) |
| AWS layer | Telemetry, model serving, automated action, alerting | CloudWatch, X-Ray, SageMaker, Lambda, SNS |

## Novelty rationale (summary — full tiered detail in `ai-models/README.md`)

- **Heterogeneous/relational message passing** is the highest-value single upgrade: it directly answers the literature survey's own gap statement, since no reviewed work types message-passing weights by technology generation.
- **Hierarchical multi-agent RL coordination** (stretch goal) is the strongest mitigation-side claim: it directly closes a gap the original DRL rate-limiting paper's authors named themselves as unsolved future work.
- **Multi-task prediction head + calibrated uncertainty** turns a single probability score into an operationally useful statement ("87% confident a cascade starts at Gateway-X in ~4.2 min"), which is a genuinely useful airline-operator-facing capability, not just a metrics-table improvement.
- The full set of Tier 1/Tier 2 options, their complexity, and the suggested build order live in `ai-models/README.md` — this diagram shows the recommended core path, not every option under consideration.
