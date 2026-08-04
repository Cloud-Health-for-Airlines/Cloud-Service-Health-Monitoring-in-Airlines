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
        P1["Attention-weighted temporal GNN (GAT + GRU)"]
        P2["Generation-gap edge features"]
        P3["Boundary sync-drift score"]
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

    subgraph Mitigation["Boundary-Scoped Multi-Agent RL Circuit Breaker"]
        M1["Per-boundary-node RL agents (MAPPO)"]
        M2["Shared critic / coordination signal"]
        M3["Throttle / isolate boundary-gateway node(s)"]
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
    P1 --> P2
    P1 --> P3
    P1 --> T1
    T1 --> T2
    P2 --> CP1
    T2 --> CP1
    CP1 --> CP2
    CP2 --> E1
    E1 --> E2
    CP2 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> G1
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
| Cascade-Probability Engine | Predict cross-generation cascade probability & lead time | Attention/temporal GNN + cascade-probability modeling |
| Boundary sync-drift score | Quantify legacy/cloud state divergence at the gateway | Digital-twin state-synchronization formulation |
| **Temporal Point Process Module** | Estimate *when* a cascade will cross the boundary, not just whether | Neural Hawkes Process / RMTPP (self-exciting event intensity) |
| **Conformal Prediction Calibration** | Give every alert a statistically-valid confidence bound; bound false-alarm rate | Split conformal prediction, sliding-window temporal calibration |
| **LLM Explanation Agent** | Turn raw graph/timing/RL output into a human-readable incident brief | Retrieval-augmented LLM agent, agentic AIOps pattern |
| **Multi-Agent RL Circuit Breaker** | Coordinate throttling across boundary nodes so a local fix doesn't starve a dependent service | MAPPO / MADDPG multi-agent reinforcement learning |
| AWS layer | Telemetry, model serving, automated action, alerting | CloudWatch, X-Ray, SageMaker, Lambda, SNS |

## Why these four (novelty rationale)

- **Multi-Agent RL circuit breaker** is the strongest addition: it directly closes a gap the original DRL rate-limiting paper's authors named themselves as unsolved future work (single-service optimization can starve a dependent service — exactly the cascading-failure scenario this project targets).
- **Temporal Point Process module** upgrades "cascade probability" into "cascade probability *and* estimated time," making the project's headline lead-time metric statistically principled rather than informal.
- **Conformal Prediction calibration** is the cheapest addition and the one that makes every other component's output trustworthy — it turns raw model scores into statistically-guaranteed confidence bounds without needing to retrain anything.
- **LLM Explanation Agent** is the most demo-friendly: it's what turns a dashboard full of numbers into a live, understandable incident narrative for a presentation.
