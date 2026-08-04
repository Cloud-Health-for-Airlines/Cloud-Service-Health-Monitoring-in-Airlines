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

    subgraph Mitigation["Boundary-Scoped Adaptive Circuit Breaker"]
        M1["DQN + A3C hybrid RL agent"]
        M2["Throttle / isolate boundary-gateway node"]
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
    P2 --> M1
    P3 --> M1
    M1 --> M2
    M2 --> G1
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
| Circuit Breaker | Automatically throttle/isolate the boundary node before cascade spreads | RL-based adaptive rate limiting, scoped to boundary nodes |
| AWS layer | Telemetry, model serving, automated action, alerting | CloudWatch, X-Ray, SageMaker, Lambda, SNS |
