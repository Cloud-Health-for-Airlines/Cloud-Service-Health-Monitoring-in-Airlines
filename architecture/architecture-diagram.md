# Architecture & Framework Diagrams

This document contains canonical Mermaid diagrams representing the actual BACCP system. All diagrams strictly distinguish between:
1. **[Locally Verified]** — Fully implemented, trained with real weights, and verified via automated tests in the local environment.
2. **[Live AWS Pending Credentials]** — Fully packaged and integration-tested adapter; live AWS deployment pending provision of production AWS IAM credentials.
3. **[Planned]** — Long-term production extensions (e.g. multi-node EKS cluster, federated cross-carrier learning).

---

## 1. Architecture Overview (Operational Hierarchy)

```mermaid
flowchart TD
    classDef existing fill:#1a365d,stroke:#3182ce,stroke-width:2px,color:#ffffff;
    classDef verified fill:#064e3b,stroke:#38a169,stroke-width:2px,color:#ffffff;
    classDef aws_pending fill:#744210,stroke:#d69e2e,stroke-width:2px,stroke-dasharray: 4 4,color:#ffffff;
    classDef planned fill:#2d3748,stroke:#a0aec0,stroke-width:1px,stroke-dasharray: 2 2,color:#cbd5e0;

    subgraph Tier0["Airline Operational Domain"]
        OPS["Airline Operations & Passenger Traffic [Locally Verified]"]
    end

    subgraph Tier1["Cloud-Native Microservices (cloud-native)"]
        RES["Reservations Service :8081 [Locally Verified]"]
        CREW["Crew Scheduling Service :8082 [Locally Verified]"]
        BAG["Baggage Handling Service :8083 [Locally Verified]"]
    end

    subgraph Tier2["Integration Boundary (boundary-gateway)"]
        GW["Integration Gateway :8084 [Locally Verified]\n(HTTP-to-TCP Protocol Transition & Rate Limiting)"]
    end

    subgraph Tier3["Legacy Mainframe Core (legacy)"]
        MF["Legacy Mainframe Core :9090 [Locally Verified]\n(CICS / COBOL Simulation, Internal Isolated Network)"]
    end

    subgraph Tier4["Kernel-Level Non-Intrusive Observability"]
        EBPF["eBPF Socket / TCP kprobe [Locally Verified]\n(tcp_v4_connect.bt + collect.py)"]
    end

    subgraph Tier5["Topological Discovery & Graph Structuring"]
        DEP["Dependency Graph Construction [Locally Verified]\n(aggregate.py / canonical graph schema)"]
    end

    subgraph Tier6["Predictive Intelligence Engine"]
        GNN["HeteroRGCN Predictor [Locally Verified]\n(cascade_predictor_tier1a.pt, 362 KB)"]
        LIVE_GNN["SageMaker Endpoint [Live AWS Pending Credentials]\n(ai-models/deploy/sagemaker/model.tar.gz)"]
        PROB["Cascade Probability Head [Locally Verified]\n(100% Recall, 82.24% Precision, 0.9026 F1)"]
        LEAD["Lead-Time & Conformal Head [Locally Verified]\n(176.8s Mean Warning + 90% Conformal CI)"]
    end

    subgraph Tier7["Automated Mitigation & Incident Notification"]
        LCB["Lambda Circuit Breaker [Locally Verified]\n(PPOCircuitBreakerInference + Idempotency)"]
        LIVE_RL["Trained PPO Policy [Locally Verified]\n(circuit_breaker_ppo.pt, 42.7 KB, 87.9% Res. Retained)"]
        SNS["SNS Alert Publisher [Locally Verified]\n(Structured Alert with 8 Attributes)"]
        ACT["Mitigation Action [Locally Verified]\n(Dynamic Gateway Throttling / Boundary Isolation)"]
    end

    %% Edge Connections
    OPS --> RES
    OPS --> CREW
    OPS --> BAG
    
    RES -->|HTTP:8080| GW
    CREW -->|HTTP:8080| GW
    BAG -->|HTTP:8080| GW
    
    GW -->|TCP:9090| MF
    
    EBPF -.->|Kernel Probe| GW
    EBPF -->|Raw Socket Events| DEP
    
    DEP --> GNN
    GNN -.-> LIVE_GNN
    GNN --> PROB
    PROB --> LEAD
    
    LEAD -->|High / Critical Risk| LCB
    LCB --> LIVE_RL
    LEAD -->|High / Critical Risk| SNS
    
    LCB -->|Actuate Throttle / Isolate| ACT
    ACT -->|429 Fast-Fail Rate Limiting| GW

    class OPS,RES,CREW,BAG,GW,MF,EBPF,DEP,SNS,ACT existing;
    class GNN,PROB,LEAD,LCB,LIVE_RL verified;
    class LIVE_GNN aws_pending;
```

---

## 2. Component Diagram

```mermaid
graph TD
    classDef existing fill:#1a365d,stroke:#3182ce,stroke-width:2px,color:#ffffff;
    classDef verified fill:#064e3b,stroke:#38a169,stroke-width:2px,color:#ffffff;
    classDef aws_pending fill:#744210,stroke:#d69e2e,stroke-width:2px,stroke-dasharray: 4 4,color:#ffffff;
    classDef planned fill:#2d3748,stroke:#a0aec0,stroke-width:1px,stroke-dasharray: 2 2,color:#cbd5e0;

    subgraph ROOT["Repository Component Structure"]
        subgraph F_FRONTEND["frontend/ [Locally Verified]"]
            FE_DASH["React 18 Dashboard: App.jsx, components/*"]
            FE_ADAPT["Resilient API Adapter: api/adapter.js"]
            FE_CFG["Dynamic Config: config.js (VITE_API_URL)"]
        end

        subgraph F_BACKEND["backend/ [Locally Verified]"]
            BE_API["REST Server: api/app.py"]
            BE_CFG["Config Manager: cloud/config.py (.env.example)"]
            BE_CW["CloudWatch: cloud/cloudwatch.py (8 metrics)"]
            BE_XR["X-Ray: cloud/xray.py (6 operations context)"]
            BE_SM["SageMaker Adapter: cloud/sagemaker.py [Dual-Mode Verified]"]
            BE_LM["Lambda Mitigator: cloud/lambda_handler.py [PPO Action Layer]"]
            BE_SNS["SNS Publisher: cloud/sns.py (8 alert attributes)"]
            BE_ORCH["Cloud Orchestrator: cloud/orchestrator.py"]
            BE_TEST["Test Suite: tests/test_backend.py (35 tests)"]
            BE_E2E["E2E Runner: tests/verify_end_to_end.py (76 tests)"]
        end

        subgraph F_TESTBED["testbed/ [Implemented]"]
            TB_DC["Docker Compose: docker-compose.yml"]
            TB_SVC["Services: services/cloud-service, boundary-gateway, legacy-core"]
            TB_BPF["eBPF Discovery: discovery/bpftrace/tcp_v4_connect.bt, collect.py"]
            TB_CHAOS["Chaos Engine: chaos/chaos.py, network.sh, profiles.json"]
        end

        subgraph F_DB["database/ [Implemented]"]
            DB_SCH["PostgreSQL Schema: schema.sql (nodes, edges, telemetry)"]
            DB_SEED["Seeder: seed.py"]
        end

        subgraph F_AIMODELS["ai-models/ [Locally Verified]"]
            AI_GNN["Trained HeteroRGCN: cascade_predictor_tier1a.pt (362 KB)"]
            AI_RL["Trained PPO Agent: circuit_breaker_ppo.pt (42.7 KB)"]
            AI_BASE["Baseline Models: baseline_flat_graph.pt & domain_typed.pt"]
            AI_TEST["Test Suite: tests/test_*.py (18 tests)"]
            AI_EVAL["Evaluation Harness: evaluation/run_full_evaluation.py"]
        end
    end

    class FE_DASH,FE_ADAPT,FE_CFG,BE_API,BE_CFG,BE_CW,BE_XR,BE_SNS,BE_ORCH,BE_TEST,BE_E2E,TB_DC,TB_SVC,TB_BPF,TB_CHAOS,DB_SCH,DB_SEED existing;
    class BE_SM,BE_LM,AI_GNN,AI_RL,AI_BASE,AI_TEST,AI_EVAL verified;
```

---

## 3. Data-Flow Diagram

```mermaid
sequenceDiagram
    autonumber
    participant App as Airline Microservices (cloud-native)
    participant Gateway as Boundary Gateway (boundary-gateway)
    participant Kernel as eBPF Probe (tcp_v4_connect.bt)
    participant Legacy as Legacy Mainframe (legacy)
    participant Backend as BACCP Cloud Orchestrator
    participant CloudWatch as Amazon CloudWatch
    participant XRay as AWS X-Ray
    participant SageMaker as SageMaker Adapter
    participant Lambda as Lambda Mitigation Client
    participant SNS as Amazon SNS Alert
    participant Dashboard as React SRE Dashboard

    App->>Gateway: HTTP REST Requests (Booking / Baggage / Crew)
    Gateway->>Legacy: Forward over TCP:9090
    Kernel-->>Gateway: kprobe:tcp_v4_connect captures TCP socket latency & connect time
    Legacy-->>Gateway: TCP Response
    Gateway-->>App: HTTP 200 OK

    Note over Gateway,Kernel: Fault Injected via Chaos Engine (network-delay / connection-drop)
    Gateway->>Legacy: TCP Connection Stalled / Dropped
    Kernel-->>Backend: High TCP connect RTT & socket timeouts captured
    
    Backend->>Backend: Compute State Synchronization Drift: ε(t) = ||Φ(t) - Ψ(t)|| / ||Φ(t)|| * 100
    Backend->>CloudWatch: Publish boundary_health_score, service_latency, error_rate
    Backend->>XRay: Record cross-generation trace segment: cloud -> gateway -> legacy
    
    Backend->>SageMaker: predict_cascade(graph, telemetry_features, boundary_features, ε(t))
    Note over SageMaker: Multi-Task Head computes P(cascade), Lead Time τ, 90% Conformal Bounds
    SageMaker-->>Backend: Prediction: P=88%, τ=35s, Root Cause=boundary-gateway, Severity=CRITICAL
    
    Backend->>CloudWatch: Publish cascade_probability=0.88, prediction_lead_time=35.0
    
    Note over Backend: Risk Evaluation: P >= 0.75 or ε(t) >= 70% -> Action = OPEN
    Backend->>Lambda: invoke_mitigation(event_type=cascade_mitigation, action=OPEN, gateway=boundary-gateway)
    Lambda->>Gateway: Apply Rate-Limiting / Isolation (Simulated guardrail active)
    Lambda-->>Backend: Status: 200 OK (State: OPEN)
    
    Backend->>SNS: publish_cascade_alert(severity=CRITICAL, probability=0.88, lead_time=35s)
    SNS-->>Backend: Alert Dispatched (Buffered in local mode)
    
    Dashboard->>Backend: GET /api/alerts, GET /api/health/boundary (Polling every 6s)
    Backend-->>Dashboard: Live Alert with Conformal Bounds & Recommended Mitigation
    Dashboard->>Dashboard: Render Critical Warning Badge & Countdown Timer (35s remaining)
```

---

## 4. Cloud Integration Diagram

```mermaid
flowchart TD
    classDef existing fill:#1a365d,stroke:#3182ce,stroke-width:2px,color:#ffffff;
    classDef prototype fill:#744210,stroke:#d69e2e,stroke-width:2px,stroke-dasharray: 4 4,color:#ffffff;
    classDef planned fill:#2d3748,stroke:#a0aec0,stroke-width:1px,stroke-dasharray: 2 2,color:#cbd5e0;

    subgraph LayerApp["1. Application & Testbed Runtime"]
        APP["Airline Services Testbed [Existing]\n(Reservations, Crew, Baggage, Gateway, Legacy Core)"]
        EBPF_APP["eBPF Socket Tracer [Existing]\n(Kernel-space connection latency & error capture)"]
    end

    subgraph LayerObs["2. Observability Ingestion"]
        CW["Amazon CloudWatch Publisher [Existing]\n(BACCP/AirlineCloudHealth Namespace, 8 Metrics)"]
        XR["AWS X-Ray Trace Recorder [Existing]\n(Context-managed trace_operation across 6 paths)"]
    end

    subgraph LayerBackend["3. Backend Integration Layer"]
        CFG["Centralized Config [Existing]\n(CLOUD_MODE: local | aws)"]
        ORCH["BACCP Cloud Orchestrator [Existing]\n(Telemetry -> Predict -> Mitigate -> Alert)"]
        REST["BACCP REST API Server [Existing]\n(backend/api/app.py :8000)"]
    end

    subgraph LayerPredict["4. Model Inference Adapter"]
        SM_ADAPT["SageMaker Model Adapter [Existing Adapter / Prototype Engine]\n(Input: graph, telemetry, boundary features)"]
        SM_LOCAL["Calibrated Analytical Fallback [Prototype]\n(Conformal bounds: 90% confidence interval)"]
        SM_LIVE["SageMaker Realtime Endpoint [Planned]\n(Hosting ai-models/ PyTorch RGCN weights)"]
    end

    subgraph LayerMitigate["5. Mitigation Layer"]
        LAMBDA_CLI["Lambda Mitigation Client [Existing / Prototype]\n(Structured cascade_mitigation payload)"]
        LAMBDA_SIM["Local Simulation Handler [Existing]\n(Simulates Lambda, updates CircuitBreakerManager)"]
        LAMBDA_LIVE["Live AWS Lambda Function [Planned Integration]\n(AWS_LAMBDA_FUNCTION_NAME)"]
    end

    subgraph LayerAlert["6. Notification Layer"]
        SNS_PUB["Amazon SNS Publisher [Existing]\n(8 required alert attributes + local buffer)"]
        SNS_TOPIC["AWS SNS Topic ARN [Existing in aws mode]\n(AWS_SNS_TOPIC_ARN)"]
    end

    subgraph LayerUI["7. Monitoring Presentation"]
        DASH["BACCP React Dashboard [Existing]\n(System Overview, Dependency Graph, Drift Gauge, Alerts)"]
    end

    %% Data Connections
    APP -->|Telemetry & Metrics| CW
    APP -->|Cross-boundary traces| XR
    EBPF_APP -->|State drift ε(t)| ORCH
    
    CFG --> ORCH
    ORCH --> CW
    ORCH --> XR
    ORCH --> SM_ADAPT
    
    SM_ADAPT --> SM_LOCAL
    SM_ADAPT -.->|AWS Mode| SM_LIVE
    
    SM_ADAPT -->|Prediction: P, τ, severity| ORCH
    
    ORCH -->|Risk Evaluation: High/Critical| LAMBDA_CLI
    LAMBDA_CLI --> LAMBDA_SIM
    LAMBDA_CLI -.->|AWS Mode| LAMBDA_LIVE
    
    ORCH -->|Risk Evaluation: High/Critical| SNS_PUB
    SNS_PUB -.->|AWS Mode| SNS_TOPIC
    
    REST --> DASH
    CW -.->|Metrics API| REST
    ORCH -.->|Pipeline status| REST

    class APP,EBPF_APP,CW,XR,CFG,ORCH,REST,SM_ADAPT,LAMBDA_CLI,LAMBDA_SIM,SNS_PUB,SNS_TOPIC,DASH existing;
    class SM_LOCAL prototype;
    class SM_LIVE,LAMBDA_LIVE planned;
```

---

## 5. Prediction & Alerting Flow

```mermaid
flowchart TD
    classDef existing fill:#1a365d,stroke:#3182ce,stroke-width:2px,color:#ffffff;
    classDef prototype fill:#744210,stroke:#d69e2e,stroke-width:2px,stroke-dasharray: 4 4,color:#ffffff;
    classDef planned fill:#2d3748,stroke:#a0aec0,stroke-width:1px,stroke-dasharray: 2 2,color:#cbd5e0;

    IN["Input Telemetry Snapshot [Existing]\n- Graph Nodes & Edges\n- Service Latencies & Error Rates\n- Boundary Sync Drift ε(t)"] --> PREDICT

    subgraph Inference["Model Inference Layer"]
        PREDICT["SageMaker Cascade Predictor [Prototype]\n(Local Analytical Engine / Live Endpoint)"]
        CALIB["Split Conformal Calibration [Prototype]\n1 - α = 0.90 Coverage Guarantee"]
    end

    PREDICT --> CALIB
    CALIB --> EVAL["Risk & Severity Evaluation [Existing]"]

    EVAL -->|P >= 0.75 OR ε(t) >= 70%| CRIT["CRITICAL SEVERITY [Existing]\nLead Time: 8s - 45s"]
    EVAL -->|P >= 0.55 OR ε(t) >= 45%| HIGH["HIGH SEVERITY [Existing]\nLead Time: 30s - 120s"]
    EVAL -->|P >= 0.35| MED["MEDIUM SEVERITY [Existing]\nLead Time: 90s - 240s"]
    EVAL -->|P < 0.35| LOW["LOW / NOMINAL [Existing]\nLead Time: > 300s"]

    CRIT --> ACT_OPEN["Lambda Mitigation: Action = OPEN [Prototype]\nThrottle Rate: 100% (Full Boundary Isolation)"]
    HIGH --> ACT_THROTTLE["Lambda Mitigation: Action = THROTTLED [Prototype]\nDynamic Throttle Rate: 30% - 75%"]
    MED --> ACT_WARN["Dashboard Warning Alert [Existing]\nCircuit Breaker: CLOSED"]
    LOW --> ACT_RESET["Nominal Operation [Existing]\nCircuit Breaker: CLOSED (Reset)"]

    ACT_OPEN --> DISPATCH_SNS["SNS Alert Dispatcher [Existing]\n(Severity: CRITICAL, Urgency: Immediate)"]
    ACT_THROTTLE --> DISPATCH_SNS_HIGH["SNS Alert Dispatcher [Existing]\n(Severity: HIGH, Urgency: High)"]
    
    ACT_OPEN --> CW_METRIC["CloudWatch: publish_circuit_breaker_action [Existing]"]
    ACT_THROTTLE --> CW_METRIC

    class IN,EVAL,CRIT,HIGH,MED,LOW,ACT_WARN,ACT_RESET,DISPATCH_SNS,DISPATCH_SNS_HIGH,CW_METRIC existing;
    class PREDICT,CALIB,ACT_OPEN,ACT_THROTTLE prototype;
```

---

## 6. Multi-Tier Deployment View

```mermaid
flowchart TB
    classDef container fill:#1a365d,stroke:#3182ce,stroke-width:2px,color:#ffffff;
    classDef host fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#ffffff;
    classDef aws fill:#744210,stroke:#d69e2e,stroke-width:2px,color:#ffffff;

    subgraph DockerHost["Physical / VM Linux Docker Host"]
        direction TB

        subgraph NetCloud["Docker Network: cloud-tier (Bridge)"]
            C_RES["reservations:8080\n(Host Port 8081)"]
            C_CREW["crew:8080\n(Host Port 8082)"]
            C_BAG["baggage:8080\n(Host Port 8083)"]
        end

        subgraph NetDual["Dual-Homed Gateway Container"]
            C_GW["boundary-gateway:8080\n(Host Port 8084)\nConnected to cloud-tier AND legacy-tier"]
        end

        subgraph NetLegacy["Docker Network: legacy-tier (Internal Isolated)"]
            C_LEG["legacy-core:9090\n(Unpublished, unreachable from cloud microservices)"]
        end

        subgraph NetData["Docker Network: data-tier (Internal Isolated)"]
            C_DB["PostgreSQL 16:5432\n(DB: baccp, schema.sql)"]
        end

        subgraph HostProcesses["Host Kernel & Application Processes"]
            K_BPF["Linux Kernel eBPF (kprobe:tcp_v4_connect)"]
            B_API["BACCP Backend REST Server :8000 (Python 3)"]
            F_UI["BACCP React Dashboard :3000 (Vite / Node.js)"]
        end

        C_RES -->|HTTP:8080| C_GW
        C_CREW -->|HTTP:8080| C_GW
        C_BAG -->|HTTP:8080| C_GW
        C_GW -->|TCP:9090| C_LEG
        K_BPF -.->|Trace kernel connections| C_GW
        B_API -->|Read topology & seed| C_DB
        F_UI -->|REST Polling 6s| B_API
    end

    subgraph AWSCloud["AWS Cloud VPC (Enabled in CLOUD_MODE=aws)"]
        AWS_CW["Amazon CloudWatch\n(Namespace: BACCP/AirlineCloudHealth)"]
        AWS_XR["AWS X-Ray Daemon :2000 (UDP)"]
        AWS_SM["Amazon SageMaker\n(Endpoint: baccp-cascade-predictor)"]
        AWS_LM["AWS Lambda Function\n(baccp-circuit-breaker-mitigator)"]
        AWS_SNS["Amazon SNS Topic\n(baccp-cascade-alerts)"]
    end

    B_API -.->|Metrics API| AWS_CW
    B_API -.->|Trace UDP| AWS_XR
    B_API -.->|Invoke Endpoint| AWS_SM
    B_API -.->|Invoke Function| AWS_LM
    B_API -.->|Publish Topic| AWS_SNS

    class C_RES,C_CREW,C_BAG,C_GW,C_LEG,C_DB,K_BPF,B_API,F_UI container;
    class DockerHost,HostProcesses host;
    class AWS_CW,AWS_XR,AWS_SM,AWS_LM,AWS_SNS aws;
```
