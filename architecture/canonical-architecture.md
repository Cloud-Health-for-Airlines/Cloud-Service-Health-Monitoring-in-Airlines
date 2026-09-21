# BACCP Canonical System Architecture

**Boundary-Aware Cross-Generation Cascade Predictor (BACCP) for Airline IT Systems**

This canonical document specifies the complete system architecture of BACCP. It accurately reflects the **actual repository implementation**, explicitly distinguishing between **Existing Implementation**, **Prototype Implementation**, and **Planned / Future Components**.

---

## 1. Implementation Status Legend

To maintain strict scientific and engineering integrity, all components across the diagrams and specifications are classified into one of three implementation states:

| Status Badge | Meaning | Implementation Location |
| :--- | :--- | :--- |
| `[Existing]` | **Fully implemented and operational** in the codebase. Tested with automated unit/integration suites. | `backend/api/app.py`, `backend/cloud/config.py`, `backend/cloud/cloudwatch.py`, `backend/cloud/xray.py`, `backend/cloud/sns.py`, `backend/cloud/orchestrator.py`, `frontend/src/*`, `testbed/services/*`, `testbed/discovery/*`, `database/schema.sql`. |
| `[Prototype]` | **Functioning working prototype / simulated provider**. Uses calibrated analytical inference or simulated action guardrails pending full model training. | `backend/cloud/sagemaker.py` (Analytical engine with $1-\alpha=0.90$ conformal bounds), `backend/cloud/lambda_handler.py` (simulated local circuit breaker). |
| `[Planned]` | **Architecture-specified design scheduled for Phase-II**. Concrete interface contracts exist, but model weights or production integration are future deliverables. | `ai-models/` (Trained PyTorch RGCN weights, continuous Hawkes point process training, online RL PPO policy training). |

---

## 2. BACCP End-to-End System Architecture

This diagram directly models the operational hierarchy of an airline IT ecosystem: passenger and operations traffic flowing into cloud-native microservices, converging through the boundary integration gateway to the legacy mainframe, traced non-intrusively via eBPF, transformed into generation-typed dependency graphs, analyzed for cascade risk and lead time, and actuating automated circuit breaking and SNS alerts.

```mermaid
flowchart TD
    classDef existing fill:#1a365d,stroke:#3182ce,stroke-width:2px,color:#ffffff;
    classDef prototype fill:#744210,stroke:#d69e2e,stroke-width:2px,stroke-dasharray: 4 4,color:#ffffff;
    classDef planned fill:#2d3748,stroke:#a0aec0,stroke-width:1px,stroke-dasharray: 2 2,color:#cbd5e0;

    subgraph Tier0["Airline Operational Domain"]
        OPS["Airline Operations & Passenger Traffic [Existing]"]
    end

    subgraph Tier1["Cloud-Native Microservices (cloud-native)"]
        RES["Reservations Service :8081 [Existing]"]
        CREW["Crew Scheduling Service :8082 [Existing]"]
        BAG["Baggage Handling Service :8083 [Existing]"]
    end

    subgraph Tier2["Integration Boundary (boundary-gateway)"]
        GW["Integration Gateway :8084 [Existing]\n(HTTP-to-TCP Protocol Transition & Rate Limiting)"]
    end

    subgraph Tier3["Legacy Mainframe Core (legacy)"]
        MF["Legacy Mainframe Core :9090 [Existing]\n(CICS / COBOL Simulation, Internal Isolated Network)"]
    end

    subgraph Tier4["Kernel-Level Non-Intrusive Observability"]
        EBPF["eBPF Socket / TCP kprobe [Existing]\n(tcp_v4_connect.bt + collect.py)"]
    end

    subgraph Tier5["Topological Discovery & Graph Structuring"]
        DEP["Dependency Graph Construction [Existing]\n(aggregate.py / canonical graph schema)"]
    end

    subgraph Tier6["Predictive Intelligence Engine"]
        GNN["GNN Prediction Engine [Prototype]\n(Heterogeneous RGCN Message-Passing Specification)"]
        LIVE_GNN["SageMaker PyTorch Trained Model [Planned]\n(ai-models/ Phase-II Online Training)"]
        PROB["Cascade Probability Computation [Prototype]\nP(cascade) via Sigmoidal Drift Response"]
        LEAD["Risk & Lead-Time Evaluator [Prototype]\n(Neural Hawkes Formulation + 90% Conformal Bounds)"]
    end

    subgraph Tier7["Automated Mitigation & Incident Notification"]
        LCB["Lambda Circuit Breaker Client [Prototype]\n(Structured Mitigation Payload + Rate Limiter)"]
        LIVE_RL["Trained PPO/SAC Policy [Planned]\n(Safe RL Constrained MDP in ai-models/)"]
        SNS["SNS Alert Publisher [Existing]\n(High-Priority Incident Dispatcher)"]
        ACT["Mitigation Action [Existing / Prototype]\n(Dynamic Gateway Throttling / Boundary Isolation)"]
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
    LCB -.-> LIVE_RL
    LEAD -->|High / Critical Risk| SNS
    
    LCB -->|Actuate Throttle / Isolate| ACT
    ACT -->|429 Fast-Fail Rate Limiting| GW

    %% Styling
    class OPS,RES,CREW,BAG,GW,MF,EBPF,DEP,SNS,ACT existing;
    class GNN,PROB,LEAD,LCB prototype;
    class LIVE_GNN,LIVE_RL planned;
```

---

## 3. AWS Observability & Cloud Integration Layer

BACCP incorporates a modular provider/adapter architecture connecting the testbed application to AWS CloudWatch, AWS X-Ray, Amazon SageMaker, AWS Lambda, and Amazon SNS. It operates in dual-mode: **local mode** (zero AWS credentials required, in-memory buffers) and **live AWS mode** (via `boto3`).

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

## 4. Component Architecture & Source Tree Mapping

Every architectural component corresponds to concrete files in the repository:

```mermaid
graph TD
    classDef existing fill:#1a365d,stroke:#3182ce,stroke-width:2px,color:#ffffff;
    classDef prototype fill:#744210,stroke:#d69e2e,stroke-width:2px,stroke-dasharray: 4 4,color:#ffffff;
    classDef planned fill:#2d3748,stroke:#a0aec0,stroke-width:1px,stroke-dasharray: 2 2,color:#cbd5e0;

    subgraph ROOT["Repository Root"]
        subgraph F_FRONTEND["frontend/ [Existing]"]
            FE_DASH["React 18 Dashboard: App.jsx, components/*"]
            FE_ADAPT["Resilient API Adapter: api/adapter.js"]
            FE_CFG["Dynamic Config: config.js (VITE_API_URL)"]
        end

        subgraph F_BACKEND["backend/ [Existing]"]
            BE_API["REST Server: api/app.py"]
            BE_CFG["Config Manager: cloud/config.py (.env.example)"]
            BE_CW["CloudWatch: cloud/cloudwatch.py (8 metrics)"]
            BE_XR["X-Ray: cloud/xray.py (6 operations context)"]
            BE_SM["SageMaker Adapter: cloud/sagemaker.py [Prototype Engine]"]
            BE_LM["Lambda Mitigator: cloud/lambda_handler.py [Prototype Client]"]
            BE_SNS["SNS Publisher: cloud/sns.py (8 alert attributes)"]
            BE_ORCH["Cloud Orchestrator: cloud/orchestrator.py"]
            BE_TEST["Test Suite: tests/test_backend.py (22 tests)"]
        end

        subgraph F_TESTBED["testbed/ [Existing]"]
            TB_DC["Docker Compose: docker-compose.yml"]
            TB_SVC["Services: services/cloud-service, boundary-gateway, legacy-core"]
            TB_BPF["eBPF Discovery: discovery/bpftrace/tcp_v4_connect.bt, collect.py"]
            TB_CHAOS["Chaos Engine: chaos/chaos.py, network.sh, profiles.json"]
        end

        subgraph F_DB["database/ [Existing]"]
            DB_SCH["PostgreSQL Schema: schema.sql (nodes, edges, telemetry)"]
            DB_SEED["Seeder: seed.py"]
        end

        subgraph F_AIMODELS["ai-models/ [Planned]"]
            AI_GNN["Trained PyTorch RGCN: cascade-predictor/ [Planned]"]
            AI_RL["Trained PPO Agent: circuit-breaker/ [Planned]"]
            AI_DOC["Interface Specification: README.md [Existing]"]
        end
    end

    class FE_DASH,FE_ADAPT,FE_CFG,BE_API,BE_CFG,BE_CW,BE_XR,BE_SNS,BE_ORCH,BE_TEST,TB_DC,TB_SVC,TB_BPF,TB_CHAOS,DB_SCH,DB_SEED,AI_DOC existing;
    class BE_SM,BE_LM prototype;
    class AI_GNN,AI_RL planned;
```

---

## 5. End-to-End Data Flow Architecture

The data pipeline processes information from Linux kernel probes through graph derivation, analytical inference, and presentation:

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

## 6. Prediction & Mitigation Decision Flow

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

## 7. Multi-Tier Deployment View

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

---

## 8. Architectural Consistency & Traceability Matrix

Every architectural node has been compared against the actual repository source code and REST APIs:

| Architectural Component | Generation Type | Status | Repository Source File | Verification / Test |
| :--- | :--- | :--- | :--- | :--- |
| **Reservations Service** | `cloud-native` | `[Existing]` | [`testbed/services/cloud-service/app.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/testbed/services/cloud-service/app.py) | Verified via `testbed/docker-compose.yml` (Port 8081) |
| **Crew Service** | `cloud-native` | `[Existing]` | [`testbed/services/cloud-service/app.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/testbed/services/cloud-service/app.py) | Verified via `testbed/docker-compose.yml` (Port 8082) |
| **Baggage Service** | `cloud-native` | `[Existing]` | [`testbed/services/cloud-service/app.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/testbed/services/cloud-service/app.py) | Verified via `testbed/docker-compose.yml` (Port 8083) |
| **Boundary Gateway** | `boundary-gateway` | `[Existing]` | [`testbed/services/boundary-gateway/app.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/testbed/services/boundary-gateway/app.py) | Verified via `testbed/docker-compose.yml` (Port 8084) |
| **Legacy Mainframe** | `legacy` | `[Existing]` | [`testbed/services/legacy-core/app.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/testbed/services/legacy-core/app.py) | Verified via internal network (Port 9090) |
| **eBPF Kernel Probing** | N/A | `[Existing]` | [`testbed/discovery/bpftrace/tcp_v4_connect.bt`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/testbed/discovery/bpftrace/tcp_v4_connect.bt) | Captures socket connect latency without bytecode injection |
| **Dependency Graph** | N/A | `[Existing]` | [`testbed/discovery/aggregate.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/testbed/discovery/aggregate.py) | Generates `dependency-graph.json` with 5 typed nodes |
| **Backend REST API** | N/A | `[Existing]` | [`backend/api/app.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/api/app.py) | Exposes `/api/health`, `/api/graph`, `/api/alerts`, etc. |
| **Cloud Configuration** | N/A | `[Existing]` | [`backend/cloud/config.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/config.py) | Manages `CLOUD_MODE` ('local' / 'aws') and `.env.example` |
| **Amazon CloudWatch** | N/A | `[Existing]` | [`backend/cloud/cloudwatch.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/cloudwatch.py) | Publishes all 8 metrics to `BACCP/AirlineCloudHealth` |
| **AWS X-Ray Tracing** | N/A | `[Existing]` | [`backend/cloud/xray.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/xray.py) | Context-managed operation tracing across 6 paths |
| **SageMaker Adapter** | N/A | `[Prototype]` | [`backend/cloud/sagemaker.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/sagemaker.py) | Analytical engine with 90% conformal intervals |
| **Lambda Mitigator** | N/A | `[Prototype]` | [`backend/cloud/lambda_handler.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/lambda_handler.py) | Simulated circuit breaker with structured payload |
| **Amazon SNS Alerting** | N/A | `[Existing]` | [`backend/cloud/sns.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/sns.py) | Dispatches alerts with all 8 required incident attributes |
| **Cloud Orchestrator** | N/A | `[Existing]` | [`backend/cloud/orchestrator.py`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/backend/cloud/orchestrator.py) | End-to-end pipeline coordination |
| **React SRE Dashboard** | N/A | `[Existing]` | [`frontend/src/App.jsx`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/frontend/src/App.jsx) | 7 core sections + interactive chaos playground |
| **PyTorch RGCN Weights** | N/A | `[Planned]` | `ai-models/` | Varad ownership; planned for Phase-II training |
| **Safe RL Circuit Breaker** | N/A | `[Planned]` | `ai-models/` | Varad ownership; planned for Phase-II training |
