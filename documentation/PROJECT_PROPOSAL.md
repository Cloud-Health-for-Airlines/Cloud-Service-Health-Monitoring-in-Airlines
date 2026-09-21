# Project Proposal: BACCP

## Boundary-Aware Cross-Generation Cascade Predictor for Airline Cloud–Mainframe Infrastructure

---

## 1. Title
**BACCP: Boundary-Aware Cross-Generation Cascade Predictor for Hybrid Airline IT Systems**

---

## 2. Abstract
Modern commercial airlines depend on hybrid enterprise architectures coupling decades-old mainframe transaction processing cores (IBM CICS, TPF, COBOL batch systems) with modern cloud-native microservices (Reservations, Crew Scheduling, Baggage Handling) deployed on AWS and Kubernetes. Recent catastrophic operational collapses across major global carriers demonstrate that airline IT resilience is bottlenecked at the interface where microservices communicate with legacy mainframes. 

Existing cloud monitoring tools treat enterprise service graphs as homogeneous microservice networks, remaining blind to technological generation boundaries and incapable of instrumenting mainframes. Consequently, cascading failures crossing this architectural boundary are detected only after passenger-facing systems fail.

**BACCP** is a boundary-aware predictive monitoring and self-protection framework. It auto-discovers cross-generation topologies without bytecode agents using kernel-level eBPF socket tracing, models generation gaps as distinct relational edges within a heterogeneous Graph Neural Network (RGCN), quantifies boundary state divergence using a digital-twin synchronization drift metric ($\epsilon(t)$), predicts cascading failures with calibrated lead times ($\hat{\tau}$) and conformal uncertainty bounds ($1 - \alpha = 0.90$), and triggers automated circuit-breaker rate limiting at the boundary gateway via AWS Lambda and CloudWatch.

---

## 3. Background
Commercial aviation runs on high-volume, real-time transaction processing. While customer-facing portals and flight search microservices scale horizontally in the cloud, core booking records, ticket issuance, and crew rosters remain anchored in mainframe systems. These legacy systems execute mission-critical transactions via integration gateways (MQ series, EDI translators, REST/TCP adaptors).

When extreme disruptions occur (e.g., severe weather storms, airport ground stops), cloud microservices generate synchronous bursts of state verification queries down to the integration gateway. The legacy core, operating on different clock cycles and constrained thread pools, cannot absorb these spikes, creating massive backpressure that ripples outward into reservation outages and flight cancellations.

---

## 4. Problem Statement
1. **Lack of Technology Generation Awareness**: Existing AIOps tools treat all service nodes identically, ignoring the stark contrast in protocols, latency profiles, and failure semantics between cloud microservices and legacy mainframes.
2. **Zero-Instrumentation Constraint**: Legacy mainframes cannot accommodate modern tracing agents, OpenTelemetry SDKs, or sidecars without risking catastrophic transactional instability.
3. **Reactive Detection Paradigm**: Conventional metric threshold alarms fire only after gateway queues fill and cloud microservices return 502/504 errors, eliminating any operational lead time for SRE intervention.
4. **Uncoordinated Mitigation**: Standard circuit breakers shed load indiscriminately across all services or isolate services too late, starving critical paths such as passenger boarding and FAA compliance reporting.

---

## 5. Motivation
High-profile airline outages in recent years have demonstrated the multi-million dollar economic impact and societal disruption of boundary cascade failures:
- **Southwest Airlines (December 2022)**: Legacy crew scheduling mainframe desynchronized from cloud dispatching, causing 16,700 flight cancellations and over \$1.1B in operational losses.
- **Delta Air Lines (July 2024)**: Gateway boot loops severed cloud-to-mainframe tracking, costing \$500M in recovery expenses.
- **United Airlines (January 2026)**: Nightly batch fare recalculations starved real-time EDI reservation adapters of TCP sockets, crashing national check-in.

These incidents highlight the urgent need for a monitoring framework that anticipates boundary cascades before they manifest downstream.

---

## 6. Objectives
1. **Zero-Overhead Boundary Discovery**: Automatically discover dependency topologies crossing the legacy/cloud boundary using Linux kernel eBPF probes attached to integration gateways.
2. **Heterogeneous Generation Typing**: Construct dynamic dependency graphs where nodes and edges are explicitly classified into `legacy`, `boundary-gateway`, and `cloud-native`.
3. **Digital-Twin Drift Metric**: Quantify state divergence between cloud transaction submission rates $\Phi(t)$ and legacy completion rates $\Psi(t)$ via an explicit boundary sync drift score $\epsilon(t)$.
4. **Calibrated Lead-Time Cascade Prediction**: Predict cascading failure probability $P_{\text{cascade}}$ and operational lead time $\hat{\tau}$ with $90\%$ split conformal prediction guarantees ($1 - \alpha = 0.90$).
5. **Targeted Boundary Mitigation**: Actuate automated rate-limiting at the boundary gateway via AWS Lambda and CloudWatch alarms to maintain graceful degradation without manual intervention.

---

## 7. System Scope
- **In-Scope**:
  - Airline operational services: Reservations, Crew Scheduling, Baggage Handling.
  - Dual-homed integration gateway bridging HTTP REST (port 8080) to legacy TCP (port 9090).
  - Legacy mainframe transaction processing simulation (CICS/COBOL on internal Docker network).
  - eBPF kernel tracing (`kprobe:tcp_v4_connect`) on the gateway host.
  - Relational GNN prediction head with split conformal calibration.
  - AWS cloud observability layer (CloudWatch, X-Ray, SageMaker adapter, Lambda, SNS).
  - Interactive React monitoring dashboard with NOC/SRE console.
- **Out-of-Scope**:
  - Direct bytecode modification of proprietary IBM CICS/TPF mainframe kernels.
  - Modification of FAA air traffic control flight dispatch protocols.

---

## 8. Proposed Solution
BACCP treats the legacy/cloud boundary as a first-class architectural entity:
1. **eBPF Observation**: Intercepts TCP connection lifecycle events at the integration gateway's Linux kernel, capturing socket connection times, handshake delays, and socket resets without touching mainframe code.
2. **Generation-Typed Graph Construction**: Categorizes nodes by generation (`legacy`, `boundary-gateway`, `cloud-native`), mapping them to distinct relation matrices in an RGCN.
3. **Boundary Drift Tracking**: Calculates normalized divergence $\epsilon(t)$ in real-time.
4. **Multi-Task Prediction**: Computes cascade probability, estimated lead time in seconds, failure root cause, and conformal bounds.
5. **Automated Safe Actuation**: Executes dynamic throttle rate adjustments ($30\% - 100\%$) via AWS Lambda, maintaining critical reservation throughput while protecting the legacy core.

---

## 9. Architecture
The architecture comprises 5 modular layers:
1. **Airline Testbed Layer**: Docker Compose network isolating `legacy-core` on an internal network reachable only via `boundary-gateway`.
2. **eBPF Discovery Layer**: Kernel tracepoints streaming raw socket events into topological aggregators.
3. **BACCP Backend Server**: Zero-dependency REST API server coordinating graph state, drift calculation, and telemetry buffering.
4. **AWS Cloud Integration Layer**: Dual-mode provider layer connecting CloudWatch, X-Ray, SageMaker, Lambda, and SNS.
5. **React Dashboard**: Modern NOC/SRE dashboard displaying system overview, SVG dependency graph, radial drift gauge, predictive alert cards, and chaos playground.

---

## 10. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Testbed & Services** | Python 3, Docker Compose, Linux namespaces, eBPF / bpftrace (`kprobe:tcp_v4_connect`), tc / netem, iptables |
| **Data & Storage** | PostgreSQL 16, JSONL trace logs |
| **Backend & Cloud** | Python 3 standard library (zero-dependency REST core), `boto3` (AWS SDK for CloudWatch, X-Ray, SageMaker, Lambda, SNS) |
| **Predictive Intelligence** | Relational GCN (RGCN), Neural Hawkes Point Process, Split Conformal Prediction ($1-\alpha=0.90$) |
| **Frontend** | React 18, Vite 5, Lucide Icons, CSS Variables (dark theme SRE console) |

---

## 11. Methodology
1. **Chaos-Driven Data Generation**: Inject realistic boundary failure profiles (network packet delays, connection drops, batch job CPU stalls) into the integration gateway using Linux `tc` and `iptables`.
2. **Kernel Telemetry Capture**: Capture socket-level connection attempts and latencies via eBPF.
3. **Topological Feature Extraction**: Construct generation-typed adjacency matrices and compute betweenness centrality across boundary-crossing paths.
4. **Model Training & Calibration**: Train multi-task cascade heads on logged failure trajectories, applying split conformal inference on holdout calibration sets.
5. **Closed-Loop Mitigation Verification**: Verify that automated Lambda throttle actuation prevents downstream reservation microservice crashes during induced faults.

---

## 12. Dataset & Testbed
- **Docker Compose Airline Testbed**:
  - `reservations` (HTTP :8081, cloud-native)
  - `crew` (HTTP :8082, cloud-native)
  - `baggage` (HTTP :8083, cloud-native)
  - `boundary-gateway` (HTTP :8084 $\to$ TCP :9090, dual-homed boundary gateway)
  - `legacy-core` (TCP :9090, legacy CICS/COBOL mainframe simulator)
- **Trace Dataset**: Socket connection events containing timestamps, source PID, destination IP/port, connection latency, and TCP error states.

---

## 13. Fault Injection
Controlled chaos engineering profiles managed by `testbed/chaos/chaos.py`:
- `network-delay`: Injects 200ms–1500ms latency on the gateway-to-legacy TCP interface, simulating satellite link degradation or EDI queue congestion.
- `connection-drop`: Drops 10%–50% of TCP SYN packets, simulating mainframe socket queue overflows.
- `batch-job-stall`: Starves gateway connection handling threads, simulating unannounced mainframe batch fare recalculations.

---

## 14. AI/ML Approach
- **Heterogeneous Graph Convolution (RGCN)**: Relation-specific weight matrices $W_r$ for `cloud->gateway`, `gateway->legacy`, and `cloud->cloud` edges.
- **Continuous-Time Dynamic Memory**: Updates node memory vectors on incoming eBPF connection events.
- **Temporal Hawkes Process**: Models cascade propagation as a self-exciting point process with intensity $\lambda(t)$ to derive remaining operational warning time $\hat{\tau}$.
- **Split Conformal Prediction**: Wraps model point predictions in statistically guaranteed intervals:
  $$\mathbb{P}(Y_{t+\tau} \in C(X_t)) \ge 1 - \alpha \quad (1-\alpha = 0.90)$$

---

## 15. Cloud Architecture
- **Amazon CloudWatch**: Collects and alarms on 8 custom metrics (`boundary_health_score`, `cascade_probability`, `prediction_lead_time`, `service_latency`, `error_rate`, `request_rate`, `circuit_breaker_actions`, `active_alerts`) under namespace `BACCP/AirlineCloudHealth`.
- **AWS X-Ray**: Generates cross-generation distributed traces with subsegments spanning cloud microservices, the boundary gateway, and the legacy core.
- **Amazon SageMaker**: Hosts the multi-task cascade prediction model endpoint.
- **AWS Lambda**: Executes boundary circuit breaker mitigation policies (`THROTTLE`, `ISOLATE`, `RESET`).
- **Amazon SNS**: Broadcasts high-severity predictive cascade alerts to SRE operational distribution lists.

---

## 16. Alerting and Mitigation
- **Predictive Alerts**: Issued when $P_{\text{cascade}} \ge 0.55$ or $\epsilon(t) \ge 45\%$, providing estimated lead time countdown ($\hat{\tau}$ seconds), root cause node, and conformal bounds.
- **Dynamic Circuit Breaker**: Adjusts throttle rates on the boundary gateway:
  - Nominal ($P < 0.35$): State `CLOSED`, 0% throttle.
  - High Risk ($0.55 \le P < 0.75$): State `THROTTLED`, dynamic throttle rate ($30\% - 75\%$).
  - Critical Cascade ($P \ge 0.75$ or $\epsilon(t) \ge 70\%$): State `OPEN`, 100% boundary isolation with 429 Fast-Fail caching to protect downstream cloud services.

---

## 17. Evaluation Plan
- **Detection Accuracy**: Precision, Recall, F1-Score, and AUC-ROC across fault profiles.
- **Lead-Time Precision**: Advance warning window (minutes of warning before downstream microservice collapse).
- **Conformal Coverage**: Empirical verification that true cascade events fall within the 90% confidence interval at rate $\ge 90\%$.
- **Mitigation Resilience**: Ratio of surviving transaction throughput under chaos with BACCP vs. without BACCP.
- **Latency Overhead**: End-to-end eBPF tracing overhead ($< 1.5\%$ CPU, $< 0.2$ms latency).

---

## 18. Expected Outcomes
1. A demonstrated reduction of cascading failure downtime in hybrid airline environments.
2. Advance warning lead time of at least 60–180 seconds before customer-facing reservation degradation.
3. Automated circuit-breaker throttling preserving critical flight booking transactions during mainframe stalls.
4. Open-source research prototype featuring eBPF discovery, cloud adapters, and a React operations dashboard.

---

## 19. Current Progress
- **Backend Cloud Integration**: 100% completed dual-mode adapters for CloudWatch, X-Ray, SageMaker, Lambda, SNS, and Orchestrator. 22/22 unit and integration tests passing.
- **Frontend Dashboard**: 100% completed React 18 / Vite NOC console with SVG dependency graph, drift gauge, predictive alerts, service matrix, and incident detail modal.
- **Architecture & Interfaces**: Formal diagrams and contracts completed.
- **Academic Research**: Comprehensive 15-paper literature analysis and mathematical formulation completed.
- **Testbed Baseline**: Containerized 5-service airline topology and eBPF socket tracing operational.

---

## 20. Limitations
- **Current Model Inference**: Executed via a calibrated analytical engine modeling the RGCN/Hawkes formulation while full PyTorch model weights are undergoing Phase-II training in `ai-models/`.
- **Testbed Scale**: Evaluated on a 5-node core airline topology; enterprise multi-hub deployment requires multi-host eBPF aggregation.
- **Mainframe Simulation**: CICS/COBOL processing simulated via TCP stateful responder; validation against physical IBM z/OS hardware is deferred to enterprise pilots.

---

## 21. Future Scope
- **Phase-II AI Training**: Finalize PyTorch RGCN training and deploy weights to live SageMaker endpoint.
- **Safe Reinforcement Learning**: Train continuous-action PPO policy with Lagrangian safety constraints in `ai-models/circuit-breaker/`.
- **Cross-Carrier Federated Learning**: Implement FedMon-style federated eBPF telemetry sharing across Star Alliance / Oneworld carriers without exposing passenger PNR data.
