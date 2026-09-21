# Research Gap Analysis: Cross-Generation Cascade Prediction in Airline IT

**Boundary-Aware Cross-Generation Cascade Predictor (BACCP)**

---

## 1. Executive Context & Motivation

Commercial aviation systems represent the highest-stakes hybrid distributed architectures in global industry. Modern passenger booking engines, crew roster rescheduling, and baggage reconciliation microservices execute in cloud-native containerized clusters on AWS and Kubernetes. Yet, core transaction processing, seat inventory reservation records, and ticketing depend on 40-year-old mainframe architectures (IBM CICS, TPF, COBOL batch engines).

While both cloud monitoring and AI for IT Operations (AIOps) have matured significantly, existing literature suffers from critical conceptual blind spots when applied to cross-generation airline IT estates. 

This document synthesizes the **6 fundamental research gaps** identified across the literature survey (spanning 15 foundational papers across cloud resilience, eBPF tracing, Graph Neural Networks, temporal point processes, and reinforcement learning rate-limiting), demonstrating why existing solutions fail during airline boundary cascades.

---

## 2. Comparative Analysis Matrix of Prior Art

| Study & Venue | Primary Technique | Target Domain | Acknowledged Boundary Limitations |
| :--- | :--- | :--- | :--- |
| **Barua & Kaiser (2022)**<br>*IEEE Access* | Microservices Architecture Decomposition | Airline Reservations | Assumes a clean, greenfield rebuild; completely omits legacy mainframe integration and has zero observability layer. |
| **Krasnovsky & Zorkin (2021)**<br>*IEEE Trans. Serv. Comput.* | Automated Model Discovery & Trace Parsing | Hybrid Enterprise Integration | Reconstructs static architecture diagrams post-hoc from logs; lacks real-time streaming detection and dynamic graph typing. |
| **Tang et al. (I³, 2022)**<br>*ACM SIGKDD* | Heterogeneous Graph Cascade Prediction | Physical Infrastructure (Power/Roads) | Types nodes by physical domain (substation vs. intersection); never considers technology generation as a relation axis. |
| **Li (2026)**<br>*IEEE Trans. Knowl. Data Eng.* | Attention-GNN Cascade Predictor | Homogeneous Cloud Microservices | Treats all microservice edges uniformly; misses generation latency variance and batch-vs-realtime protocol mismatches. |
| **Papp et al. (ZeroTracer, 2023)**<br>*IEEE S&P* | eBPF Kernel-Level Socket Tracing | Linux Enterprise Servers | Captures TCP socket events without app modification, but focuses exclusively on security forensics, not cascade prediction. |
| **Lyu et al. (2021)**<br>*ACM SIGMETRICS* | Deep RL Adaptive Rate Limiting (DQN+A3C) | High-Volume Web Scale (500M req/day) | Operates per-service indiscriminately; lacks coordination across boundary gateways and ignores downstream mainframe backpressure. |
| **Rossi et al. (TGN, 2020)**<br>*ICML* | Continuous-Time Dynamic Graph Memory | Social & Citation Networks | High memory complexity; uncalibrated for extreme clock-speed disparities (sub-second cloud vs. multi-hour mainframe batch). |
| **Vovk et al. / Angelopoulos (2021)**<br>*Found. Trends Mach. Learn.* | Distribution-Free Conformal Prediction | Computer Vision & NLP | General calibration framework; has not been coupled with cross-generation IT dependency networks. |

---

## 3. The 6 Critical Research Gaps

### Gap 1: Treating Cloud Systems as Homogeneous Graphs
- **The State of Art**: Modern GNN-based root cause analysis (RCA) and failure prediction models (e.g., Li 2026, MicroRank, CauseInfer) model software topologies as homogeneous graphs $G = (V, E)$ where every vertex $v_i$ is a microservice container and every directed edge $e_{ij}$ represents a standard HTTP/gRPC synchronous call.
- **Why It Fails in Airlines**: A call between the `reservations` and `crew` microservices has a standard deviation latency of $\pm 4$ms over HTTP. An edge crossing from the integration gateway to the legacy mainframe over TCP:9090 exhibits completely different physical dynamics: queueing latencies fluctuating from $15$ms to $3,500$ms, batch lockouts, and asynchronous transaction completion. Treating these edges with shared message-passing weights forces the neural network to average out the very latency anomalies that signal boundary collapse.
- **BACCP Resolution**: BACCP formalizes the topology as a **generation-typed heterogeneous graph** $G = (V, E, \mathcal{T}, \mathcal{R})$ where $\mathcal{T} = \{\text{legacy}, \text{boundary-gateway}, \text{cloud-native}\}$ and relational message-passing matrices $W_r$ uniquely weight boundary-crossing transactions.

---

### Gap 2: The Zero-Instrumentation Observability Paradox
- **The State of Art**: Enterprise observability platforms (DataDog, Dynatrace, OpenTelemetry) require injecting language-specific bytecode agents (Java agents, Go binary wrappers) or sidecar containers into every workload to inject trace context headers (`traceparent`).
- **Why It Fails in Airlines**: 1980s IBM CICS/COBOL mainframes running on z/OS cannot run Linux Docker sidecars or carry W3C distributed trace headers. Instrumenting legacy transaction code introduces existential business risks (a single corrupted register halts flight operations). Consequently, enterprise architects leave the legacy core uninstrumented, creating a total observability black hole at the exact point where failures originate.
- **BACCP Resolution**: BACCP leverages **Linux kernel eBPF probes** (`kprobe:tcp_v4_connect`) attached non-intrusively to the integration gateway. By monitoring socket connection times, TCP window sizes, and SYN retries at the OS kernel boundary, BACCP measures legacy degradation with zero application bytecode modification.

---

### Gap 3: Inability to Predict Cross-Boundary Cascades
- **The State of Art**: Existing failure prediction approaches frame the problem as binary anomaly classification: outputting a static score $\hat{y} \in [0, 1]$ indicating whether a failure is occurring right now.
- **Why It Fails in Airlines**: An alert that fires 3 seconds before the reservation system crashes is operationally useless. Operators require actionable **operational lead time** ($\hat{\tau} \ge 60$s) to enact mitigations. Furthermore, failure propagation across generation boundaries is non-linear: a minor 10% queue buildup on a mainframe MQ listener does not degrade linearly—it remains invisible until a threshold is reached, after which thread pools exhaust instantly and trigger a violent backward TCP reset storm.
- **BACCP Resolution**: BACCP integrates **Neural Hawkes Point Processes** into the graph representation, modeling boundary events as self-exciting point processes with conditional intensity $\lambda(t)$ to calculate the exact expected lead time $\hat{\tau}$ until downstream service failure.

---

### Gap 4: Absence of Boundary-Specific Health Features
- **The State of Art**: Telemetry ingestion engines monitor standard per-service RED metrics (Rate, Errors, Duration) and resource metrics (CPU, Memory, Disk I/O).
- **Why It Fails in Airlines**: CPU usage on a gateway server is virtually meaningless during an airline cascade; CPU remains low while gateway worker threads are blocked waiting on mainframe TCP sockets. Standard metric thresholds remain green while disaster is underway.
- **BACCP Resolution**: BACCP introduces a domain-specific mathematical metric: **Digital-Twin State Synchronization Drift** ($\epsilon(t)$):
  $$\epsilon(t) = \frac{\|\Phi(t) - \Psi(t)\|_2}{\|\Phi(t)\|_2 + \delta} \times 100$$
  where $\Phi(t)$ is the cloud gateway transaction submission rate and $\Psi(t)$ is the legacy mainframe transaction completion rate. When $\epsilon(t) > 45\%$, state divergence is flagged immediately, long before CPU or memory alarms trigger.

---

### Gap 5: Neural Overconfidence & Lack of Statistical Calibration
- **The State of Art**: Deep learning failure detectors output raw uncalibrated sigmoid probabilities.
- **Why It Fails in Airlines**: Raw neural network probabilities suffer from severe overconfidence under distribution shift (e.g., during extreme holiday passenger surges or unprecedented blizzard weather). If an automated circuit breaker acts on uncalibrated false alarms, it unnecessarily isolates the reservation gateway, stranding thousands of passengers and causing self-inflicted operational paralysis.
- **BACCP Resolution**: BACCP wraps all cascade probability predictions in **Split Conformal Prediction Intervals** ($1 - \alpha = 0.90$). The system acts only when the lower conformal bound guarantees high risk, mathematically bounding the false-alarm rate below $\alpha = 0.10$.

---

### Gap 6: Disconnect Between Failure Prediction and Automated Mitigation
- **The State of Art**: The research community has traditionally bifurcated into two disconnected camps: the AIOps community builds predictive models that output alerts to human dashboards, while the systems resilience community builds reactive circuit breakers (e.g., Netflix Hystrix, Envoy rate-limiters) that actuate only after errors exceed 50%.
- **Why It Fails in Airlines**: Human SREs cannot parse complex multi-service graph alerts and make manual throttling decisions in the 30–60 second window of an airline cascade. Conversely, reactive circuit breakers trip too late—after thousands of flight booking sessions have already been corrupted in-flight.
- **BACCP Resolution**: BACCP couples predictive cascade inference directly with an **automated boundary-scoped Reinforcement Learning circuit breaker** implemented via AWS Lambda. When high-confidence cascade risk is detected, the breaker dynamically throttles non-critical microservice requests at the gateway ($30\% - 75\%$) while preserving high-priority reservation checkouts.

---

## 4. Summary of BACCP Contributions

```
Existing Literature Gaps                    BACCP Solution
─────────────────────────────────────────────────────────────────────────────
Homogeneous Graph Assumptions       ───►    Generation-Typed Heterogeneous RGCN
Legacy Tracing Impossibility        ───►    Zero-Instrumentation Gateway eBPF
Static Binary RCA                   ───►    Hawkes Temporal Lead-Time Prediction
Generic RED System Metrics          ───►    Digital-Twin Sync Drift Score ε(t)
Uncalibrated Deep Learning          ───►    Split Conformal Bounds (1 - α = 0.90)
Alert-Only / Reactive Mitigation    ───►    Boundary-Scoped RL Circuit Breaker
```
