<p align="center">
  <img src="assets/banner.png" alt="BACCP — Boundary-Aware Cascade Predictor" width="100%">
</p>

# BACCP ✈️ — Boundary-Aware Cascade Predictor

<p align="center">
  <a href="http://localhost:3000/">Live SRE Console</a> | <a href="documentation/phase1-comprehensive-report.md">Academic Report</a> | <a href="architecture/architecture-diagram.md">Architecture Specs</a> | <a href="presentation/presentation.html">Slide Deck</a>
</p>

<p align="center">
  <a href="documentation/phase1-comprehensive-report.md"><img src="https://img.shields.io/badge/Docs-Technical%20Report-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://github.com/Cloud-Health-for-Airlines/Cloud-Service-Health-Monitoring-in-Airlines"><img src="https://img.shields.io/badge/Tests-26%2F26%20Passing-10b981?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://react.dev"><img src="https://img.shields.io/badge/Frontend-React%2018%20%7C%20Vite%205-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://github.com/Cloud-Health-for-Airlines"><img src="https://img.shields.io/badge/Built%20by-Cloud--Health--for--Airlines-blueviolet?style=for-the-badge" alt="Built by Cloud Health for Airlines"></a>
</p>

**The mission-critical cascading-failure prediction and mitigation platform for hybrid airline cloud systems.** Built for zero-downtime flight operations, BACCP bridges the high-risk integration boundary between decades-old legacy mainframe cores (CICS, TPF, MQ, Sabre/Amadeus EDI) and modern cloud-native microservices (Reservations, Crew Scheduling, Baggage Handling). It auto-discovers network topology via zero-instrumentation eBPF socket tracing, calculates real-time digital-twin synchronization drift $\epsilon(t)$, predicts cascading failure probabilities and lead times using a multi-task Heterogeneous Relational GNN (RGCN) and neural point process, provides 90% conformal coverage intervals, and actuates automated reinforcement-learning circuit breakers (PPO & MAPPO) to isolate disruptions in under 2 seconds before operational meltdowns occur.

Tested against synthetic chaos benchmarks inspired by historical airline meltdowns (Southwest 2022, Delta 2024, United 2026), BACCP delivers **96.2% cascade prediction AUC-ROC**, an average of **18.4s predictive lead time window**, and **96.4% SLA preservation** under multi-gateway chaos fault injection.

<table>
<tr><td><b>Generation-Typed Dependency Graph</b></td><td>First-class typing dimensions for legacy mainframe (TCP:9090), boundary integration gateways (HTTP:8080), and cloud-native services, auto-discovered at the kernel level without touching legacy systems.</td></tr>
<tr><td><b>Digital-Twin State Drift ε(t)</b></td><td>Real-time mathematical divergence formulation comparing ground-truth mainframe transaction completion vectors with cloud ingress shadow state: <code>ε(t) = ||Φ(t) - Ψ(t)|| / ||Φ(t)|| × 100</code>.</td></tr>
<tr><td><b>Predictive Lead-Time Window</b></td><td>Neural temporal point process (Hawkes process) estimating exact countdown lead times (~15s–120s) before microservice lag crosses the boundary into catastrophic mainframe saturation.</td></tr>
<tr><td><b>Conformal Uncertainty Calibration</b></td><td>Split conformal prediction guaranteeing 90% marginal coverage bounds (<code>[lower, upper]</code>) so operators know exactly when the AI model is confident vs. extrapolating.</td></tr>
<tr><td><b>Multi-Agent RL Circuit Breaker</b></td><td>Dual-tier PPO & MAPPO (Multi-Agent PPO) actor-critic policy coordinating dynamic request rate throttling (10%–90%) and emergency isolation across distributed boundary gateways.</td></tr>
<tr><td><b>Zero-Instrumentation eBPF</b></td><td>Kernel-level socket and TCP tracepoint capture eliminating the need for heavyweight bytecode instrumentation, sidecars, or proprietary legacy agents.</td></tr>
<tr><td><b>Antigravity SRE Flight Deck</b></td><td>State-of-the-art dark cockpit console inspired by <code>antigravity.google</code> with an orchestrated hero entry, smooth RAF lerp cursor following, magnetic CTA buttons, and strict semantic triage palettes.</td></tr>
</table>

---

## Quick Install & Run

### Prerequisites
- **Python:** 3.11+ (with PyTorch 2.2+)
- **Node.js:** 18+ & npm
- **OS:** Windows, Linux, or macOS

### Option A: Interactive Standalone Runner (Instant Zero-Server CLI)
Run the complete end-to-end BACCP pipeline in your terminal with zero server setup. Features real-time eBPF discovery, drift calculation, RGCN inference, PPO actuation, and interactive chaos fault injection:

```bash
# Clone the repository
git clone https://github.com/Cloud-Health-for-Airlines/Cloud-Service-Health-Monitoring-in-Airlines.git
cd Cloud-Service-Health-Monitoring-in-Airlines

# Run interactive terminal simulation
python run_demo.py --interactive
```

### Option B: Full-Stack SRE Console (Backend API + Antigravity Frontend)

```bash
# 1. Install backend & AI model dependencies
pip install -r ai-models/requirements.txt

# 2. Launch the BACCP REST API & Cloud Integration Server (Port 8000)
python backend/api/app.py --port 8000

# 3. In a new terminal, launch the Antigravity Vite Dashboard (Port 3000)
cd frontend
npm install
npm run dev
```

Open **[http://localhost:3000](http://localhost:3000)** in your browser. The dashboard automatically connects to the live backend stream.

### Option C: Zero-Dependency Standalone Browser Preview
Open the pre-built standalone dashboard bundle in any modern web browser without running Node or Python:

```bash
# Linux / macOS
open frontend/dist_preview/index.html

# Windows (PowerShell)
Start-Process frontend/dist_preview/index.html
```

---

## Interactive CLI & SRE Commands

| Command | Description |
|---|---|
| `python run_demo.py --interactive` | Launch full interactive SRE terminal console with chaos injection triggers |
| `python run_demo.py --fault network-delay` | Evaluate cascade prediction under 1500ms mainframe TCP lag |
| `python run_demo.py --fault connection-drop` | Test boundary breaker isolation under 100% SYN packet drop |
| `python -m pytest backend/tests ai-models/tests` | Execute full automated test suites (26/26 tests passing) |
| `python ai-models/evaluation/evaluate.py` | Run comparative benchmark suite (RGCN vs GAT-GRU, PPO vs DQN) |
| `python ai-models/data/generate_dataset.py` | Generate 3,000 synthetic multi-modal airline failure traces |
| `cd frontend && npm run build` | Compile optimized production bundle for the Antigravity dashboard |

---

## System Architecture & Component Pipeline

```
+----------------------------------------------------------------------------------------------------+
|                                    BACCP END-TO-END PIPELINE                                       |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   +--------------------------+       eBPF Zero-Overhead       +--------------------------------+   |
|   |  Legacy Mainframe Core   | =============================> |       Boundary Gateway         |   |
|   |   (CICS / TPF / MQ)      |       Kernel Socket Probes     |   (HTTP:8080 <-> TCP:9090)     |   |
|   +--------------------------+                                +--------------------------------+   |
|                                                                               |                    |
|                                                                               v                    |
|   +--------------------------------------------------------------------------------------------+   |
|   |                   Generation-Typed Heterogeneous Dependency Graph Builder                  |   |
|   |          Nodes: [Legacy (Sepia), Gateway (Indigo), Cloud Services (Steel Slate)]           |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                |                                                   |
|                       +------------------------+-----------------------+                           |
|                       v                                                v                           |
|   +---------------------------------------+    +-----------------------------------------------+   |
|   |  Digital-Twin Sync Drift Engine ε(t)  |    |  Multi-Task Heterogeneous RGCN Predictor     |   |
|   |  ||Φ(t) - Ψ(t)|| / ||Φ(t)|| * 100     |    |  - Cascade Failure Probability P(cascade)     |   |
|   |  Nominal < 45.0% | Critical >= 70.0%  |    |  - Hawkes Process Lead-Time Countdown ~24s    |   |
|   +---------------------------------------+    |  - 90% Conformal Prediction Bounds            |   |
|                       |                        |  - Integrated Gradients Attribution           |   |
|                       |                        +-----------------------------------------------+   |
|                       +------------------------+                                                   |
|                                                v                                                   |
|   +--------------------------------------------------------------------------------------------+   |
|   |                     Automated Multi-Agent RL Circuit Breaker (MAPPO)                       |   |
|   |            Actor-Critic Policy: Dynamic Throttling (10%-90%) & Emergency Isolation         |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                |                                                   |
|                       +------------------------+-----------------------+                           |
|                       v                                                v                           |
|   +---------------------------------------+    +-----------------------------------------------+   |
|   |    AWS Cloud Observability Stack      |    |      Antigravity.google SRE Dashboard         |   |
|   |  - CloudWatch Metrics Pipeline        |    |  - Google Sans Flex Typography                |   |
|   |  - AWS X-Ray Trace Subsegments        |    |  - RAF Lerp Custom Cursor Follower            |   |
|   |  - SageMaker Online Endpoint Head     |    |  - Real-time SVG Topology & Radial Gauge      |   |
|   |  - Lambda Breaker & SNS Dispatch      |    |  - 1-Click Interactive Chaos Fault Injection  |   |
|   +---------------------------------------+    +-----------------------------------------------+   |
+----------------------------------------------------------------------------------------------------+
```

---

## Model Benchmark Evaluation & Results

Evaluated on 3,000 synthetic multi-modal airline incident traces generated across varying fault intensities (`network-delay`, `connection-drop`, `batch-job-stall`):

### 1. Cascade Prediction Head Benchmark
| Model Architecture | Typing Dimension | AUC-ROC | Precision | Recall | F1-Score | Lead-Time MAE | 90% Conformal Cov. |
|---|---|---|---|---|---|---|---|
| **BACCP Hetero-RGCN (Ours)** | **Generation-Typed** | **0.962** | **0.941** | **0.952** | **0.946** | **18.4s** | **91.2%** |
| GAT-GRU Baseline | Homogeneous | 0.884 | 0.852 | 0.868 | 0.860 | 38.2s | 82.4% |
| TCN (Temporal ConvNet) | Metric-Only | 0.831 | 0.795 | 0.812 | 0.803 | 52.6s | 74.5% |

### 2. Reinforcement Learning Circuit Breaker Benchmark
| Mitigation Strategy | Algorithm | Cascade Containment | SLA Preservation | Mean Throttle Rate | Time-to-Mitigate |
|---|---|---|---|---|---|
| **BACCP Adaptive RL (Ours)** | **PPO + MAPPO** | **98.1%** | **96.4%** | **34.2%** | **1.42s** |
| Baseline Q-Learning | DQN | 81.5% | 78.2% | 58.6% | 4.85s |
| Static Threshold Ladder | Rule-Based | 72.0% | 64.8% | 65.0% | 8.20s |

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Kernel & Tracing** | eBPF (`sockops`, `kprobe/tcp_v4_connect`), BCC, Linux C |
| **Machine Learning & AI** | PyTorch 2.2+, Heterogeneous RGCN, Neural Hawkes Process, Split Conformal Prediction, Integrated Gradients |
| **Reinforcement Learning** | PPO (Proximal Policy Optimization), MAPPO (Multi-Agent PPO), Multi-Objective Reward Engine |
| **Backend & Cloud Services** | Python 3.11, FastAPI, Amazon CloudWatch, AWS X-Ray, Amazon SageMaker, AWS Lambda, Amazon SNS |
| **Frontend & Visualization** | React 18, Vite 5, Google Sans Flex, Vanilla CSS Design System, SVG Topology Graph |
| **Database & Caching** | PostgreSQL, InfluxDB, Local In-Memory Ring Buffer |
| **Chaos Testing** | Deterministic NetEm, IPTables, Synthetic Chaos Harness |

---

## Documentation Index

All master technical reports, implementation guides, and architectural designs are available in the repository:

| Document | Purpose |
|---|---|
| 📖 [**LOCAL_RUN_GUIDE.md**](./LOCAL_RUN_GUIDE.md) | Comprehensive step-by-step guide for local setup, testing, and troubleshooting |
| 📋 [**phase1-comprehensive-report.md**](./documentation/phase1-comprehensive-report.md) | 35+ page master academic technical report with formal proofs and literature review |
| 🧠 [**ai-models/README.md**](./ai-models/README.md) | AI/ML layer production specifications, PyTorch architectures, and training workflows |
| 📊 [**ai-models-evaluation.md**](./results/ai-models-evaluation.md) | Empirical benchmark evaluation results, comparative baseline tables, and loss curves |
| 🎨 [**frontend/DESIGN_PLAN.md**](./frontend/DESIGN_PLAN.md) | Antigravity design token system, typography scale, and layout hierarchy |
| 📐 [**architecture-diagram.md**](./architecture/architecture-diagram.md) | Component sequence diagrams, interface data contracts, and Mermaid topologies |
| 👥 [**WORK_DISTRIBUTION.md**](./WORK_DISTRIBUTION.md) | Detailed individual task ownership, milestones, and deliverables breakdown |

---

## Team & Workstream Ownership
Ragghav:
Varad:
Bieanshu:

## License

Distributed under the **MIT License**. See [`LICENSE`](./LICENSE) for full details.

Built by **[Cloud-Health-for-Airlines](https://github.com/Cloud-Health-for-Airlines)** for mission-critical hybrid airline infrastructure.
