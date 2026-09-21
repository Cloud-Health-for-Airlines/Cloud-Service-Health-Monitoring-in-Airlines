# frontend/

Monitoring dashboard for BACCP (*Boundary-Aware Cross-Generation Cascade Predictor*).

## Features

- **Generation-Typed Dependency Topology**: Visualizes cloud-native microservices (`reservations`, `crew`, `baggage`), the integration boundary gateway (`boundary-gateway`), and the uninstrumented mainframe core (`legacy-core`), with dynamic edge particle flows and observation counts.
- **Boundary Health Score**: Digital-twin-derived state synchronization drift metric ($\epsilon(t)$) displayed via a circular radial gauge, warning threshold alarms (45%), and critical boundaries (70%).
- **Predictive Cascade Alerts**: Real-time multi-task inference engine showing estimated cascade lead time countdown (warning window before boundary crossing), cascade probability with **90% Conformal Prediction confidence bounds**, root cause attribution, and an AIOps natural-language incident brief.
- **RL Circuit Breaker Controls**: Real-time status display (`CLOSED`, `THROTTLED`, `OPEN`), dynamic rate-limiting slider, and interactive mitigation action audit log.
- **Chaos Engineering Playground**: Inject deterministic testbed faults (`network-delay` 1500ms, `connection-drop` 100% SYN, `batch-job-stall` 15s, or clear) directly from the UI to observe live cascade prediction and automated mitigation in action.
- **AWS Cloud Observability**: Status indicators for CloudWatch metrics stream, AWS X-Ray distributed trace segments, SageMaker model inference, Lambda breaker actions, and Amazon SNS alert dispatch.

## Component Structure

```
frontend/
├── dist_preview/
│   └── index.html               → Zero-dependency standalone dashboard (instant browser preview)
├── src/
│   ├── api/
│   │   └── client.js            → REST API client with automatic offline demo fallback
│   ├── components/
│   │   ├── DependencyGraph.jsx       → SVG generation-typed topology
│   │   ├── BoundaryHealthScore.jsx   → Radial gauge & drift metrics
│   │   ├── PredictiveAlerts.jsx      → Lead-time countdown & conformal bounds
│   │   ├── CircuitBreakerControl.jsx → Mitigation controls & action history
│   │   ├── TelemetryView.jsx         → Chaos fault injection playground
│   │   └── CloudStatus.jsx           → AWS services operational cards
│   ├── App.jsx                  → Main layout & real-time polling
│   └── main.jsx                 → React root mount
├── index.html                   → Vite entrypoint
├── package.json                 → Dependencies and scripts
└── vite.config.js               → Vite configuration
```

## Running the Dashboard

### Option A: Instant Standalone Preview (Zero Dependencies)
You can open the self-contained dashboard directly in any browser without running `npm install`:

```bash
open frontend/dist_preview/index.html
```

### Option B: Vite Development Server
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser. Connects automatically to the backend API running at `http://localhost:8000`.

