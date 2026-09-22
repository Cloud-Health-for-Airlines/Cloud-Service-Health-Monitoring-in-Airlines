# BACCP Dashboard UI Redesign — Phase 0 Design Plan

**Project:** Airline IT Boundary-Aware Cascade Predictor (BACCP)  
**Target:** Visual and Information-Architecture Overhaul of `frontend/`  
**Audience:** SREs and Flight Operations Center Engineers (Fast-Triage Operations Console)  
**Status:** Approved for Implementation (Phase 0 Complete)

---

## 1. Executive Summary & Root Problem Analysis

The current dashboard is functionally robust and accurately integrated with backend telemetry, but suffers from severe visual noise and "SaaS-card fatigue":
1. **Card Uniformity:** Every element is wrapped in an identical `.sre-card` (1px border, 8px radius, uniform padding), flattening the hierarchy between a critical cascading alert and a static AWS daemon address.
2. **Shouty ALL-CAPS:** Section titles, metric names, and column headers are uniformly capitalized, creating visual fatigue and reducing readability.
3. **Triple Status Redundancy:** Overall system health and circuit breaker throttle states are repeated in 3 separate places (Header bar, KPI pill row, Circuit Breaker panel) with conflicting phrasings (`GATEWAY: THROTTLED (42%)`, `Overall System Health: HEALTHY`, `STATE: THROTTLED (42%)`).
4. **Color Ambiguity:** Semantic status colors (green/amber/red) are mixed with decorative badge colors (purple gateway badge, amber legacy badge, blue cloud badge), causing operators to misread architectural taxonomy as degraded health.
5. **Nested Box Syndrome:** Diagnostic sections (like AWS Observability) nest 4 bordered boxes inside a parent bordered box.
6. **Raw Math in Main View:** Formal LaTeX/ASCII state divergence formulas ($\epsilon(t) = ||\Phi(t) - \Psi(t)|| / ||\Phi(t)|| \times 100$) occupy permanent primary screen real estate instead of on-demand documentation.

---

## 2. Token System & Palette Definition

### 2.1 Base UI Tokens (Dark Obsidian Flight-Deck)
```css
:root {
  /* Surfaces & Canvas */
  --bg-canvas: #090d16;          /* Deep cockpit night */
  --bg-surface: #0f172a;         /* Primary panel elevation */
  --bg-surface-subtle: #131d31;  /* Secondary / nested groupings */
  --bg-surface-elevated: #1e293b;/* Hover states & dropdowns */
  
  /* Borders & Dividers (Subtle, non-competing) */
  --border-quiet: rgba(148, 163, 184, 0.08); /* 1px subtle separation */
  --border-card: rgba(148, 163, 184, 0.16);  /* Standard panel border */
  --border-focus: #38bdf8;                    /* Keyboard focus ring */
  
  /* Typography */
  --text-primary: #f8fafc;       /* High-contrast readouts */
  --text-secondary: #94a3b8;     /* Field labels, units, descriptions */
  --text-tertiary: #64748b;      /* Micro metadata, timestamps */
  
  /* Brand / Interactive Accent */
  --accent-brand: #38bdf8;       /* Aviation electric cyan (links, tabs, focus) */
  --accent-brand-bg: rgba(56, 189, 248, 0.12);
}
```

### 2.2 Strict Semantic Status Palette
Used **strictly** for operational health states (never for decoration or architectural categorization):
```css
:root {
  /* Healthy / Nominal (Green) */
  --status-ok: #10b981;
  --status-ok-bg: rgba(16, 185, 129, 0.10);
  --status-ok-border: rgba(16, 185, 129, 0.28);
  
  /* Degraded / Warning / Throttled (Amber) */
  --status-warn: #f59e0b;
  --status-warn-bg: rgba(245, 158, 11, 0.10);
  --status-warn-border: rgba(245, 158, 11, 0.28);
  
  /* Critical / Blown Breaker / Cascade Imminent (Coral Red) */
  --status-crit: #ef4444;
  --status-crit-bg: rgba(239, 68, 68, 0.12);
  --status-crit-border: rgba(239, 68, 68, 0.32);
}
```

### 2.3 Quiet Architectural Taxonomy Chips
Distinct from health status — muted, earthy/neutral tones so an operator never mistakes a legacy mainframe node for an unhealthy service:
```css
:root {
  /* Legacy Mainframe (Warm Muted Slate / Sepia) */
  --tax-legacy-text: #d4c5b9;
  --tax-legacy-bg: #221c17;
  --tax-legacy-border: #3d3228;

  /* Boundary Gateway (Neutral Slate-Indigo) */
  --tax-gateway-text: #cbd5e1;
  --tax-gateway-bg: #1e1e2e;
  --tax-gateway-border: #33334d;

  /* Cloud-Native Microservice (Cool Slate-Blue) */
  --tax-cloud-text: #94a3b8;
  --tax-cloud-bg: #141c28;
  --tax-cloud-border: #233144;
}
```

---

## 3. Typography & Type Scale

- **UI Sans:** `Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
- **Technical / Metrics Monospace:** `'JetBrains Mono', 'SF Mono', Consolas, monospace`  
  *(Monospace is strictly reserved for latencies, IP/ports, timestamps, IDs, and numeric readouts — never for section labels).*

### Type Scale Hierarchy
| Level | Font Size | Line Height | Weight | Application |
|---|---|---|---|---|
| `display-1` | 36px (2.25rem) | 1.1 | 700 (Mono) | Hero Metric Readout (Drift %, Cascade Prob) |
| `display-2` | 24px (1.5rem) | 1.2 | 700 (Mono) | Top-line KPI Values, Lead-time countdown |
| `heading-1` | 18px (1.125rem) | 1.3 | 600 (Sans) | Hero Stage Titles (`Generation-typed dependency graph`) |
| `heading-2` | 15px (0.9375rem) | 1.4 | 600 (Sans) | Secondary Section Titles, Alert Titles |
| `body-base` | 13px (0.8125rem) | 1.5 | 400 (Sans) | Table cells, explanatory text, drawer details |
| `caption` | 11px (0.6875rem) | 1.4 | 500 (Sans) | Field labels, timestamps, metadata |
| `chip-micro` | 10.5px (0.65rem) | 1.2 | 600 (Sans) | Short status chips (`NOMINAL`, `THROTTLED`) |

### Casing Convention
- **Sentence Case:** Used for 100% of headings, section labels, buttons, and table column titles (e.g., *“Service domain”*, *“Observed latency”*, *“Current breaker state”*).
- **Uppercase Restriction:** Restricted exclusively to 1-2 word micro status badges (`HEALTHY`, `DEGRADED`, `CRITICAL`, `THROTTLED`).

---

## 4. Spacing Scale & Elevation Primitives

### Spacing Scale
- `space-1`: 4px | `space-2`: 8px | `space-3`: 12px | `space-4`: 16px | `space-5`: 24px | `space-6`: 32px | `space-7`: 48px

### Structural Primitives
1. **`.panel-hero`**: Dominant elevation, subtle 1px border with slight top glow, 20px padding. Reserved for the Dependency Graph & Boundary Drift Hero Stage.
2. **`.panel-subtle`**: Quiet surface, borderless or ultra-quiet border (`rgba(148,163,184,0.08)`), separated primarily by background contrast and 16px padding.
3. **`.metric-hero`**: Large display number (32px), small muted label above, micro context line below.
4. **`.metric-compact`**: In-line key-value pair for dense secondary telemetry.
5. **Button Hierarchy**:
   - `btn-primary`: Bright blue accent for standard positive triggers (Refresh, Apply throttle).
   - `btn-secondary`: Quiet slate outline for neutral filters, disclosures, resets.
   - `btn-destructive`: High-consequence amber/red with warning indicator (`Emergency isolate gateway`) — visually distinct from routine throttling.

---

## 5. Information Architecture & Restructured Wireframe

### ASCII Layout Wireframe

```
+========================================================================================================+
| [✈️ BACCP] Airline Cascade Monitor  |  Status: ● HEALTHY  |  Alerts: 0  |  Breaker: CLOSED (0%)  | [↻ Refresh] |
+========================================================================================================+
|                                                                                                        |
| TOP-LINE SYSTEM OVERVIEW (Differentiated, NOT 6 equal cards)                                           |
| +------------------------------------+-----------------------+-----------------------+---------------+ |
| | HERO 1: CASCADE PROBABILITY        | HERO 2: BOUNDARY DRIFT| PREDICTED ROOT CAUSE  | TELEMETRY     | |
| | 8.2% (Low Risk Envelope)           | 14.2%  (Norm: <45.0%) | boundary-gateway      | Live (1s poll)| |
| +------------------------------------+-----------------------+-----------------------+---------------+ |
|                                                                                                        |
| HERO STAGE: CORE MISSION-CRITICAL ARTIFACTS (Dominant visual weight)                                   |
| +---------------------------------------------------------+------------------------------------------+ |
| | 🌐 Generation-Typed Dependency Graph (60% width)         | 🛡️ Boundary Drift & Sync Gauge (40% width)| |
| | - Discovered via zero-overhead eBPF                     | - Radial Sync Gauge: 14.2%               | |
| | - Clear visual separation:                              | - Side-by-side comparative latency meter:| |
| |   [Cloud-Native (3)] ---> [Gateway] ===> [Legacy-Core]  |     Legacy Core: 68.2 ms (TCP:9090)      | |
| | - Live throughput & TCP socket telemetry on edges       |     Cloud Ingress: 18.6 ms (HTTP:8080)   | |
| | - Active fault highlights on critical boundary edge     | - [ℹ️ View drift formula & methodology]  | |
| +---------------------------------------------------------+------------------------------------------+ |
|                                                                                                        |
| PREDICTIVE INCIDENT & MITIGATION BANNER (High triage priority)                                         |
| +----------------------------------------------------------------------------------------------------+ |
| | [OK] Nominal Operating Envelope — No cascading failure predicted across hybrid boundary.             |
| | (When alert active: Full triage bar showing root node, lead-time countdown, and mitigation action)   |
| +----------------------------------------------------------------------------------------------------+ |
|                                                                                                        |
| CONSOLIDATED OPERATIONS & MITIGATION SUITE                                                             |
| +----------------------------------------------------------------------------------------------------+ |
| | Navigation Tabs:  [ Active Monitoring ]   [ Circuit Breaker Controls ]   [ Chaos Testbed & Infra ] | |
| +----------------------------------------------------------------------------------------------------+ |
| | TAB 1: SERVICE HEALTH MATRIX (Cross-generation telemetry table, sentence-case, health bars)         |
| | Service Domain      | Generation      | Status   | Latency | Req/s | Error % | Health Score          |
| | Legacy Core         | Legacy (Sepia)  | Nominal  | 68.2ms  | 57.5  | 0.0%    | [||||||||||] 86/100   |
| | Integration Gateway | Gateway (Slate) | Nominal  | 45.0ms  | 57.5  | 0.1%    | [||||||||||] 89/100   |
| | Reservations        | Cloud (Steel)   | Nominal  | 18.5ms  | 24.5  | 0.0%    | [||||||||||] 98/100   |
| | ...                                                                                                |
| +----------------------------------------------------------------------------------------------------+ |
| | TAB 2: CIRCUIT BREAKER & MITIGATION (Consolidated state, throttle slider, emergency actions, log)   |
| | Current State: CLOSED (0%) | Last Actuation: Nominal | Target: boundary-gateway                      |
| | [Slider 10-90%] [Apply Throttle] | [⚠️ Emergency Isolate Gateway] | [🔄 Reset Breaker]              |
| | Recent Actuation Audit Table                                                                         |
| +----------------------------------------------------------------------------------------------------+ |
| | TAB 3: CHAOS TESTBED & AWS INFRASTRUCTURE (Progressive disclosure for diagnostic/testing controls)   |
| | - Testbed Fault Injection Buttons (Delay 1500ms, Connection Drop, Batch Stall, Clear)               |
| | - Flattened AWS Observability Strip (CloudWatch, X-Ray, SageMaker, Lambda) - No nested boxes!       |
| +----------------------------------------------------------------------------------------------------+ |
+========================================================================================================+
```

---

## 6. Self-Critique Against Anti-Patterns

1. **Did we swap 6 equal boxes for 6 equal boxes in another color?**  
   *No.* System Overview is restructured into 2 primary Hero KPI metrics (Cascade Prob & Boundary Drift) + a dense, quieter secondary telemetry strip.
2. **Does status appear 3 times?**  
   *No.* The top flight-deck header provides the single authoritative system status. Lower panels show supporting granular metrics (latency delta, actuation log) without restating the high-level banner.
3. **Are nested boxes eliminated?**  
   *Yes.* The AWS Observability component is flattened into a single, clean status strip without border-in-border cards.
4. **Is the mathematical formula in the way of triage?**  
   *No.* The raw divergence equation $\epsilon(t)$ is moved behind a collapsible disclosure button (`View drift formula`), leaving the radial gauge and plain-English explanation front and center.
5. **Are testing controls cluttering live operations?**  
   *No.* Chaos injection controls are organized in a dedicated tab or collapsible drawer, keeping the operational flight deck clean.

---

## 7. Next Implementation Steps
- **Phase 1:** CSS tokens in `frontend/src/index.css` (primitives for `.panel-hero`, `.panel-subtle`, `.metric-hero`, semantic badges, quiet taxonomy chips).
- **Phase 2:** Top bar & `SystemOverview.jsx` (Authoritative status, hierarchical KPIs, sentence case).
- **Phase 3:** `DependencyGraph.jsx` & `BoundaryHealth.jsx` (Hero stage canvas, comparative latency meter, formula disclosure).
- **Phase 4:** `PredictiveAlerts.jsx` & `IncidentDetail.jsx` (Fast triage hierarchy, lead-time priority).
- **Phase 5:** Consolidate `CircuitBreakerStatus.jsx`, `CircuitBreakerControl.jsx`, and secondary tab for `ChaosControls.jsx`.
- **Phase 6:** `ServiceHealthTable.jsx` (Clean borders, sentence case, semantic bars).
- **Phase 7:** `AwsObservabilityPanel` (Flattened grid, no nested boxes).
- **Phase 8:** Responsive check, accessibility focus rings, WCAG contrast verification, and `npm run build`.
