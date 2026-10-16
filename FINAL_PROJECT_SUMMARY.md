# BACCP: Boundary-Aware Cross-Generation Cascade Predictor
## Comprehensive Cross-Task Final Project Summary & Integration Report

**Project:** Cloud Airlines Service Health — AI/ML System Integration  
**Component:** Boundary-Aware Cross-Generation Cascade Predictor (BACCP)  
**Version:** `baccp-production-v1.5.0`  
**Date:** October 15, 2026  
**Status:** COMPLETE & VERIFIED — Ready for Formal Engineering Review  

---

> [!IMPORTANT]
> **CRITICAL DATA PROVENANCE & PERFORMANCE DISCLAIMER:**  
> All model training, evaluation metrics, conformal calibrations, and offline stress matrices in this project were conducted on **physics-informed synthetic incident trajectories** generated via the Cloud Airlines digital-twin environment. While the boundary gateway eBPF probe captures actual TCP connection attempt counts in live containers, live application latency percentiles, queue backlog depths, and multi-step time-series histories are **not instrumented** in the current live deployment.  
> **Synthetic evaluation does not establish production performance.** In live production, uncalibrated network jitter, OS scheduler delays, unmodeled legacy mainframe thread starvation, and non-stationary traffic will alter error distributions.

---

## 1. Executive Summary & Architecture Overview

The **Boundary-Aware Cross-Generation Cascade Predictor (BACCP)** is an enterprise reliability and ML inference system designed to detect, forecast, and explain cascading degradation across heterogeneous microservices (`reservations`, `crew`, `baggage`), boundary gateways (`boundary-gateway`), and mainframe infrastructure (`legacy-core`).

BACCP integrates three synergistic ML components into the Python/REST backend:
1. **Multi-Horizon Workload Forecasting**: Provides point predictions and 90% finite-sample conformal prediction intervals across 1-step (20s), 5-step (100s), and 15-step (300s) horizons.
2. **Degradation & Failure Risk Prediction**: Produces isotonic-calibrated degradation probabilities, advisory alert recommendations, continuous latency/backlog trajectory forecasts, and conditional time-to-degradation estimations.
3. **Transparent Attributions & Root-Cause Localization**: Computes exact TreeSHAP attributions on margin log-odds, separates measured from inferred features, and executes graph-based bottleneck localization with strict abstention under ambiguous conditions.
4. **Lightweight Backend & Prometheus Observability**: Connects to telemetry via a dual-mode adapter (`replay` default and `live`), utilizes single-process model loading, bounded LRU caching (size 256), a 2-worker `ThreadPoolExecutor` for asynchronous SHAP calculations, and exposes standard Prometheus metrics with bounded labels and alert-episode deduplication.

```
+----------------------------------------------------------------------------------------------------+
|                                    BACCP INFERENCE ARCHITECTURE                                    |
|                                                                                                    |
|  +------------------------+        +------------------------------------------------------------+  |
|  |   Telemetry Adapter    |        |                 ML Inference Service                       |  |
|  | - replay mode (default)| -----> | - Single-process bundle loader (Workload, Deg, Explain)    |  |
|  | - live mode (503 gaps) |        | - Bounded LRU Cache (256 items, TTL / snapshot invalidated)|  |
|  | - 60s staleness limit  |        | - ThreadPoolExecutor(max_workers=2) for CPU SHAP           |  |
|  +------------------------+        +------------------------------------------------------------+  |
|               |                                                  |                                 |
|               v                                                  v                                 |
|  +------------------------+        +------------------------------------------------------------+  |
|  |   Prometheus Metrics   |        |                     REST API Endpoints                     |  |
|  | - Degradation prob     |        | - GET /forecast/workload?service=...&horizon=...           |  |
|  | - Alert episodes (dedup) <-----+ - GET /predict/degradation?service=...&horizon=...          |  |
|  | - Matched forecast err |        | - GET /explain/{service}?horizon=...                       |  |
|  | - Telemetry freshness  |        | - GET /metrics (Standard Prometheus exposition)            |  |
|  +------------------------+        +------------------------------------------------------------+  |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Dataset Provenance, Splits & Generation

The system was evaluated using 275 physics-informed simulated incidents generated with micro-cycle flight booking traffic, eBPF boundary gateway transit, and heterogeneous cloud/mainframe services.

### 2.1. Partition Split Counts
The dataset (`ai-models/data/degradation_prediction/degradation_dataset.json`, SHA-256: `b19ef7024036...`) is strictly partitioned into non-overlapping incident cohorts:

| Partition Split Name | Incident Count | Step Count | Primary Purpose | Incident Nature & Perturbations |
| :--- | :---: | :---: | :--- | :--- |
| **`train`** | 120 | 18,000 | Model training & GBDT fitting | Stationary baseline + nominal fault injections |
| **`val`** | 30 | 4,500 | Hyperparameter tuning & early stopping | Validation across nominal service degradation |
| **`cal`** | 25 | 3,750 | Conformal quantile & isotonic calibration | Unbiased calibration set for probabilities & intervals |
| **`standard_test`** | 25 | 3,750 | Primary held-out evaluation | Stationary operations, nominal fault distribution |
| **`capacity_loss_test`** | 25 | 3,750 | Out-of-distribution robustness test | Severe physical thread/core capacity drops (40-70%) |
| **`boundary_slowdown_test`** | 25 | 3,750 | Out-of-distribution robustness test | Elevated boundary gateway latency & clock sync drift |
| **`surge_test`** | 25 | 3,750 | Out-of-distribution robustness test | Abrupt 2.5x traffic surges exceeding cluster capacity |
| **Total** | **275** | **41,250** | **Full Incident Corpus** | **All 5 cloud and mainframe services** |

---

## 3. Subsystem 1: Workload Forecasting (Task 1)

### 3.1. Methodology & Models
- Evaluated models: **Persistence Baseline**, **Seasonal Naive** ($P=15$ steps / 300s micro-cycle), **Autoregressive Ridge**, **Gradient Boosted Decision Trees (GBDT)**, and **Auto-Fallback Policy**.
- Forecast Horizons: $H=1$ (20s), $H=5$ (100s), $H=15$ (300s).
- Conformal Uncertainty: Exact finite-sample conformal prediction intervals with 90% nominal coverage guarantees under exchangeability.
- Capacity Headroom: Explicitly computes `capacity_margin_req_s` and `available_headroom_req_s` based on service-specific saturation limits.

### 3.2. Measured Evaluation Results (Standard Test Split)

| Service | Horizon | Model | MAE (req/s) | RMSE | WAPE (%) | Rel. Improvement vs Persistence | 90% Empirical Coverage | Mean Interval Width |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `reservations` | H=1 (20s) | Persistence | 3.183 | 4.475 | 5.42% | Baseline | 88.39% | 12.298 |
| `reservations` | H=1 (20s) | **GBDT (Selected)** | **2.321** | **3.245** | **3.95%** | **+27.1%** | **91.41%** | **9.981** |
| `reservations` | H=5 (100s) | Persistence | 6.550 | 11.565 | 11.09% | Baseline | 86.95% | 27.112 |
| `reservations` | H=5 (100s) | **GBDT (Selected)** | **4.121** | **7.718** | **6.98%** | **+37.1%** | **92.79%** | **18.127** |
| `reservations` | H=15 (300s) | Persistence | 6.654 | 12.307 | 11.14% | Baseline | 92.52% | 30.622 |
| `reservations` | H=15 (300s) | **GBDT (Selected)** | **4.516** | **8.249** | **7.56%** | **+32.1%** | **93.64%** | **23.516** |
| `crew` | H=5 (100s) | Persistence | 3.760 | 6.423 | 11.46% | Baseline | 87.08% | 15.522 |
| `crew` | H=5 (100s) | **GBDT (Selected)** | **2.350** | **4.232** | **7.16%** | **+37.5%** | **94.69%** | **11.043** |
| `baggage` | H=5 (100s) | Persistence | 5.181 | 9.161 | 11.29% | Baseline | 86.49% | 20.610 |
| `baggage` | H=5 (100s) | **GBDT (Selected)** | **3.210** | **6.304** | **6.99%** | **+38.0%** | **92.00%** | **13.287** |
| `boundary-gateway`| H=5 (100s) | **GBDT (Selected)** | **8.158** | **15.503**| **6.68%** | **+36.5%** | **93.90%** | **35.454** |
| `legacy-core` | H=5 (100s) | **GBDT (Selected)** | **8.006** | **15.093**| **6.70%** | **+36.8%** | **94.27%** | **34.821** |

### 3.3. Baseline Comparisons & Where Baselines Perform Better
- **Where GBDT Excels**: Across all services at medium and long horizons ($H=5, H=15$), GBDT consistently outperforms baselines by 25% to 38% in WAPE.
- **Where Simpler Baselines Perform Better / Trade-offs**:
  - **Persistence Baseline**: At $H=1$ (20s ahead), Persistence achieves MAE 3.18 req/s (vs GBDT 2.32 req/s) with zero computational overhead and zero risk of model artifact corruption or drift. For ultra-low-latency in-kernel packet filtering, persistence is a sensible fallback.
  - **Seasonal Naive**: Fails severely under aperiodic traffic shifts (WAPE 11.68% at $H=1$, a -115% degradation vs persistence) due to rigid phase alignment assumptions.

---

## 4. Subsystem 2: Degradation & Failure Prediction (Tasks 2 & 4A)

### 4.1. Model Architecture & Evaluated Cohorts
- **Static SLO Threshold Alerting**: Flags degradation if current latency $> 100$ms or backlog $> 50$ requests.
- **Training-Fitted Z-Score Anomaly**: Unsupervised Gaussian anomaly detection on historical feature distributions.
- **Model A (Telemetry Only)**: Gradient Boosted Trees trained purely on local telemetry and cross-service lag features (22 features).
- **Model B (Telemetry + Workload Forecast)**: Gradient Boosted Trees integrating real-time telemetry with upstream multi-step workload forecasts (35 features; configured production default).
- **Model C (Temporal Comparator GRU)**: Bidirectional Gated Recurrent Unit capturing sequential temporal dynamics across 15-step windows.

### 4.2. Measured Classification & Calibration Performance

| Horizon | Model / Baseline | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier Score | ECE (10-bin) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=1 (20s)** | Static SLO Threshold | 0.0028 | 0.0500 | 0.0053 | N/A | N/A | N/A | N/A |
| **H=1 (20s)** | Z-Score Anomaly | 0.0000 | 0.0000 | 0.0000 | N/A | N/A | N/A | N/A |
| **H=1 (20s)** | **Model A (Telemetry Only)** | **0.6739** | 0.7750 | **0.7209** | **0.7543** | **0.9949** | **0.0055** | 0.0123 |
| **H=1 (20s)** | Model B (Telemetry + Forecast) | 0.5000 | **0.8500** | 0.6296 | 0.6280 | 0.9905 | 0.0061 | **0.0101** |
| **H=1 (20s)** | Model C (Temporal GRU) | 0.5094 | 0.6750 | 0.5806 | 0.5188 | 0.9894 | 0.0080 | 0.0130 |
| **H=5 (100s)** | Static SLO Threshold | 0.0036 | 0.0100 | 0.0053 | N/A | N/A | N/A | N/A |
| **H=5 (100s)** | Z-Score Anomaly | 0.0000 | 0.0000 | 0.0000 | N/A | N/A | N/A | N/A |
| **H=5 (100s)** | Model A (Telemetry Only) | 0.5107 | 0.7150 | 0.5958 | 0.7291 | 0.9402 | 0.0376 | 0.0644 |
| **H=5 (100s)** | **Model B (Telemetry + Forecast)** | 0.4351 | **0.9050** | 0.5877 | 0.8152 | 0.9695 | 0.0323 | **0.0534** |
| **H=5 (100s)** | **Model C (Temporal GRU)** | **0.5599** | 0.8650 | **0.6798** | **0.8249** | **0.9755** | **0.0309** | 0.0538 |
| **H=15 (300s)**| Static SLO Threshold | 0.0048 | 0.0030 | 0.0037 | N/A | N/A | N/A | N/A |
| **H=15 (300s)**| Z-Score Anomaly | 0.0000 | 0.0000 | 0.0000 | N/A | N/A | N/A | N/A |
| **H=15 (300s)**| Model A (Telemetry Only) | 0.3516 | **0.9668** | 0.5157 | 0.7835 | 0.9029 | 0.1229 | 0.1621 |
| **H=15 (300s)**| **Model B (Telemetry + Forecast)** | **0.4230** | 0.9456 | **0.5845** | **0.8660** | **0.9411** | **0.1060** | **0.1515** |
| **H=15 (300s)**| Model C (Temporal GRU) | 0.3941 | **0.9668** | 0.5599 | 0.8477 | 0.9399 | 0.1096 | 0.1534 |

### 4.3. Early Warning Lead Time & Actionable Recall

| Horizon | Model | Event Recall | Actionable Recall ($\ge 20$s) | Missed Events | Median Lead Time | P10 Lead Time | False Alarms / Day | Time-to-Onset MAE |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=1** | Static SLO | 5.0% | 5.0% | 38/40 | 20.0s | 20.0s | 66.39 | N/A |
| **H=1** | Model B | **85.0%** | **85.0%** | 6/40 | 20.0s | 20.0s | 44.81 | 0.0s (85.0% cov) |
| **H=5** | Static SLO | 5.0% | 5.0% | 38/40 | 20.0s | 20.0s | 76.37 | N/A |
| **H=5** | Model A | 100.0% | 100.0% | 0/40 | 80.0s | 38.0s | 152.73 | N/A |
| **H=5** | **Model B** | **100.0%** | **100.0%** | **0/40** | **100.0s** | **80.0s** | 160.37 | **12.0s (90.5% cov)** |
| **H=5** | Model C | 100.0% | 100.0% | 0/40 | 100.0s | 60.0s | 103.09 | N/A |
| **H=15** | Static SLO | 2.5% | 2.5% | 39/40 | 20.0s | 20.0s | 52.24 | N/A |
| **H=15** | Model A | 100.0% | 100.0% | 0/40 | 280.0s | 120.0s | 502.78 | N/A |
| **H=15** | **Model B** | **100.0%** | **100.0%** | **0/40** | **260.0s** | **120.0s** | **395.04** | **26.7s (94.6% cov)** |
| **H=15** | Model C | 100.0% | 100.0% | 0/40 | 280.0s | 120.0s | 408.10 | N/A |

### 4.4. Where Other Models Perform Better & Why Model B is Retained
1. **Short Horizon ($H=1$, 20s)**:
   - **Model A outperforms Model B** in F1 score (0.7209 vs 0.6296) and exhibits lower false alarm rates (24.90 vs 44.81/day).  
   - *Technical Explanation*: At 20s ahead, degradation is driven almost entirely by immediate local backpressure and queue depth. Introducing multi-step workload forecasts introduces redundant feature variance at this short scale.
2. **Medium Horizon ($H=5$, 100s)**:
   - **Model C (Temporal GRU) outperforms Model B** in F1 score (0.6798 vs 0.5877) and lower false alarms (103.09 vs 160.37/day).  
   - *Technical Explanation*: The bidirectional GRU smoothly accumulates 15-step sequential telemetry trends without tabular feature engineering.
3. **Rationale for Retaining Model B as Single Configured Default**:
   - Model B significantly outperforms Model A at long horizons ($H=15$: F1 0.5845 vs 0.5157; paired bootstrap $\Delta F1 = +0.0688$, 95% CI $[+0.0398, +0.0934]$, statistically significant).
   - Model B executes inference in $<1.5$ms on CPU and permits exact, real-time **TreeSHAP** explanations with zero approximation error, whereas deep GRUs require expensive background sampling or Integrated Gradients.

---

## 5. Robustness Analysis & Task 5A Measured Evidence

### 5.1. Reproducible Missing-Telemetry Experiment & Measured Evidence
In previous iterations, missing telemetry tests relied on external scratch scripts or inserted constants. Under Task 5A, the causal dropped-telemetry robustness evaluation is directly reproducible from a checked-in repository CLI command:
```bash
python3 ai-models/degradation_prediction/evaluate_task4.py --rerun-dropped-telemetry
```
This command directly re-evaluates the $H=5$ (100s) held-out condition, updates `results/degradation_prediction/task4_evaluation_metrics.json`, and outputs measured metrics for both Model B and the genuine training-fitted Z-score anomaly baseline.

- **Preprocessing Sensitivity vs Production Staleness Enforcement**:
  - `causal_missing_telemetry_pipeline` is formally documented as a **preprocessing sensitivity experiment** (tabular tolerance under carry-forward with refreshed timestamps).
  - **Causal Incomplete Telemetry Evaluation**: Evaluates raw 15% random dropouts from the telemetry stream with surviving genuine timestamps preserved, passing through actual production inference alignment:
    1. **Short Gaps ($\le 60$s)**: Verified causal carry-forward with `OBSERVATIONS_CARRIED_FORWARD` warnings.
       - **Model B ($H=5$) Measured**: $F_1 = 0.5844$, $PR\text{-}AUC = 0.7868$, Precision = 0.4327, Recall = 0.9000, with a **99.6% carry-forward rate** across affected windows.
       - **Z-Score Baseline Measured**: $F_1 = 0.0$, Precision = 0.0, Recall = 0.0 ($TP=0, FP=60, FN=200, TN=2490$). Because instantaneous thresholding cannot foresee pre-onset degradation 100 seconds ahead when current telemetry is still nominal, the baseline achieves zero recall and zero F1 on future forecasting. (Previously unmeasured/hard-coded constants are completely eliminated).
    2. **Long Gaps ($> 60$s)**: An 80s telemetry outage was tested against `MAX_TELEMETRY_STALENESS_SECONDS = 60.0s`. Exactly **100% of queries were rejected** ($2625 / 2625$) with structured HTTP 422 `STALENESS_LIMIT_EXCEEDED`. Zero nominal values were silently imputed.

### 5.2. Robustness Stress Matrix (Model B, $H=5$)

| Stress Condition | Description & Perturbation Details | Samples | F1 Score | PR-AUC | Telemetry & Observability Behavior |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **clean_reference** | Standard held-out test cohort (`standard_test`) | 2750 | **0.5877** | 0.8152 | Clean baseline performance. |
| **causal_missing_telemetry_pipeline** | 15% dropouts with refreshed timestamps (preprocessing sensitivity) | 2750 | **0.5922** | 0.8145 | Preprocessing sensitivity experiment. |
| **causal_incomplete_telemetry ($\le 60$s)** | 15% dropped, genuine timestamps preserved, causal carry-forward | 2750 | **0.5844** | 0.7868 | Model B F1=0.5844, PR-AUC=0.7868, Z-Score F1=0.0, 99.6% carry-forward rate. |
| **causal_incomplete_telemetry ($> 60$s)** | 80s telemetry outage exceeding staleness limit | 2625 | **REJECTED** | N/A | **100% rejected (HTTP 422/503), zero nominal imputation.** |
| **feature_zeroing_sensitivity** | 20% random tabular feature zeroing | 2750 | **0.5286** | 0.6072 | Uncalibrated tabular sensitivity. |
| **feature_space_noise** | $\pm 5\%$ Gaussian noise applied to continuous metrics | 2750 | **0.4444** | 0.6126 | Robustness to telemetry jitter. |
| **capacity_loss_test** | Unseen physical thread pool capacity collapse | 2750 | **0.5714** | 0.6437 | Rapid backlog backpressure detection. |
| **boundary_slowdown_test**| Unseen cross-generation gateway transit drift | 2750 | **0.6746** | 0.6402 | High sensitivity to gateway latency & sync drift. |
| **surge_test** | Unseen 2.5x sudden traffic bursts | 2750 | **0.6102** | 0.6506 | Early workload surge alert trigger. |

---

## 6. Subsystem 3: Explanations & Localization (Task 3)

### 6.1. TreeSHAP Attribution Guarantees
- Attributions are calculated directly on the model margin (log-odds of degradation) using `shap.TreeExplainer`.
- **Exact Additivity Guarantee**: $\sum_{i} \phi_i = f(x) - \mathbb{E}[f(X)]$. Verified reconstruction error $< 1\times 10^{-14}$ across all test cases.
- **Explicit Provenance**: Every attribution feature explicitly reports `source_provenance` (`measured` for eBPF/telemetry columns, `inferred` for workload forecasts/rollouts).
- **Domain-Specific Reference Baselines**: Reference values are computed from background training partition distributions, ensuring attributions represent deviations from healthy operations.

### 6.2. Graph-Based Localization & Honest Abstention States
- Interrogates service dependency graph edges (`reservations` $\to$ `boundary-gateway` $\to$ `legacy-core`).
- Calculates counterfactual risk reductions $\Delta R(n)$ by resetting suspected upstream node metrics to nominal reference baselines.
- **Honest State Mapping**:
  - **Ambiguous Ties**: When multiple candidate nodes produce indistinguishable risk reductions (`is_tie=True`), the system strictly returns `localization_status: "abstain_ambiguous_tie"` and **never** falsely reports the incident as `"localized"`.
  - **Insufficient Reduction**: When $\max_n \Delta R(n) < 0.05$, the system returns `localization_status: "abstain_unknown_unlocalized"`.
  - **Ranked Candidates**: The API returns actual `candidate_rankings` with node scores and counterfactual probabilities rather than referencing nonexistent fields.
- **Causality Disclaimer**: The brief and API responses explicitly state that attributions measure statistical model sensitivity and do not establish physical causality.

---

## 7. Subsystem 4: Backend Integration & Correctness Hardening (Task 5 & 5A)

### 7.1. Unified Alert Threshold Policy & Provenance
A single shared alert policy configuration (`backend/ml/alert_policy.py`) governs all operational decisions across the platform:
- **$H=1$ (20s)**: `0.280`
- **$H=5$ (100s)**: `0.300`
- **$H=15$ (300s)**: `0.320`

**Trace of Discrepancy**: The frozen model bundle `degradation_models_bundle.pkl` originally contained training-time heuristic attributes (`[0.74, 0.36, 0.35]`), causing the API to display `0.36` for $H=5$ while Task 4 evaluation was conducted at `0.300`. This discrepancy was resolved by binding prediction (`/predict/degradation`), explanation (`/explain/{service}`), incident briefs, alert-gated timing, and Prometheus metrics to the shared evaluated policy.  
**Provenance Disclosure**: These operational thresholds were predeclared in Task 4 to achieve balanced actionable early warning ($F_1 = 0.584$, Recall = 0.90) under asymmetric cost constraints. They were **not** derived from a formal validation-grid search and must not be described as mathematically optimal thresholds. Primary model weights and probability calibration remain strictly frozen.

### 7.2. Telemetry Adapter & Replay Clock Handling (`backend/ml/telemetry_adapter.py`)
- **Per-Service & Boundary Freshness**:
  $$\text{Freshness} = T_{\text{clock}} - T_{\text{latest}}(\text{service})$$
  Freshness is verified per requested service and boundary stream. A fresh neighbouring service **cannot** mask a stale requested service or boundary stream. Any freshness gap $> 60.0$s returns HTTP 422 `STALENESS_LIMIT_EXCEEDED`.
- **Clock Constancy Across Horizons**: The requested replay clock timestamp is strictly preserved across horizons ($H=1, 5, 15$) and is never rewound or clamped based on forecast horizon.
- **Parameter Validation**: Replay times and steps are validated as finite and non-negative; non-finite or negative parameters return structured HTTP 400 `INVALID_REPLAY_PARAMETER`. Requests with fewer than 4 observation steps return HTTP 422 `UNAVAILABLE_INSUFFICIENT_TELEMETRY`.

### 7.3. Forecast Error Timing & Cross-Incident Isolation (`backend/ml/metrics.py`)
- Workload forecast error is **not** recorded immediately upon forecast generation. It is registered as pending under a collision-proof key:
  $$(\text{incident\_id}, \text{service}, \text{horizon}, T_{\text{origin}}, T_{\text{target}})$$
- Pending forecasts are resolved only on subsequent requests when the replay clock reaches the target timestamp ($T_{\text{clock}} \ge T_{\text{target}}$), using only observations available at that clock.
- Forecasts from one incident are never matched against observations from another incident.
- Bounded memory (max 500 entries) and 10-minute TTL expiration prevent memory leakage.
- Repeated cache-hit requests do not duplicate pending forecasts or observations.

### 7.4. Concurrent Serving & Metric Responsiveness (`backend/api/app.py`)
- The server utilizes `ThreadingHTTPServer` to handle concurrent HTTP requests without thread starvation.
- CPU-heavy explanations are governed by a semaphore (`max 4 concurrent/queued requests`), returning HTTP 503 `EXPLANATION_CAPACITY_EXCEEDED` if saturated.
- `/metrics` responds immediately (< 200ms) even while heavy explanations execute in parallel.
- Initial `telemetry_available` is initialized to 0 (unverified) until verified by snapshot processing.

---

## 8. Frozen Artifact Hashes & Dataset Manifest

All model artifacts are frozen and verified by SHA-256 hashes:

| Artifact Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- |
| `ai-models/weights/workload_forecasting/workload_models_bundle.pkl` | `20c632c399997c7644f6e493e8a51d176ad3b3dc54272ed9df75b319624fa9f5` | Frozen bundle for GBDT, Ridge, and Conformal Calibrators |
| `ai-models/weights/degradation_prediction/degradation_models_bundle.pkl`| `3587d73ef55a06cb618b4277ab2ea030c2ad3c4b9b8dc0aabeb8c858776da885` | Frozen bundle for Models A, B, C and Isotonic Calibrators |
| `ai-models/data/degradation_prediction/degradation_dataset.json` | `b19ef70240364d34120fee019c6aa23c292643f8a5dff622ca1d4ba537f7768d` | 275-incident degradation trajectories and labels |
| `ai-models/data/workload_forecasting/workload_dataset.json` | `4eae4b853aede1635d714f0e46237f3ab2de6f13fd17f9fad219f93434278133` | Workload time-series dataset |
| `ai-models/weights/cascade_predictor_tier1a.pt` | `09f8d287fa468e535ed96502d259dc615647d83ec174214426aa2c0e0c5753e6` | PyTorch cascade predictor model |
| `ai-models/weights/circuit_breaker_ppo.pt` | `d00593818e2efbc3031f53acf6fbcf47e3e19592eaaeca00df49c53fe91a693b` | Reinforcement learning circuit breaker policy |

---

## 9. Codebase File Inventory by Subsystem

### 9.1. Backend API & ML Integration (`backend/`)
- `backend/ml/__init__.py`: Package initialization.
- `backend/ml/telemetry_adapter.py`: Dual-mode telemetry adapter, staleness enforcement, schema validation.
- `backend/ml/inference_service.py`: Singleton ML bundle loader, LRU cache, 2-worker executor, float sanitization.
- `backend/ml/metrics.py`: Prometheus metrics collector, alert deduplication, future observation error matching.
- `backend/api/app.py`: Extended HTTP REST server routing `/forecast/workload`, `/predict/degradation`, `/explain/{service}`, and `/metrics`.
- `backend/cloud/sagemaker.py`: Restored exception classes (`SageMakerError`, `SageMakerValidationError`, etc.).
- `backend/tests/test_ml_integration.py`: Comprehensive 12-condition integration test suite.
- `backend/API_DOCUMENTATION.md`: Complete API contracts, parameters, cURL examples, and error schemas.
- `backend/README.md`: Updated architecture documentation and configuration guide.

### 9.2. Machine Learning Core (`ai-models/`)
- `ai-models/workload_forecasting/`: Workload dataset generation, model definitions, conformal calibration, inference engine.
- `ai-models/degradation_prediction/`: Degradation dataset generation, Models A/B/C, TreeSHAP explainer, graph localization, evaluation suite.
- `ai-models/tests/`: Unit and integration test suites (85 tests covering forecasting, prediction, and explanations).

### 9.3. Results & Evaluation Data (`results/`)
- `results/workload_forecasting/`: Workload evaluation reports, metrics summary, conformal coverage plots.
- `results/degradation_prediction/`: Reconciled evaluation tables, bootstrap uncertainty, priority macros, stress matrices.
- `results/task5_example_responses.json`: Real response payloads captured from live in-process server.
- `results/task5_test_output.txt`: Measured test execution output.

---

## 10. Verification Test Results

### 10.1. Integration Test Suite Execution (`backend/tests/test_ml_integration.py`)
Executed via: `python3 -m unittest -v backend/tests/test_ml_integration.py`

```text
test_01_valid_workload_forecast (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify GET /forecast/workload returns valid point forecast, conformal bounds, and units. ... ok
test_02_valid_degradation_prediction (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify GET /predict/degradation returns calibrated probability, current state vs future risk, and advisory decision. ... ok
test_03_valid_explanation (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify GET /explain/{service} returns TreeSHAP attributions, top drivers, and graph localization. ... ok
test_04_prediction_and_explanation_snapshot_consistency (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify predictions and explanations for the same request refer to the identical snapshot and model. ... ok
test_05_invalid_service_and_horizon_rejection (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify 400 Bad Request on invalid service or horizon parameters. ... ok
test_06_live_mode_insufficient_telemetry_rejection (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify live mode returns structured 503 instead of silently imputing missing metrics. ... ok
test_07_stale_telemetry_rejection (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify telemetry exceeding 60-second staleness limit is rejected with HTTP 422. ... ok
test_07b_requested_service_stale_while_neighbours_fresh (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify a fresh neighbour does NOT mask stale requested-service data (HTTP 422). ... ok
test_07c_boundary_stale_while_services_fresh (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify stale boundary stream is rejected with HTTP 422 even when all services are fresh. ... ok
test_07d_replay_timestamp_unchanged_across_horizons (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify replay timestamp is not rewound or clamped based on forecast horizon. ... ok
test_07e_invalid_and_nonfinite_replay_arguments (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify invalid or non-finite replay parameters return structured 400 Bad Request. ... ok
test_08_missing_models_fallback_availability (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify missing ML model artifacts return structured 503 without crashing the backend. ... ok
test_09_cache_reuse_and_alert_deduplication (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify cache hits return identical results and do not re-increment Prometheus alert episode counter. ... ok
test_10_prometheus_metrics_exposition (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify /metrics exposes degradation probabilities, alert episodes, and forecast errors with bounded labels. ... ok
test_11_forecast_error_delayed_arrival_and_resolution (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify workload forecast error is NOT recorded until the target arrives at subsequent snapshot. ... ok
test_11b_no_cross_incident_matching_and_cache_deduplication (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify pending forecasts never match observations from a different incident, and cache hits deduplicate. ... ok
test_12_existing_api_routes_compatibility (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify all existing API routes continue to work without contract regressions. ... ok
test_13_unified_alert_thresholds_boundary_decisions_and_consistency (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify unified alert policy boundary decisions immediately below and at each threshold. ... ok
test_14_concurrent_metrics_responsiveness_during_slow_explanation (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify /metrics responds immediately while an explanation request is processing in parallel. ... ok
test_15_graph_localization_tie_and_candidate_rankings (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify graph localization tie maps to abstain_ambiguous_tie (not localized) and returns candidate rankings. ... ok
test_16_telemetry_availability_uninitialized_and_stale_update (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify telemetry_available starts at 0 (unverified) and marks unavailable on stale data. ... ok
test_17_injected_predictor_near_threshold_regression (backend.tests.test_ml_integration.TestMLBackendIntegration)
Regression: Verify core alert decision is preserved when probability = threshold - 0.00004 across all horizons, and at threshold. ... ok
test_18_explanation_capacity_held_until_work_finishes (backend.tests.test_ml_integration.TestMLBackendIntegration)
Verify explanation semaphore holds capacity across timeouts until work actually finishes. ... ok

----------------------------------------------------------------------
Ran 23 tests in 6.574s

OK
```

### 10.2. Core ML Test Suites (`ai-models/tests/`)
Executed via: `python3 -m unittest discover -s ai-models/tests -p "test_*.py"`
- `test_workload_forecasting.py`: 18/18 tests passed.
- `test_degradation_prediction.py`: 48/48 tests passed (including causal carry-forward and 60s staleness rejection).
- `test_degradation_explanations.py`: 19/19 tests passed (including exact TreeSHAP additivity and localization abstention).
- **Total ML Unit Tests**: **85/85 passed (100% success rate)**.

---

## 11. Live Telemetry Gaps & Unverified Behaviors

1. **eBPF Instrumentation Scope**:
   The existing eBPF probe captures `tcp_connection_attempt` counts only. Application-level metrics (measured latency percentiles, queue backlog, completed rates, and error rates) are not instrumented in live containers.
2. **Mainframe `legacy-core` Visibility**:
   `legacy-core` is an uninstrumented legacy mainframe. Its internal thread pools and queue backlogs cannot be directly scraped. The system models `legacy-core` through boundary gateway egress metrics.
3. **No Distributed Persistent Time-Series Store**:
   The backend does not include a distributed time-series database (e.g. InfluxDB or Prometheus remote storage). Historical windowing is handled in-memory.
4. **Process Restart Ephemerality**:
   Cached inferences and Prometheus counters (`baccp_advisory_alert_episodes_total`) reset upon process restart.
