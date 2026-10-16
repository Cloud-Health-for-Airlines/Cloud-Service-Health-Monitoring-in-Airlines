# BACCP Service and Boundary Degradation Prediction (Task 2C)

## Overview

The Degradation Prediction module forecasts multi-step forward-window service SLO violations and boundary synchronization deterioration across cloud airline services (`reservations`, `crew`, `baggage`, `boundary-gateway`, `legacy-core`).

## Discrepancy Reconciliation (Task 2B Audit)

- **Root Cause Confirmed**: In Task 2B, diagnostic evaluation commands executed `extract_split_samples` with `forecaster=None`, which triggered the heuristic persistence fallback ($D_{\text{fc}} = \lambda_{\text{curr}}$). This persistence fallback yielded $H=1$ recall of **6/8** (Model A) and **2/8** (Model B).
- In contrast, the full pipeline loaded `WorkloadForecaster.get_instance()`, feeding trained out-of-sample multi-step ML forecasts into Model B, yielding $H=1$ recall of **7/8** (Model A) and **3/8** (Model B).
- **Resolution**: All diagnostic fallback executions are explicitly distinguished from trained ML forecast evaluations. Magic indices have been replaced with versioned lookups (`get_feature_index`) and strict schema assertions (`assert_feature_schema`).

## Target Definitions

1. **Target 1: Probability of Confirmed Sustained Degradation**:
   - Predicts the forward window $(t, t+H]$ degradation probability across $H \in \{1, 5, 15\}$ (20s, 100s, 300s).
   - **Operational Rule**: Degradation is confirmed only after **three consecutive breached snapshots** ($\ge 3$ steps exceeding SLO).
   - Early warning lead time is measured from alert timestamp to $t_{\text{onset}}$ (timestamp of the *first* breached snapshot in the confirmed sequence).
   - Valid warning criterion: $0 < t_{\text{onset}} - t_{\text{alert}} \le H \times 20\text{s}$.
2. **Target 2: Conditional Time-to-Onset Regression**:
   - Predicts estimated seconds until degradation onset.
   - Incidents without observed onset are treated as **censored**; non-events return `None`.
   - Evaluated conditionally on eligible observed events.
3. **Target 3: Observable Degradation Type & Severity**:
   - Types: `LATENCY_BREACH`, `BACKLOG_SATURATION`, `ERROR_SPIKE`, `BOUNDARY_DRIFT_DETERIORATION`, `COMPOUND_MULTIPLE`.
   - Severity: `NOMINAL`, `MINOR`, `MAJOR`, `CRITICAL` based on documented SLO exceedance ratios.

## Model Comparison

1. **Persistence Baseline**: Propagates current telemetry forward.
2. **Static Thresholding Baseline**: Triggers alert when current metrics reach 75% of SLO threshold.
3. **Simple Velocity Anomaly Baseline**: Alerts when latency/backlog/drift rate of change exceeds $2\sigma$.
4. **Model A (Telemetry Only)**: Gradient-boosted trees over 22 historical telemetry features.
5. **Model B (Telemetry + Workload Forecast)**: Gradient-boosted trees over 35 features including multi-step workload forecasts, conformal interval width, capacity headroom, and causal rollout queue delay.
6. **Model B System-Calibrated**: Post-processed through isotonic regression fitted on aggregated calibration origins.
7. **Model C (PyTorch Temporal GRU Comparator)**: 2-layer GRU (hidden dim 32) processing $(4, 16)$ sequence tensors with dependency-graph neighbor context and capacity headroom.

## Reproduction Commands

```bash
# 1. Run all unit tests (30 degradation tests + 16 workload forecasting tests)
PYTHONPATH="ai-models:." python -m pytest ai-models/tests/test_degradation_prediction.py ai-models/tests/test_workload_forecasting.py -v

# 2. Re-train all models and save bundle
PYTHONPATH="ai-models:." python ai-models/degradation_prediction/train.py

# 3. Run full evaluation across all splits and baselines
PYTHONPATH="ai-models:." python ai-models/degradation_prediction/evaluate.py

# 4. Lock manifest and evaluate on untouched final assessment set
PYTHONPATH="ai-models:." python ai-models/degradation_prediction/lock_and_evaluate_untouched.py

# 5. Build review ZIP bundle
PYTHONPATH="ai-models:." python ai-models/degradation_prediction/package_review_zip.py
```

## Priority Weighting Formula

$$\text{Metric}_{\text{priority}} = \frac{1.5 \times \text{Metric}_{\text{reservations}} + 1.0 \times \text{Metric}_{\text{crew}} + 0.5 \times \text{Metric}_{\text{baggage}}}{3.0}$$

## Task 3: Degradation Prediction Explanation Engine

### Overview
`DegradationExplainer` provides transparent, model-grounded explanations for BACCP degradation predictions:
1. **Tree SHAP Attributions (Model A & Model B)**: Explains the pre-calibrated log-odds decision margin $f(x)$ with machine-precision completeness. Uses training-only background samples. Retains positive and negative drivers.
2. **Integrated Gradients (Model C Temporal GRU)**: Path-integrated gradients evaluated on CPU with trapezoidal quadrature, decomposing event logit across 4 timesteps and 16 features.
3. **5 Feature Groups**:
   - `workload`: offered rate, completed rate, throughput ratio, dropped requests, rate divergence, forecast demand, rollout arrivals.
   - `saturation`: backlog, roll max backlog, backlog delta, global max backlog, capacity headroom, overload, rollout backlog.
   - `network`: latency, roll mean latency, latency delta, error rate, global max latency ratio, rollout latency projection.
   - `boundary_gateway`: sync drift score, gateway queue depth, drift delta, boundary latency, boundary error rate.
   - `uncertainty_and_reliability`: interval lower, interval upper, interval width, fallback active.
4. **Graph-Aware Suspected-Origin Localization**:
   - Evaluates node-group counterfactual perturbations (restoring each candidate node to nominal baseline).
   - Distinguishes affected service, suspected contributing node, propagation bottleneck, and verified injected origin (UNAVAILABLE on production dataset).
   - Returns `unknown_unlocalized` when maximum risk reduction is $< 5\%$.
5. **Deterministic Incident Briefs**:
   - Structured markdown briefs populated strictly from computed model predictions and attributions.
   - Differentiates "no alert" from "confirmed healthy".

### Task 3 Reproduction Commands

```bash
# 1. Run all unit and regression tests (including Task 3 explainer tests)
PYTHONPATH="ai-models:." python -m pytest ai-models/tests/test_degradation_prediction.py ai-models/tests/test_degradation_explanations.py -v

# 2. Run full explanation evaluation benchmark
PYTHONPATH="ai-models:." python ai-models/degradation_prediction/evaluate_explanations.py

# 3. Package Task 3 review bundle and manifest
PYTHONPATH="ai-models:." python ai-models/degradation_prediction/package_task3_bundle.py
```

## Cumulative Change Log

- **Task 2A**: Strict forward-window early-warning accounting ($0 < t_{\text{onset}} - t_{\text{alert}} \le H \times 20\text{s}$), alert deduplication cooldown, first onset vs recurrence tracking.
- **Task 2B**: Data provenance audit (eBPF captures vs simulated metrics), runtime observability audit, priority-weighted metrics.
- **Task 2C**: Discrepancy reconciliation (diagnostic `forecaster=None` persistence fallback vs trained forecaster), schema assertions (`baccp-degradation-features-v1.2.0`), zero magic indices.
- **Task 2D**: Dynamic discrepancy computation without hardcoded counts, Model A forecaster invariance, origin-only operational state determination, recovery eligibility rule.
- **Task 2E**: Shared workload forecaster adapter (`AdaptedWorkloadForecast`), regular sampling grid alignment (`dt = 20s`), future-only stream rejection, documented staleness limit (`MAX_TELEMETRY_STALENESS_SECONDS = 60s`).
- **Task 3**:
  - Prerequisite 1: Regular sampling grid alignment with source timestamps and imputation flags, eliminating unequal padding timestamps `[0,20,40,60,80]` vs `[0,0,20,60,80]`.
  - Prerequisite 2: Heuristic persistence fallback marked uncalibrated with `nominal_coverage=None` and `is_calibrated=False`.
  - Prerequisite 3: Artifact load assertion (raises `RuntimeError` if forecaster artifact failed to load).
  - Prerequisite 4: Packaging manifest exclusion (manifest excludes itself from files list; separate manifest and archive hashes).
  - Production `DegradationExplainer` implementing Tree SHAP, Integrated Gradients, 5 feature domains, node-group perturbation localization, and deterministic incident briefs.
  - Evaluation benchmark, diagnostic fixture (`diagnostic_localization_fixture.json`), and comprehensive regression test suite (`test_degradation_explanations.py`).

