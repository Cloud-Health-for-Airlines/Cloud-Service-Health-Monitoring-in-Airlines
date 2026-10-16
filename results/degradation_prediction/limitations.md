# Task 4A Limitations & Empirical Verification

## 1. What Synthetic Data Supports
- **Algorithmic Validation**: Verifies that gradient-boosted decision trees and temporal GRU comparators capture non-linear queue accumulation, cross-service propagation, and upstream workload changes across simulated airline services.
- **Invariants Verified**: Enforces causal timestamp alignment, exact TreeSHAP additivity (< 1e-14 error), absence of future data leakage, and rigorous censoring of unobservable forward breaches.
- **Baseline Behavior**: Demonstrates that while static SLO rules cannot generate proactive early warnings, statistical anomaly detection (Z-scores) can detect pre-breach variance, though with higher false discovery rates than trained models.

## 2. What Remains Unvalidated
- **Real Telemetry Noise and Artifacts**: Live OpenTelemetry streams contain asynchronous clock jitter, multi-region NTP drift, network dropped packets, and non-stationary baseline shifts not fully replicated by synthetic Gaussian perturbations.
- **Statistical Significance Limits**: Paired bootstrap comparisons reveal that while Model B's workload forecast features provide statistically established F1 gains at $H=15$ (+0.0665, 95% CI [0.0398, 0.0934]), at $H=5$ the improvement is statistically uncertain (95% CI crosses zero), and at $H=1$ Model A (telemetry-only) achieves higher F1.
- **Feature Attributions vs Physical Root Cause**: TreeSHAP and counterfactual interventions describe mathematical model sensitivity, NOT verified physical causes.
- **Telemetry Dropout Sensitivity vs Staleness Enforcement**: The `create_causal_missing_telemetry_cohort` test manually carried observations forward and refreshed timestamps, functioning as a preprocessing sensitivity experiment rather than verifying production staleness enforcement. Production staleness enforcement is verified separately by passing incomplete telemetry with surviving original timestamps through the inference alignment path: short gaps within 60s are causally aligned with carry-forward warnings (`OBSERVATIONS_CARRIED_FORWARD`), while gaps exceeding 60s are strictly rejected with structured unavailable states without nominal imputation.

## 3. Deployment Scope
- **Backend Prediction Endpoints**: The inference engine is ready for prediction endpoint integration. Live shadow deployment and circuit-breaker actuation are downstream deployment steps outside Task 4/5 endpoint scope.
