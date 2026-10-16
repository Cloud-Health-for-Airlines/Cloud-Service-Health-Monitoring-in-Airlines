# BACCP TASK 3A: REPRESENTATIVE INCIDENT BRIEFS & VERIFIED CASE EVIDENCE

This document compiles deterministic, human-readable incident briefs paired with computed model evidence across the 6 required operational scenarios. All cases were selected from empirically verified sustained degradation outcomes using standard observability rules.

## Case 1: Verified True Early Warning (TP - Sustained Degradation)

- **Incident ID**: `inc_val_0128`
- **Evaluated Service**: `crew` (H=1)
- **Prediction Origin Timestamp**: 420.0s | Forecast Window: [420.0s, 440.0s]
- **Label Observability Status**: `OBSERVABLE`
- **Current Operational State at Origin**: Within Nominal SLO Limits (Latency: 28.5ms < 50.0ms)
- **Actual Future Onset**: 440.0s
- **Predicted Alert**: True (Risk: 76.0%)
- **Classification**: `TP` | Lead Time: 20.0s
- **Estimated Time-to-Degradation**: 20.0s

```markdown
### BACCP INCIDENT EXPLANATION BRIEF: CREW [H=1 (20s)]
- **Affected Service**: crew (Generation: cloud)
- **Prediction Origin Timestamp**: 420.0s (t_origin)
- **Forecast Window**: [420.0s, 440.0s] (H=1 steps / 20s ahead, Target Ts: 440.0s)
- **Model Variant**: model_b_telemetry_forecast [Schema: baccp-degradation-features-v1.4.0]
- **Observed SLO State at Origin**: Within Nominal SLO Limits (Observed latency 22.0ms < 100.0ms, backlog 0.0 < 50.0)
- **Future Predicted Risk**: Degradation Risk: 76.0% (Calibrated event probability: 0.7600)
- **Alert Decision**: ALERT ACTIVE (Predicted risk 76.0% >= threshold 74.0%)
- **Conditional Time-to-Onset**: 20.0s (Alert-gated conditional estimate)
- **Predicted Type & Severity**: COMPOUND_MULTIPLE / NOMINAL

#### Top Supporting Drivers (Model Attribution):
1. **wf_capacity_margin** = 2.917 req/s (Ref: 52.2521 req/s) | Attribution: +1.7332 log-odds [Saturation]
2. **latency_roll_mean_3** = 22.4867 ms (Ref: 28.8341 ms) | Attribution: +1.4961 log-odds [Network]
3. **rate_divergence_abs** = 43.42 req/s (Ref: 1.6314 req/s) | Attribution: +0.9368 log-odds [Workload]

#### Graph Localization & Suspected Contributing Origin:
- **Suspected Contributing Node**: crew (Generation: cloud)
- **Propagation Bottleneck**: boundary-gateway
- **Verified Injected Origin**: UNAVAILABLE

#### Data Provenance & Reliability Limitations:
- **Telemetry Provenance**: SYNTHETIC_SIMULATED (eBPF captures TCP attempts; application latency/backlog simulated)
- **Input Reliability Status**: SYNTHETIC_SIMULATED (Warnings: None (Nominal inputs))
- **Attribution Methodology**: TreeSHAP (Reconstruction Error: 3.82e-15)
- **Causality Disclaimer**: Attributions measure statistical model sensitivity; they do NOT establish physical causality.
```

---

## Case 2: Verified False Alarm (FP - Premature Alert Without Follow-Up Breach in Window)

- **Incident ID**: `inc_val_0128`
- **Evaluated Service**: `crew` (H=1)
- **Prediction Origin Timestamp**: 400.0s | Forecast Window: [400.0s, 420.0s]
- **Label Observability Status**: `OBSERVABLE`
- **Current Operational State at Origin**: Within Nominal SLO Limits
- **Actual Future Onset**: None (No onset in window)
- **Predicted Alert**: True (Risk: 76.0%)
- **Classification**: `FP` | Lead Time: None
- **Estimated Time-to-Degradation**: 20.0s

```markdown
### BACCP INCIDENT EXPLANATION BRIEF: CREW [H=1 (20s)]
- **Affected Service**: crew (Generation: cloud)
- **Prediction Origin Timestamp**: 400.0s (t_origin)
- **Forecast Window**: [400.0s, 420.0s] (H=1 steps / 20s ahead, Target Ts: 420.0s)
- **Model Variant**: model_b_telemetry_forecast [Schema: baccp-degradation-features-v1.4.0]
- **Observed SLO State at Origin**: Within Nominal SLO Limits (Observed latency 22.9ms < 100.0ms, backlog 0.0 < 50.0)
- **Future Predicted Risk**: Degradation Risk: 76.0% (Calibrated event probability: 0.7600)
- **Alert Decision**: ALERT ACTIVE (Predicted risk 76.0% >= threshold 74.0%)
- **Conditional Time-to-Onset**: 20.0s (Alert-gated conditional estimate)
- **Predicted Type & Severity**: COMPOUND_MULTIPLE / NOMINAL

#### Top Supporting Drivers (Model Attribution):
1. **wf_capacity_margin** = 3.977 req/s (Ref: 52.2521 req/s) | Attribution: +1.3734 log-odds [Saturation]
2. **latency_roll_mean_3** = 22.5633 ms (Ref: 28.8341 ms) | Attribution: +1.1151 log-odds [Network]
3. **rate_divergence_abs** = 34.51 req/s (Ref: 1.6314 req/s) | Attribution: +0.9211 log-odds [Workload]

#### Graph Localization & Suspected Contributing Origin:
- **Suspected Contributing Node**: crew (Generation: cloud)
- **Propagation Bottleneck**: boundary-gateway
- **Verified Injected Origin**: UNAVAILABLE

#### Data Provenance & Reliability Limitations:
- **Telemetry Provenance**: SYNTHETIC_SIMULATED (eBPF captures TCP attempts; application latency/backlog simulated)
- **Input Reliability Status**: SYNTHETIC_SIMULATED (Warnings: None (Nominal inputs))
- **Attribution Methodology**: TreeSHAP (Reconstruction Error: 5.66e-15)
- **Causality Disclaimer**: Attributions measure statistical model sensitivity; they do NOT establish physical causality.
```

---

## Case 3: Verified Missed Event (FN - Sub-Threshold Risk Preceding Sustained Breach)

- **Incident ID**: `inc_val_0128`
- **Evaluated Service**: `reservations` (H=1)
- **Prediction Origin Timestamp**: 380.0s | Forecast Window: [380.0s, 400.0s]
- **Label Observability Status**: `OBSERVABLE`
- **Current Operational State at Origin**: Within Nominal SLO Limits
- **Actual Future Onset**: 400.0s
- **Predicted Alert**: False (Risk: 29.4%)
- **Classification**: `FN` | Lead Time: None
- **Estimated Time-to-Degradation**: Unavailable (No alert or nominal)

```markdown
### BACCP INCIDENT EXPLANATION BRIEF: RESERVATIONS [H=1 (20s)]
- **Affected Service**: reservations (Generation: cloud)
- **Prediction Origin Timestamp**: 380.0s (t_origin)
- **Forecast Window**: [380.0s, 400.0s] (H=1 steps / 20s ahead, Target Ts: 400.0s)
- **Model Variant**: model_b_telemetry_forecast [Schema: baccp-degradation-features-v1.4.0]
- **Observed SLO State at Origin**: Within Nominal SLO Limits (Observed latency 17.9ms < 100.0ms, backlog 0.0 < 50.0)
- **Future Predicted Risk**: Degradation Risk: 29.4% (Calibrated event probability: 0.2941)
- **Alert Decision**: NO ALERT ISSUED (Predicted risk 29.4% < threshold 74.0%)
- **Conditional Time-to-Onset**: Unavailable: Conditional time-to-degradation estimate is gated to active alert states (alert_issued=False; risk 29.4% < threshold 74.0%).
- **Predicted Type & Severity**: NOMINAL / NOMINAL

#### Top Supporting Drivers (Model Attribution):
1. **wf_capacity_margin** = -4.424 req/s (Ref: 52.2521 req/s) | Attribution: +1.3936 log-odds [Saturation]
2. **latency_roll_mean_3** = 17.75 ms (Ref: 28.8341 ms) | Attribution: +1.0954 log-odds [Network]
3. **rate_divergence_abs** = 35.22 req/s (Ref: 1.6314 req/s) | Attribution: +0.947 log-odds [Workload]

#### Graph Localization & Suspected Contributing Origin:
- **Suspected Contributing Node**: reservations (Generation: cloud)
- **Propagation Bottleneck**: boundary-gateway
- **Verified Injected Origin**: UNAVAILABLE

#### Data Provenance & Reliability Limitations:
- **Telemetry Provenance**: SYNTHETIC_SIMULATED (eBPF captures TCP attempts; application latency/backlog simulated)
- **Input Reliability Status**: SYNTHETIC_SIMULATED (Warnings: None (Nominal inputs))
- **Attribution Methodology**: TreeSHAP (Reconstruction Error: 2.66e-15)
- **Causality Disclaimer**: Attributions measure statistical model sensitivity; they do NOT establish physical causality.
```

---

## Case 4: Verified True Negative (TN - Stable Nominal Operation)

- **Incident ID**: `inc_val_0120`
- **Evaluated Service**: `reservations` (H=1)
- **Prediction Origin Timestamp**: 60.0s | Forecast Window: [60.0s, 80.0s]
- **Label Observability Status**: `OBSERVABLE`
- **Current Operational State at Origin**: Within Nominal SLO Limits
- **Actual Future Onset**: None (No onset in window)
- **Predicted Alert**: False (Risk: 1.8%)
- **Classification**: `TN` | Lead Time: None
- **Estimated Time-to-Degradation**: Unavailable (No alert or nominal)

```markdown
### BACCP INCIDENT EXPLANATION BRIEF: RESERVATIONS [H=1 (20s)]
- **Affected Service**: reservations (Generation: cloud)
- **Prediction Origin Timestamp**: 60.0s (t_origin)
- **Forecast Window**: [60.0s, 80.0s] (H=1 steps / 20s ahead, Target Ts: 80.0s)
- **Model Variant**: model_b_telemetry_forecast [Schema: baccp-degradation-features-v1.4.0]
- **Observed SLO State at Origin**: Within Nominal SLO Limits (Observed latency 17.3ms < 100.0ms, backlog 0.0 < 50.0)
- **Future Predicted Risk**: Degradation Risk: 1.8% (Calibrated event probability: 0.0175)
- **Alert Decision**: NO ALERT ISSUED (Predicted risk 1.8% < threshold 74.0%)
- **Conditional Time-to-Onset**: Unavailable: Conditional time-to-degradation estimate is gated to active alert states (alert_issued=False; risk 1.8% < threshold 74.0%).
- **Predicted Type & Severity**: NOMINAL / NOMINAL

#### Top Supporting Drivers (Model Attribution):
1. **wf_fallback_active** = 1.0 boolean {0,1} (Ref: 1.0 boolean {0,1}) | Attribution: +0.4324 log-odds [Uncertainty And Reliability]
2. **latency_roll_mean_3** = 17.4967 ms (Ref: 28.8341 ms) | Attribution: +0.4161 log-odds [Network]
3. **curr_completed_rate** = 50.17 req/s (Ref: 40.3428 req/s) | Attribution: +0.3648 log-odds [Workload]

#### Graph Localization & Suspected Contributing Origin:
- **Suspected Contributing Node**: unknown_unlocalized (Generation: unknown)
- **Propagation Bottleneck**: none_identified
- **Verified Injected Origin**: UNAVAILABLE

#### Data Provenance & Reliability Limitations:
- **Telemetry Provenance**: SYNTHETIC_SIMULATED (eBPF captures TCP attempts; application latency/backlog simulated)
- **Input Reliability Status**: SYNTHETIC_SIMULATED (Warnings: None (Nominal inputs))
- **Attribution Methodology**: TreeSHAP (Reconstruction Error: 8.88e-16)
- **Causality Disclaimer**: Attributions measure statistical model sensitivity; they do NOT establish physical causality.
```

---

## Case 5: Telemetry Stream Truncation & Causal Grid Imputation

- **Incident ID**: `inc_val_0120_truncated_crew`
- **Evaluated Service**: `reservations` (H=1)
- **Prediction Origin Timestamp**: 60.0s | Forecast Window: [60.0s, 80.0s]
- **Label Observability Status**: `OBSERVABLE`
- **Current Operational State at Origin**: Within Nominal SLO Limits
- **Actual Future Onset**: None (No onset in window)
- **Predicted Alert**: False (Risk: 1.8%)
- **Classification**: `TN_WITH_WARNINGS` | Lead Time: None
- **Estimated Time-to-Degradation**: Unavailable (No alert or nominal)

```markdown
### BACCP INCIDENT EXPLANATION BRIEF: RESERVATIONS [H=1 (20s)]
- **Affected Service**: reservations (Generation: cloud)
- **Prediction Origin Timestamp**: 60.0s (t_origin)
- **Forecast Window**: [60.0s, 80.0s] (H=1 steps / 20s ahead, Target Ts: 80.0s)
- **Model Variant**: model_b_telemetry_forecast [Schema: baccp-degradation-features-v1.4.0]
- **Observed SLO State at Origin**: Within Nominal SLO Limits (Observed latency 17.3ms < 100.0ms, backlog 0.0 < 50.0)
- **Future Predicted Risk**: Degradation Risk: 1.8% (Calibrated event probability: 0.0175)
- **Alert Decision**: NO ALERT ISSUED (Predicted risk 1.8% < threshold 74.0%)
- **Conditional Time-to-Onset**: Unavailable: Conditional time-to-degradation estimate is gated to active alert states (alert_issued=False; risk 1.8% < threshold 74.0%).
- **Predicted Type & Severity**: NOMINAL / NOMINAL

#### Top Supporting Drivers (Model Attribution):
1. **wf_fallback_active** = 1.0 boolean {0,1} (Ref: 1.0 boolean {0,1}) | Attribution: +0.4324 log-odds [Uncertainty And Reliability]
2. **latency_roll_mean_3** = 17.4967 ms (Ref: 28.8341 ms) | Attribution: +0.4161 log-odds [Network]
3. **curr_completed_rate** = 50.17 req/s (Ref: 40.3428 req/s) | Attribution: +0.3648 log-odds [Workload]

#### Graph Localization & Suspected Contributing Origin:
- **Suspected Contributing Node**: unknown_unlocalized (Generation: unknown)
- **Propagation Bottleneck**: none_identified
- **Verified Injected Origin**: UNAVAILABLE

#### Data Provenance & Reliability Limitations:
- **Telemetry Provenance**: SYNTHETIC_SIMULATED (eBPF captures TCP attempts; application latency/backlog simulated)
- **Input Reliability Status**: SYNTHETIC_SIMULATED (Warnings: OBSERVATIONS_CARRIED_FORWARD)
- **Attribution Methodology**: TreeSHAP (Reconstruction Error: 8.88e-16)
- **Causality Disclaimer**: Attributions measure statistical model sensitivity; they do NOT establish physical causality.
```

---

## Case 6: Unmeasured Legacy Capacity Handling (legacy-core)

- **Incident ID**: `inc_val_0120`
- **Evaluated Service**: `legacy-core` (H=1)
- **Prediction Origin Timestamp**: 60.0s | Forecast Window: [60.0s, 80.0s]
- **Label Observability Status**: `OBSERVABLE`
- **Current Operational State at Origin**: Within Nominal SLO Limits
- **Actual Future Onset**: None (No onset in window)
- **Predicted Alert**: False (Risk: 3.5%)
- **Classification**: `TN_UNMEASURED_CAPACITY` | Lead Time: None
- **Estimated Time-to-Degradation**: Unavailable (No alert or nominal)

```markdown
### BACCP INCIDENT EXPLANATION BRIEF: LEGACY-CORE [H=1 (20s)]
- **Affected Service**: legacy-core (Generation: legacy)
- **Prediction Origin Timestamp**: 60.0s (t_origin)
- **Forecast Window**: [60.0s, 80.0s] (H=1 steps / 20s ahead, Target Ts: 80.0s)
- **Model Variant**: model_a_telemetry_only [Schema: baccp-degradation-features-v1.4.0]
- **Observed SLO State at Origin**: Within Nominal SLO Limits (Observed latency 68.6ms < 250.0ms, backlog 0.0 < 50.0)
- **Future Predicted Risk**: Degradation Risk: 3.5% (Calibrated event probability: 0.0351)
- **Alert Decision**: NO ALERT ISSUED (Predicted risk 3.5% < threshold 15.0%)
- **Conditional Time-to-Onset**: Unavailable: Conditional time-to-degradation estimate is gated to active alert states (alert_issued=False; risk 3.5% < threshold 15.0%).
- **Predicted Type & Severity**: NOMINAL / NOMINAL

#### Top Supporting Drivers (Model Attribution):
1. **latency_roll_mean_3** = 68.6733 ms (Ref: 28.8341 ms) | Attribution: +0.7107 log-odds [Network]
2. **curr_latency_ms** = 68.56 ms (Ref: 28.8451 ms) | Attribution: +0.4382 log-odds [Network]
3. **global_max_latency_ratio** = 1.0708 ratio (Ref: 1.0364 ratio) | Attribution: +0.3382 log-odds [Network]

#### Graph Localization & Suspected Contributing Origin:
- **Suspected Contributing Node**: unknown_unlocalized (Generation: unknown)
- **Propagation Bottleneck**: none_identified
- **Verified Injected Origin**: UNAVAILABLE

#### Data Provenance & Reliability Limitations:
- **Telemetry Provenance**: SYNTHETIC_SIMULATED (eBPF captures TCP attempts; application latency/backlog simulated)
- **Input Reliability Status**: SYNTHETIC_SIMULATED (Warnings: None (Nominal inputs))
- **Attribution Methodology**: TreeSHAP (Reconstruction Error: 2.66e-15)
- **Causality Disclaimer**: Attributions measure statistical model sensitivity; they do NOT establish physical causality.
```

---

