# Programmatic Reconciled Evaluation Tables (Task 4A)

## Table 1: Overall Classification & Calibration Across Horizons

| Horizon | Model / Baseline | Sample Count | Prevalence | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier Score | ECE (10-bin) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=1 (20s)** | Static SLO Threshold Alerting | 3250 | 0.0123 | 0.0028 | 0.0500 | **0.0053** | N/A | N/A | N/A | N/A |
| **H=1 (20s)** | Training-Fitted Z-Score Anomaly | 3250 | 0.0123 | 0.0000 | 0.0000 | **0.0000** | N/A | N/A | N/A | N/A |
| **H=1 (20s)** | Model A (Telemetry Only) | 3250 | 0.0123 | 0.6739 | 0.7750 | **0.7209** | 0.7543 | 0.9949 | 0.0055 | 0.0123 |
| **H=1 (20s)** | Model B (Telemetry + Forecast) | 3250 | 0.0123 | 0.5000 | 0.8500 | **0.6296** | 0.6280 | 0.9905 | 0.0061 | 0.0101 |
| **H=1 (20s)** | Model C (Temporal Comparator GRU) | 3250 | 0.0123 | 0.5094 | 0.6750 | **0.5806** | 0.5188 | 0.9894 | 0.0080 | 0.0130 |
| **H=5 (100s)** | Static SLO Threshold Alerting | 2750 | 0.0727 | 0.0036 | 0.0100 | **0.0053** | N/A | N/A | N/A | N/A |
| **H=5 (100s)** | Training-Fitted Z-Score Anomaly | 2750 | 0.0727 | 0.0000 | 0.0000 | **0.0000** | N/A | N/A | N/A | N/A |
| **H=5 (100s)** | Model A (Telemetry Only) | 2750 | 0.0727 | 0.5107 | 0.7150 | **0.5958** | 0.7291 | 0.9402 | 0.0376 | 0.0644 |
| **H=5 (100s)** | Model B (Telemetry + Forecast) | 2750 | 0.0727 | 0.4351 | 0.9050 | **0.5877** | 0.8152 | 0.9695 | 0.0323 | 0.0534 |
| **H=5 (100s)** | Model C (Temporal Comparator GRU) | 2750 | 0.0727 | 0.5599 | 0.8650 | **0.6798** | 0.8249 | 0.9755 | 0.0309 | 0.0538 |
| **H=15 (300s)** | Static SLO Threshold Alerting | 1500 | 0.2207 | 0.0048 | 0.0030 | **0.0037** | N/A | N/A | N/A | N/A |
| **H=15 (300s)** | Training-Fitted Z-Score Anomaly | 1500 | 0.2207 | 0.0000 | 0.0000 | **0.0000** | N/A | N/A | N/A | N/A |
| **H=15 (300s)** | Model A (Telemetry Only) | 1500 | 0.2207 | 0.3516 | 0.9668 | **0.5157** | 0.7835 | 0.9029 | 0.1229 | 0.1621 |
| **H=15 (300s)** | Model B (Telemetry + Forecast) | 1500 | 0.2207 | 0.4230 | 0.9456 | **0.5845** | 0.8660 | 0.9411 | 0.1060 | 0.1515 |
| **H=15 (300s)** | Model C (Temporal Comparator GRU) | 1500 | 0.2207 | 0.3941 | 0.9668 | **0.5599** | 0.8477 | 0.9399 | 0.1096 | 0.1534 |

## Table 2: Bootstrap Uncertainty & Paired Comparisons (500 Incident-Cluster Replicates)

| Horizon | Comparison / Model | Observed F1 / Diff | Bootstrap Mean F1 / Diff | 95% Confidence Interval | Statistically Significant (Excludes 0)? |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **H=1** | Model B F1 Score | 0.6296 | 0.6292 | [0.5455, 0.7015] | N/A (Level) |
| **H=1** | Model B vs Model A (Delta F1) | -0.0913 | -0.0879 | [-0.1657, -0.0059] | **Yes (Excludes 0)** |
| **H=1** | Model B vs Model C (Delta F1) | +0.0490 | +0.0390 | [-0.1100, +0.1890] | **No (Crosses 0)** |
| **H=1** | Model B vs Static Baseline (Delta F1) | +0.6244 | +0.6235 | [+0.5430, +0.6885] | **Yes (Excludes 0)** |
| **H=1** | Model B vs Z-Score Baseline (Delta F1) | +0.6296 | +0.6292 | [+0.5455, +0.7015] | **Yes (Excludes 0)** |
| **H=5** | Model B F1 Score | 0.5877 | 0.5810 | [0.3532, 0.7596] | N/A (Level) |
| **H=5** | Model B vs Model A (Delta F1) | -0.0082 | -0.0038 | [-0.0707, +0.0723] | **No (Crosses 0)** |
| **H=5** | Model B vs Model C (Delta F1) | -0.0921 | -0.0887 | [-0.1401, -0.0321] | **Yes (Excludes 0)** |
| **H=5** | Model B vs Static Baseline (Delta F1) | +0.5824 | +0.5753 | [+0.3488, +0.7505] | **Yes (Excludes 0)** |
| **H=5** | Model B vs Z-Score Baseline (Delta F1) | +0.5877 | +0.5810 | [+0.3532, +0.7596] | **Yes (Excludes 0)** |
| **H=15** | Model B F1 Score | 0.5845 | 0.5790 | [0.3459, 0.7734] | N/A (Level) |
| **H=15** | Model B vs Model A (Delta F1) | +0.0688 | +0.0665 | [+0.0398, +0.0934] | **Yes (Excludes 0)** |
| **H=15** | Model B vs Model C (Delta F1) | +0.0246 | +0.0243 | [-0.0016, +0.0480] | **No (Crosses 0)** |
| **H=15** | Model B vs Static Baseline (Delta F1) | +0.5808 | +0.5750 | [+0.3431, +0.7698] | **Yes (Excludes 0)** |
| **H=15** | Model B vs Z-Score Baseline (Delta F1) | +0.5845 | +0.5790 | [+0.3459, +0.7734] | **Yes (Excludes 0)** |

## Table 3: Early Warning, Actionable Recall, and Exposure Accounting

| Horizon | Model | Event Recall | Actionable Recall (>=20s) | Missed Events | Median Lead Time | P10 Lead Time | False Alarms / Day | Healthy Exposure Days | Time-to-Onset MAE (Coverage) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=1** | Static SLO Threshold Alerting | 5.0% | 5.0% | 38/40 | 20.0s | 20.0s | 66.39 | 0.6025 | N/A |
| **H=1** | Training-Fitted Z-Score Anomaly | 0.0% | 0.0% | 40/40 | N/A | N/A | 26.56 | 0.6025 | N/A |
| **H=1** | Model A (Telemetry Only) | 77.5% | 77.5% | 9/40 | 20.0s | 20.0s | 24.90 | 0.6025 | N/A |
| **H=1** | Model B (Telemetry + Forecast) | 85.0% | 85.0% | 6/40 | 20.0s | 20.0s | 44.81 | 0.6025 | 0.0s (85.0%) |
| **H=1** | Model C (Temporal Comparator GRU) | 67.5% | 67.5% | 13/40 | 20.0s | 20.0s | 34.85 | 0.6025 | N/A |
| **H=5** | Static SLO Threshold Alerting | 5.0% | 5.0% | 38/40 | 20.0s | 20.0s | 76.37 | 0.5238 | N/A |
| **H=5** | Training-Fitted Z-Score Anomaly | 0.0% | 0.0% | 40/40 | N/A | N/A | 30.55 | 0.5238 | N/A |
| **H=5** | Model A (Telemetry Only) | 100.0% | 100.0% | 0/40 | 80.0s | 38.0s | 152.73 | 0.5238 | N/A |
| **H=5** | Model B (Telemetry + Forecast) | 100.0% | 100.0% | 0/40 | 100.0s | 80.0s | 160.37 | 0.5238 | 12.0s (90.5%) |
| **H=5** | Model C (Temporal Comparator GRU) | 100.0% | 100.0% | 0/40 | 100.0s | 60.0s | 103.09 | 0.5238 | N/A |
| **H=15** | Static SLO Threshold Alerting | 2.5% | 2.5% | 39/40 | 20.0s | 20.0s | 52.24 | 0.3063 | N/A |
| **H=15** | Training-Fitted Z-Score Anomaly | 0.0% | 0.0% | 40/40 | N/A | N/A | 3.26 | 0.3063 | N/A |
| **H=15** | Model A (Telemetry Only) | 100.0% | 100.0% | 0/40 | 280.0s | 120.0s | 502.78 | 0.3063 | N/A |
| **H=15** | Model B (Telemetry + Forecast) | 100.0% | 100.0% | 0/40 | 260.0s | 120.0s | 395.04 | 0.3063 | 26.7s (94.6%) |
| **H=15** | Model C (Temporal Comparator GRU) | 100.0% | 100.0% | 0/40 | 280.0s | 120.0s | 408.10 | 0.3063 | N/A |

## Table 4: Priority-Weighted Macro (3 Domains) vs Separated Infrastructure

| Horizon | Model | Priority Macro Precision | Priority Macro Recall | Priority Macro F1 | Reservations F1 | Crew F1 | Baggage F1 | Boundary Gateway F1 (Sep) | Legacy Core F1 (Sep) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=1** | Static SLO Threshold Alerting | 0.0052 | 0.0417 | **0.0093** | 0.0000 | 0.0278 | 0.0000 | 0.0000 | 0.0074 |
| **H=1** | Training-Fitted Z-Score Anomaly | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **H=1** | Model A (Telemetry Only) | 0.6667 | 0.7917 | **0.7216** | 0.7059 | 0.7059 | 0.8000 | 0.6667 | 0.7059 |
| **H=1** | Model B (Telemetry + Forecast) | 0.4939 | 0.8125 | **0.6113** | 0.7000 | 0.4762 | 0.6154 | 0.7778 | 0.6087 |
| **H=1** | Model C (Temporal Comparator GRU) | 0.3909 | 0.5417 | **0.4534** | 0.4000 | 0.4444 | 0.6316 | 0.7500 | 0.7000 |
| **H=5** | Static SLO Threshold Alerting | 0.0104 | 0.0083 | **0.0093** | 0.0000 | 0.0278 | 0.0000 | 0.0000 | 0.0074 |
| **H=5** | Training-Fitted Z-Score Anomaly | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **H=5** | Model A (Telemetry Only) | 0.7130 | 0.8792 | **0.7841** | 0.7391 | 0.8537 | 0.7800 | 0.3600 | 0.3208 |
| **H=5** | Model B (Telemetry + Forecast) | 0.7758 | 0.9750 | **0.8638** | 0.8478 | 0.8864 | 0.8667 | 0.3636 | 0.3757 |
| **H=5** | Model C (Temporal Comparator GRU) | 0.8285 | 0.9625 | **0.8898** | 0.8989 | 0.8675 | 0.9070 | 0.4748 | 0.4464 |
| **H=15** | Static SLO Threshold Alerting | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0127 |
| **H=15** | Training-Fitted Z-Score Anomaly | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **H=15** | Model A (Telemetry Only) | 0.4121 | 0.9569 | **0.5703** | 0.5048 | 0.6533 | 0.6008 | 0.4262 | 0.4215 |
| **H=15** | Model B (Telemetry + Forecast) | 0.5396 | 0.9454 | **0.6752** | 0.5918 | 0.7778 | 0.7200 | 0.4425 | 0.4537 |
| **H=15** | Model C (Temporal Comparator GRU) | 0.5450 | 0.9534 | **0.6586** | 0.5031 | 0.8552 | 0.7317 | 0.4351 | 0.4322 |

## Table 5: Feature Ablation Analysis on Model B

| Horizon | Ablation Cohort | Features Removed | Precision | Recall | F1 Score | False Alarms / Day | Actionable Recall | Median Lead Time |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=1** | Full Model B | None (all 35 features retained) | 0.5000 | 0.8500 | **0.6296** | 44.81 | 85.0% | 20.0s |
| **H=1** | No Forecast (Model A) | Removed 13 workload forecast features (indices 22..34) | 0.6739 | 0.7750 | **0.7209** | 24.90 | 77.5% | 20.0s |
| **H=1** | No Graph / Neighbour | Removed cross-service neighbour metrics (global_max_backlog, global_max_latency_ratio) | 0.7407 | 0.5000 | **0.5970** | 11.62 | 50.0% | 20.0s |
| **H=1** | No Boundary Gateway | Removed 7 boundary gateway & sync drift features (indices 11..17) | 0.5957 | 0.7000 | **0.6437** | 23.24 | 70.0% | 20.0s |
| **H=5** | Full Model B | None (all 35 features retained) | 0.4351 | 0.9050 | **0.5877** | 160.37 | 100.0% | 100.0s |
| **H=5** | No Forecast (Model A) | Removed 13 workload forecast features (indices 22..34) | 0.5107 | 0.7150 | **0.5958** | 152.73 | 100.0% | 80.0s |
| **H=5** | No Graph / Neighbour | Removed cross-service neighbour metrics (global_max_backlog, global_max_latency_ratio) | 0.5174 | 0.8900 | **0.6544** | 135.55 | 100.0% | 100.0s |
| **H=5** | No Boundary Gateway | Removed 7 boundary gateway & sync drift features (indices 11..17) | 0.4146 | 0.9350 | **0.5745** | 171.82 | 100.0% | 100.0s |
| **H=15** | Full Model B | None (all 35 features retained) | 0.4230 | 0.9456 | **0.5845** | 395.04 | 100.0% | 260.0s |
| **H=15** | No Forecast (Model A) | Removed 13 workload forecast features (indices 22..34) | 0.3516 | 0.9668 | **0.5157** | 502.78 | 100.0% | 280.0s |
| **H=15** | No Graph / Neighbour | Removed cross-service neighbour metrics (global_max_backlog, global_max_latency_ratio) | 0.3962 | 0.9456 | **0.5584** | 414.63 | 100.0% | 280.0s |
| **H=15** | No Boundary Gateway | Removed 7 boundary gateway & sync drift features (indices 11..17) | 0.3585 | 0.9758 | **0.5244** | 434.21 | 100.0% | 300.0s |

## Table 6: Robustness Stress Matrix (Model B vs Training-Fitted Z-Score Baseline, H=5)

| Stress Condition | Description & Perturbation Details | Sample Count (Prevalence) | Model B F1 | Model B PR-AUC | Z-Score Baseline F1 | Telemetry & Observability Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **clean_reference** | Standard clean test cohort (standard_test, 25 incidents at H=5) | 2750 (7.27%) | **0.5877** | 0.8152 | 0.0000 | Clean held-out baseline performance. |
| **causal_missing_telemetry_pipeline** | 15% raw telemetry dropouts simulated with manual carry-forward and refreshed timestamps (preprocessing sensitivity experiment) | 2750 (7.27%) | **0.5922** | 0.8145 | 0.0000 | Preprocessing sensitivity experiment; does not test inference staleness rejection. |
| **causal_incomplete_telemetry_alignment_within_60s** | 15% raw telemetry dropped (surviving timestamps preserved), aligned via production inference path (gaps <= 60s) | 2750 (7.27%) | **0.5844** | 0.7868 | 0.0000 | Verified causal carry-forward with OBSERVATIONS_CARRIED_FORWARD warnings (99.6% carry-forward rate). |
| **causal_incomplete_telemetry_staleness_exceeded_gt_60s** | 80s telemetry gap (> 60s limit) evaluated via production inference path | 2625 | **N/A (100% rejected)** | N/A | N/A | Strict production staleness limit enforcement: 100% rejected (2625/2625) as unavailable, zero nominal imputation. |
| **feature_zeroing_sensitivity** | 20% artificial feature zeroing sensitivity experiment | 2750 (7.27%) | **0.5286** | 0.6072 | N/A | Sensitivity to zeroing random tabular feature cells (uncalibrated drop). |
| **feature_space_noise_sensitivity** | Artificial Gaussian noise (+/- 5% sigma) applied to continuous telemetry feature space | 2750 (7.27%) | **0.4444** | 0.6126 | 0.0621 | Sensitivity to continuous feature jitter; discrete capacity flags preserved. |
| **capacity_loss_test** | Unseen fault-intensity split with physical server/thread pool capacity loss | 2750 (13.64%) | **0.5714** | 0.6437 | 0.0000 | Synthetic physical capacity collapse triggers high queue backpressure. |
| **boundary_slowdown_test** | Unseen drift stress split with boundary transit latency & sync drift degradation | 2750 (9.09%) | **0.6746** | 0.6402 | 0.0000 | Synthetic cross-system transit latency and sync drift divergence. |
| **surge_test** | Unseen workload surge split with abrupt 2.5x traffic bursts | 2750 (22.36%) | **0.6102** | 0.6506 | 0.0000 | Synthetic 2.5x traffic bursts trigger early workload alerts. |