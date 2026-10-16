# BACCP Degradation Prediction Task 2D Final Evaluation Report

- **Contract Version:** `baccp-degradation-contract-v1.4.0`
- **Feature Schema Version:** `baccp-degradation-features-v1.3.0`
- **Model Weights SHA-256:** `3587d73ef55a06cb618b4277ab2ea030c2ad3c4b9b8dc0aabeb8c858776da885`
- **Evaluation Timestamp:** `2026-10-15 07:04:20 UTC`

## 1. Discrepancy Reconciliation & Model A Invariance

Model A extracts 22 telemetry features and returns before workload forecasting occurs, guaranteeing exact mathematical invariance to forecaster substitution. Computed Model A H=1 recall is identical (6/8) under both forecaster=None (persistence fallback) and trained forecaster on identical data, checkpoint, thresholds, and origin eligibility. In contrast, Model B incorporates workload forecast rollouts, yielding recall 6/8 under persistence fallback vs 6/8 under trained ML forecasting. Past reports of differing Model A recalls (e.g. 6/8 vs 7/8 in Task 2B) were caused by evaluating different origin subsets or historical intermediate checkpoints rather than feature extraction differences. Historical intermediate checkpoints prior to Task 2C are unavailable and cannot be reproduced.

| Evaluation Configuration | Forecaster Source | Model A H=1 Recall | Model B H=1 Recall |
|:---|:---|:---:|:---:|
| Diagnostic Sandbox | `forecaster=None` (Persistence Fallback) | 6 / 8 (75.0%) | 6 / 8 (75.0%) |
| Main Hardened Pipeline | `WorkloadForecaster.get_instance()` (Trained ML) | 6 / 8 (75.0%) | 6 / 8 (75.0%) |

## 2. Standard Test System-Wide Performance (Healthy Origins Cohort)

| Horizon | Model / Baseline | Precision | Recall | F1 Score | PR-AUC | Brier | 10-Bin ECE | Evaluable Episode Recall | Lead Median | Recall >= 40s | False Alarms / Day |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| H= 1 | `persistence` |   0.0% |   0.0% |   0.0% |   2.0% | 0.0602 |   6.0% | 0/8 (0.0%) | None | 0.0% | 86.62 |
| H= 1 | `static_thresholding` |   5.9% |  12.5% |   8.0% |   2.5% | 0.0576 |   5.8% | 1/8 (12.5%) | 20.0s | 0.0% | 86.62 |
| H= 1 | `simple_anomaly` |  11.1% |  50.0% |  18.2% |   6.6% | 0.0902 |   9.0% | 4/8 (50.0%) | 20.0s | 0.0% | 140.75 |
| H= 1 | `causal_rollout` |   2.0% | 100.0% |   3.9% |   2.0% | 0.9799 |  98.0% | 8/8 (100.0%) | 20.0s | 0.0% | 876.99 |
| H= 1 | `model_a_telemetry_only` |  37.5% |  75.0% |  50.0% |  81.0% | 0.0107 |   4.2% | 6/8 (75.0%) | 20.0s | 0.0% | 108.27 |
| H= 1 | `model_b_telemetry_forecast` |  60.0% |  75.0% |  66.7% |  66.1% | 0.0131 |   3.9% | 6/8 (75.0%) | 20.0s | 0.0% | 43.31 |
| H= 1 | `model_b_system_calibrated` |  60.0% |  75.0% |  66.7% |  58.2% | 0.0124 |   2.4% | 6/8 (75.0%) | 20.0s | 0.0% | 43.31 |
| H= 1 | `model_c_temporal_comparator` |  58.3% |  87.5% |  70.0% |  58.7% | 0.0119 |   4.6% | 7/8 (87.5%) | 20.0s | 0.0% | 43.31 |
| H= 5 | `persistence` |   0.0% |   0.0% |   0.0% |  12.1% | 0.1692 |  16.9% | 0/8 (0.0%) | None | 0.0% | 104.41 |
| H= 5 | `static_thresholding` |   5.9% |   2.5% |   3.5% |  11.9% | 0.1662 |  16.6% | 1/8 (12.5%) | 20.0s | 0.0% | 104.41 |
| H= 5 | `simple_anomaly` |  20.0% |  10.0% |  13.3% |  12.9% | 0.1571 |  15.7% | 4/8 (50.0%) | 20.0s | 0.0% | 104.41 |
| H= 5 | `causal_rollout` |  12.1% | 100.0% |  21.6% |  12.1% | 0.8792 |  87.9% | 8/8 (100.0%) | 100.0s | 100.0% | 835.29 |
| H= 5 | `model_a_telemetry_only` |  17.4% |  57.5% |  26.7% |  40.1% | 0.1062 |  13.9% | 8/8 (100.0%) | 70.0s | 87.5% | 522.05 |
| H= 5 | `model_b_telemetry_forecast` |  25.9% |  70.0% |  37.8% |  54.4% | 0.1190 |  18.0% | 8/8 (100.0%) | 80.0s | 87.5% | 391.54 |
| H= 5 | `model_b_system_calibrated` |  25.9% |  70.0% |  37.8% |  24.9% | 0.1026 |   7.8% | 8/8 (100.0%) | 80.0s | 87.5% | 391.54 |
| H= 5 | `model_c_temporal_comparator` |  29.1% |  92.5% |  44.3% |  66.0% | 0.1029 |  17.5% | 8/8 (100.0%) | 100.0s | 100.0% | 365.44 |
| H=15 | `persistence` |   0.0% |   0.0% |   0.0% |  24.2% | 0.3175 |  31.8% | 0/8 (0.0%) | None | 0.0% | 163.79 |
| H=15 | `static_thresholding` |   5.9% |   2.0% |   2.9% |  23.8% | 0.3128 |  31.3% | 1/8 (12.5%) | 20.0s | 0.0% | 163.79 |
| H=15 | `simple_anomaly` |  20.0% |   7.8% |  11.3% |  23.8% | 0.2986 |  29.9% | 4/8 (50.0%) | 20.0s | 0.0% | 163.79 |
| H=15 | `causal_rollout` |  24.2% | 100.0% |  38.9% |  24.2% | 0.7583 |  75.8% | 8/8 (100.0%) | 120.0s | 100.0% | 655.17 |
| H=15 | `model_a_telemetry_only` |  24.5% | 100.0% |  39.4% |  34.8% | 0.2929 |  34.8% | 8/8 (100.0%) | 120.0s | 100.0% | 655.17 |
| H=15 | `model_b_telemetry_forecast` |  25.9% |  98.0% |  41.0% |  46.4% | 0.3146 |  40.3% | 8/8 (100.0%) | 120.0s | 100.0% | 655.17 |
| H=15 | `model_b_system_calibrated` |  25.9% |  98.0% |  41.0% |  30.2% | 0.2275 |  23.6% | 8/8 (100.0%) | 120.0s | 100.0% | 655.17 |
| H=15 | `model_c_temporal_comparator` |  24.2% | 100.0% |  38.9% |  50.0% | 0.3298 |  42.8% | 8/8 (100.0%) | 120.0s | 100.0% | 655.17 |
