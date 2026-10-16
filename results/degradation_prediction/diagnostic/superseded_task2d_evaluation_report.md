# BACCP Degradation Prediction Task 2D Final Evaluation Report

- **Contract Version:** `baccp-degradation-contract-v1.3.0`
- **Feature Schema Version:** `baccp-degradation-features-v1.2.0`
- **Model Weights SHA-256:** `3dc19a1e330eda826ec1dd8cb70a1a840777033d406528460fbeb8112ca787d5`
- **Evaluation Timestamp:** `2026-10-14 07:40:38 UTC`

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
| H= 1 | `causal_rollout` |  15.8% |  37.5% |  22.2% |   7.2% | 0.0526 |   5.3% | 3/8 (37.5%) | 20.0s | 0.0% | 86.62 |
| H= 1 | `model_a_telemetry_only` |  37.5% |  75.0% |  50.0% |  81.0% | 0.0107 |   4.2% | 6/8 (75.0%) | 20.0s | 0.0% | 108.27 |
| H= 1 | `model_b_telemetry_forecast` |  54.5% |  75.0% |  63.2% |  70.1% | 0.0110 |   3.5% | 6/8 (75.0%) | 20.0s | 0.0% | 54.14 |
| H= 1 | `model_b_system_calibrated` |  54.5% |  75.0% |  63.2% |  44.4% | 0.0146 |   1.5% | 6/8 (75.0%) | 20.0s | 0.0% | 54.14 |
| H= 1 | `model_c_temporal_comparator` |  70.0% |  87.5% |  77.8% |  71.7% | 0.0126 |   4.2% | 7/8 (87.5%) | 20.0s | 0.0% | 32.48 |
| H= 5 | `persistence` |   0.0% |   0.0% |   0.0% |  12.1% | 0.1692 |  16.9% | 0/8 (0.0%) | None | 0.0% | 104.41 |
| H= 5 | `static_thresholding` |   5.9% |   2.5% |   3.5% |  11.9% | 0.1662 |  16.6% | 1/8 (12.5%) | 20.0s | 0.0% | 104.41 |
| H= 5 | `simple_anomaly` |  20.0% |  10.0% |  13.3% |  12.9% | 0.1571 |  15.7% | 4/8 (50.0%) | 20.0s | 0.0% | 104.41 |
| H= 5 | `causal_rollout` |  20.0% |  10.0% |  13.3% |  12.9% | 0.1571 |  15.7% | 4/8 (50.0%) | 20.0s | 0.0% | 104.41 |
| H= 5 | `model_a_telemetry_only` |  17.4% |  57.5% |  26.7% |  40.1% | 0.1062 |  13.9% | 8/8 (100.0%) | 70.0s | 87.5% | 522.05 |
| H= 5 | `model_b_telemetry_forecast` |  17.0% |  65.0% |  26.9% |  44.0% | 0.1088 |  15.7% | 8/8 (100.0%) | 80.0s | 87.5% | 587.31 |
| H= 5 | `model_b_system_calibrated` |  17.0% |  65.0% |  26.9% |  12.1% | 0.1165 |  10.1% | 8/8 (100.0%) | 80.0s | 87.5% | 587.31 |
| H= 5 | `model_c_temporal_comparator` |  17.5% | 100.0% |  29.8% |  69.4% | 0.0993 |  19.8% | 8/8 (100.0%) | 100.0s | 100.0% | 665.62 |
| H=15 | `persistence` |   0.0% |   0.0% |   0.0% |  24.2% | 0.3175 |  31.8% | 0/8 (0.0%) | None | 0.0% | 163.79 |
| H=15 | `static_thresholding` |   5.9% |   2.0% |   2.9% |  23.8% | 0.3128 |  31.3% | 1/8 (12.5%) | 20.0s | 0.0% | 163.79 |
| H=15 | `simple_anomaly` |  20.0% |   7.8% |  11.3% |  23.8% | 0.2986 |  29.9% | 4/8 (50.0%) | 20.0s | 0.0% | 163.79 |
| H=15 | `causal_rollout` |  20.0% |   7.8% |  11.3% |  23.8% | 0.2986 |  29.9% | 4/8 (50.0%) | 20.0s | 0.0% | 163.79 |
| H=15 | `model_a_telemetry_only` |  24.5% | 100.0% |  39.4% |  34.8% | 0.2929 |  34.8% | 8/8 (100.0%) | 120.0s | 100.0% | 655.17 |
| H=15 | `model_b_telemetry_forecast` |  24.3% |  98.0% |  38.9% |  34.1% | 0.2966 |  34.2% | 8/8 (100.0%) | 120.0s | 100.0% | 655.17 |
| H=15 | `model_b_system_calibrated` |  24.3% |  98.0% |  38.9% |  24.3% | 0.2325 |  22.4% | 8/8 (100.0%) | 120.0s | 100.0% | 655.17 |
| H=15 | `model_c_temporal_comparator` |  24.3% | 100.0% |  39.1% |  44.9% | 0.3416 |  44.0% | 8/8 (100.0%) | 120.0s | 100.0% | 655.17 |
