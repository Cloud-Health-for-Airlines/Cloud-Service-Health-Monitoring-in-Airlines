# TASK 2 — Workload-Aware Service and Boundary Degradation Prediction Evaluation Report

- **Contract Version**: `baccp-degradation-contract-v1.0.0`
- **Dataset SHA-256**: `4940aff3e1f2f0b4d0940c5a3da2851b658605ff57eadf37ddf1cf1e46c2d5eb`
- **Evaluation Timestamp**: `2026-10-13T17:26:43Z`
- **Horizons Evaluated**: `H=1 (20s)`, `H=5 (100s)`, `H=15 (300s)`

---

## 1. Continuous Target Evaluation (Latency, Backlog, Boundary Sync Drift)

### Split: `standard_test`

| Horizon | Target | Persistence MAE (RMSE) | Causal Rollout MAE (RMSE) | Model A (Telemetry Only) MAE (RMSE) | Model B (Telemetry + Forecast) MAE (RMSE) [95% CI] |
|---|---|---|---|---|---|
| H=1 (20s) | `latency_ms` | 74.27 (270.09) | 1311.49 (2013.50) | 41.00 (101.35) | **44.77** (107.13) [39.31, 51.65] |
| H=1 (20s) | `backlog_requests` | 4.61 (27.24) | 50.16 (146.20) | 4.59 (15.04) | **4.58** (15.10) [3.77, 5.75] |
| H=1 (20s) | `boundary_sync_drift_pct` | 2.58 (7.10) | 2.58 (7.10) | 2.89 (5.24) | **3.07** (5.62) [2.80, 3.38] |
| H=5 (100s) | `latency_ms` | 400.89 (1243.63) | 4373.55 (4675.60) | 359.18 (704.65) | **356.11** (765.97) [319.13, 391.65] |
| H=5 (100s) | `backlog_requests` | 27.09 (96.91) | 55.64 (153.11) | 21.88 (43.26) | **22.09** (43.73) [19.70, 24.48] |
| H=5 (100s) | `boundary_sync_drift_pct` | 10.41 (20.07) | 10.41 (20.07) | 12.44 (16.63) | **12.46** (16.68) [11.05, 13.76] |
| H=15 (300s) | `latency_ms` | 1198.03 (2822.29) | 13252.25 (13979.54) | 943.43 (2056.77) | **927.70** (2082.12) [805.38, 1060.47] |
| H=15 (300s) | `backlog_requests` | 80.64 (179.71) | 75.88 (175.42) | 41.48 (84.11) | **41.54** (84.44) [37.25, 46.11] |
| H=15 (300s) | `boundary_sync_drift_pct` | 30.82 (45.99) | 30.82 (45.99) | 25.04 (31.27) | **24.71** (30.94) [21.77, 27.38] |

### Split: `capacity_loss_test`

| Horizon | Target | Persistence MAE (RMSE) | Causal Rollout MAE (RMSE) | Model A (Telemetry Only) MAE (RMSE) | Model B (Telemetry + Forecast) MAE (RMSE) [95% CI] |
|---|---|---|---|---|---|
| H=1 (20s) | `latency_ms` | 305.55 (680.35) | 1873.02 (3193.74) | 133.12 (353.12) | **140.16** (356.59) [131.84, 148.65] |
| H=1 (20s) | `backlog_requests` | 13.99 (51.89) | 67.20 (164.83) | 13.61 (42.13) | **13.37** (40.77) [12.68, 14.17] |
| H=1 (20s) | `boundary_sync_drift_pct` | 6.45 (12.15) | 6.45 (12.15) | 5.14 (9.58) | **5.17** (9.78) [4.88, 5.46] |
| H=5 (100s) | `latency_ms` | 1576.34 (3018.52) | 3604.37 (3809.81) | 738.56 (1617.99) | **744.25** (1656.84) [708.63, 785.30] |
| H=5 (100s) | `backlog_requests` | 79.26 (168.88) | 79.42 (179.19) | 46.28 (108.39) | **46.63** (107.62) [44.73, 48.75] |
| H=5 (100s) | `boundary_sync_drift_pct` | 16.82 (25.82) | 16.82 (25.82) | 12.78 (18.78) | **12.67** (18.53) [12.03, 13.28] |
| H=15 (300s) | `latency_ms` | 2331.45 (4336.03) | 10077.74 (10609.86) | 1581.52 (3048.04) | **1566.53** (3044.29) [1421.11, 1710.82] |
| H=15 (300s) | `backlog_requests` | 102.42 (200.83) | 61.32 (160.61) | 63.34 (132.62) | **63.41** (132.97) [55.39, 71.54] |
| H=15 (300s) | `boundary_sync_drift_pct` | 37.06 (43.43) | 37.06 (43.43) | 17.95 (21.96) | **18.12** (21.97) [16.82, 19.38] |

### Split: `boundary_slowdown_test`

| Horizon | Target | Persistence MAE (RMSE) | Causal Rollout MAE (RMSE) | Model A (Telemetry Only) MAE (RMSE) | Model B (Telemetry + Forecast) MAE (RMSE) [95% CI] |
|---|---|---|---|---|---|
| H=1 (20s) | `latency_ms` | 657.94 (2222.23) | 4018.87 (10283.78) | 359.50 (1521.37) | **361.85** (1521.66) [344.74, 379.03] |
| H=1 (20s) | `backlog_requests` | 7.69 (49.53) | 43.89 (144.28) | 7.85 (44.90) | **7.68** (44.84) [7.50, 7.86] |
| H=1 (20s) | `boundary_sync_drift_pct` | 5.84 (18.47) | 5.84 (18.47) | 5.51 (17.73) | **5.55** (17.72) [5.35, 5.71] |
| H=5 (100s) | `latency_ms` | 3447.49 (9206.84) | 6990.17 (10614.72) | 1956.03 (6386.75) | **1895.79** (6265.31) [1812.49, 1983.20] |
| H=5 (100s) | `backlog_requests` | 44.67 (140.90) | 51.87 (156.85) | 34.17 (100.43) | **34.08** (98.74) [32.60, 35.57] |
| H=5 (100s) | `boundary_sync_drift_pct` | 29.42 (45.75) | 29.42 (45.75) | 22.55 (37.15) | **22.69** (37.43) [22.08, 23.36] |
| H=15 (300s) | `latency_ms` | 5001.07 (12592.44) | 13354.42 (14218.60) | 2804.41 (8677.70) | **2801.20** (8671.52) [2280.08, 3345.17] |
| H=15 (300s) | `backlog_requests` | 70.67 (182.26) | 30.00 (113.49) | 32.00 (86.48) | **32.15** (86.27) [27.56, 36.92] |
| H=15 (300s) | `boundary_sync_drift_pct` | 61.67 (65.34) | 61.67 (65.34) | 29.40 (38.65) | **29.54** (38.70) [25.87, 33.06] |

### Split: `surge_test`

| Horizon | Target | Persistence MAE (RMSE) | Causal Rollout MAE (RMSE) | Model A (Telemetry Only) MAE (RMSE) | Model B (Telemetry + Forecast) MAE (RMSE) [95% CI] |
|---|---|---|---|---|---|
| H=1 (20s) | `latency_ms` | 209.02 (488.69) | 1462.83 (2250.30) | 118.82 (313.98) | **121.34** (310.66) [111.99, 129.92] |
| H=1 (20s) | `backlog_requests` | 22.33 (68.30) | 61.46 (159.00) | 23.56 (63.95) | **23.31** (63.34) [21.08, 25.47] |
| H=1 (20s) | `boundary_sync_drift_pct` | 8.39 (17.89) | 8.39 (17.89) | 7.39 (16.81) | **7.35** (16.70) [7.06, 7.74] |
| H=5 (100s) | `latency_ms` | 990.59 (2056.05) | 4903.20 (5459.09) | 661.74 (1309.78) | **662.24** (1326.18) [624.99, 701.17] |
| H=5 (100s) | `backlog_requests` | 75.22 (162.34) | 72.63 (172.85) | 63.96 (125.68) | **64.05** (125.05) [59.35, 68.81] |
| H=5 (100s) | `boundary_sync_drift_pct` | 28.41 (41.28) | 28.41 (41.28) | 22.99 (34.17) | **23.00** (34.04) [22.13, 23.82] |
| H=15 (300s) | `latency_ms` | 1516.90 (3067.71) | 15306.13 (16980.68) | 855.48 (1719.59) | **856.55** (1753.35) [743.06, 978.41] |
| H=15 (300s) | `backlog_requests` | 97.71 (199.21) | 50.09 (137.66) | 51.74 (110.25) | **51.79** (111.40) [42.70, 61.64] |
| H=15 (300s) | `boundary_sync_drift_pct` | 53.20 (56.58) | 53.20 (56.58) | 22.87 (30.15) | **23.29** (30.52) [20.87, 25.79] |

---

## 2. Forward-Window Degradation Event Prediction (Prevalence, Precision, Recall, F1, PR-AUC, Brier)

### Split: `standard_test`

| Horizon | Model | Prevalence | Precision | Recall | F1 Score [95% CI] | PR-AUC | Brier Score | False Alarms (Misses) |
|---|---|---|---|---|---|---|---|---|
| H=1 (20s) | Persistence | 22.5% | 98.6% | 94.5% | **96.5%** - | 94.4% | 0.0154 | 10 (40) |
| H=1 (20s) | Causal Rollout | 22.5% | 22.5% | 100.0% | **36.7%** - | 27.0% | 0.7754 | 2520 (0) |
| H=1 (20s) | Model A (Telemetry Only) | 22.5% | 97.7% | 99.0% | **98.4%** - | 99.9% | 0.0068 | 17 (7) |
| H=1 (20s) | Model B (Telemetry + Forecast) | 22.5% | 99.9% | 96.3% | **98.0%** [97.4, 98.9] | 99.8% | 0.0067 | 1 (27) |
| H=5 (100s) | Persistence | 26.6% | 98.2% | 72.6% | **83.5%** - | 78.5% | 0.0764 | 10 (200) |
| H=5 (100s) | Causal Rollout | 26.6% | 26.6% | 100.0% | **42.0%** - | 26.6% | 0.7345 | 2020 (0) |
| H=5 (100s) | Model A (Telemetry Only) | 26.6% | 97.1% | 91.5% | **94.2%** - | 96.8% | 0.0333 | 20 (62) |
| H=5 (100s) | Model B (Telemetry + Forecast) | 26.6% | 97.6% | 90.1% | **93.7%** [92.9, 95.0] | 97.0% | 0.0343 | 16 (72) |
| H=15 (300s) | Persistence | 35.7% | 100.0% | 38.7% | **55.8%** - | 60.6% | 0.2187 | 0 (328) |
| H=15 (300s) | Causal Rollout | 35.7% | 35.7% | 100.0% | **52.6%** - | 35.7% | 0.6433 | 965 (0) |
| H=15 (300s) | Model A (Telemetry Only) | 35.7% | 51.5% | 95.5% | **66.9%** - | 89.8% | 0.1260 | 481 (24) |
| H=15 (300s) | Model B (Telemetry + Forecast) | 35.7% | 51.2% | 95.3% | **66.6%** [49.3, 81.5] | 90.0% | 0.1241 | 486 (25) |

### Split: `capacity_loss_test`

| Horizon | Model | Prevalence | Precision | Recall | F1 Score [95% CI] | PR-AUC | Brier Score | False Alarms (Misses) |
|---|---|---|---|---|---|---|---|---|
| H=1 (20s) | Persistence | 44.5% | 99.8% | 94.8% | **97.2%** - | 96.9% | 0.0240 | 3 (75) |
| H=1 (20s) | Causal Rollout | 44.5% | 44.5% | 100.0% | **61.5%** - | 56.5% | 0.5554 | 1805 (0) |
| H=1 (20s) | Model A (Telemetry Only) | 44.5% | 99.9% | 95.6% | **97.7%** - | 99.1% | 0.0199 | 2 (64) |
| H=1 (20s) | Model B (Telemetry + Forecast) | 44.5% | 99.9% | 94.9% | **97.3%** [97.2, 97.5] | 99.0% | 0.0202 | 1 (74) |
| H=5 (100s) | Persistence | 52.7% | 100.0% | 74.1% | **85.1%** - | 87.8% | 0.1364 | 0 (375) |
| H=5 (100s) | Causal Rollout | 52.7% | 52.7% | 100.0% | **69.0%** - | 52.7% | 0.4731 | 1301 (0) |
| H=5 (100s) | Model A (Telemetry Only) | 52.7% | 100.0% | 75.6% | **86.1%** - | 97.6% | 0.0876 | 0 (354) |
| H=5 (100s) | Model B (Telemetry + Forecast) | 52.7% | 100.0% | 75.2% | **85.9%** [85.2, 86.4] | 97.8% | 0.0882 | 0 (359) |
| H=15 (300s) | Persistence | 60.0% | 100.0% | 36.0% | **52.9%** - | 74.4% | 0.3840 | 0 (576) |
| H=15 (300s) | Causal Rollout | 60.0% | 60.0% | 100.0% | **75.0%** - | 60.0% | 0.4000 | 600 (0) |
| H=15 (300s) | Model A (Telemetry Only) | 60.0% | 88.9% | 91.1% | **90.0%** - | 94.9% | 0.1402 | 102 (80) |
| H=15 (300s) | Model B (Telemetry + Forecast) | 60.0% | 89.6% | 90.8% | **90.2%** [88.3, 91.9] | 95.5% | 0.1363 | 95 (83) |

### Split: `boundary_slowdown_test`

| Horizon | Model | Prevalence | Precision | Recall | F1 Score [95% CI] | PR-AUC | Brier Score | False Alarms (Misses) |
|---|---|---|---|---|---|---|---|---|
| H=1 (20s) | Persistence | 30.6% | 99.8% | 95.0% | **97.3%** - | 96.3% | 0.0160 | 2 (50) |
| H=1 (20s) | Causal Rollout | 30.6% | 30.6% | 100.0% | **46.9%** - | 35.8% | 0.6935 | 2254 (0) |
| H=1 (20s) | Model A (Telemetry Only) | 30.6% | 100.0% | 95.0% | **97.4%** - | 98.7% | 0.0149 | 0 (50) |
| H=1 (20s) | Model B (Telemetry + Forecast) | 30.6% | 100.0% | 94.9% | **97.4%** [97.3, 97.5] | 98.5% | 0.0149 | 0 (51) |
| H=5 (100s) | Persistence | 36.3% | 100.0% | 75.0% | **85.7%** - | 84.0% | 0.0909 | 0 (250) |
| H=5 (100s) | Causal Rollout | 36.3% | 36.3% | 100.0% | **53.3%** - | 36.3% | 0.6371 | 1752 (0) |
| H=5 (100s) | Model A (Telemetry Only) | 36.3% | 100.0% | 75.2% | **85.9%** - | 97.6% | 0.0572 | 0 (247) |
| H=5 (100s) | Model B (Telemetry + Forecast) | 36.3% | 99.9% | 75.2% | **85.8%** [85.1, 86.3] | 98.1% | 0.0560 | 1 (248) |
| H=15 (300s) | Persistence | 40.0% | 100.0% | 41.3% | **58.5%** - | 64.8% | 0.2347 | 0 (352) |
| H=15 (300s) | Causal Rollout | 40.0% | 40.0% | 100.0% | **57.1%** - | 40.0% | 0.6000 | 900 (0) |
| H=15 (300s) | Model A (Telemetry Only) | 40.0% | 75.0% | 95.8% | **84.1%** - | 92.7% | 0.1040 | 192 (25) |
| H=15 (300s) | Model B (Telemetry + Forecast) | 40.0% | 75.2% | 95.7% | **84.2%** [81.5, 86.6] | 92.8% | 0.0994 | 189 (26) |

### Split: `surge_test`

| Horizon | Model | Prevalence | Precision | Recall | F1 Score [95% CI] | PR-AUC | Brier Score | False Alarms (Misses) |
|---|---|---|---|---|---|---|---|---|
| H=1 (20s) | Persistence | 56.9% | 95.8% | 92.6% | **94.2%** - | 93.0% | 0.0649 | 74 (137) |
| H=1 (20s) | Causal Rollout | 56.9% | 56.9% | 100.0% | **72.5%** - | 62.1% | 0.4314 | 1402 (0) |
| H=1 (20s) | Model A (Telemetry Only) | 56.9% | 96.6% | 93.2% | **94.9%** - | 98.7% | 0.0398 | 60 (125) |
| H=1 (20s) | Model B (Telemetry + Forecast) | 56.9% | 99.9% | 91.6% | **95.6%** [95.1, 96.0] | 98.7% | 0.0394 | 2 (155) |
| H=5 (100s) | Persistence | 75.3% | 97.5% | 68.8% | **80.7%** - | 90.6% | 0.2484 | 37 (646) |
| H=5 (100s) | Causal Rollout | 75.3% | 75.3% | 100.0% | **85.9%** - | 75.3% | 0.2465 | 678 (0) |
| H=5 (100s) | Model A (Telemetry Only) | 75.3% | 96.5% | 77.7% | **86.1%** - | 96.4% | 0.1434 | 58 (462) |
| H=5 (100s) | Model B (Telemetry + Forecast) | 75.3% | 96.5% | 77.2% | **85.8%** [84.2, 87.1] | 96.3% | 0.1451 | 58 (473) |
| H=15 (300s) | Persistence | 98.1% | 99.8% | 28.1% | **43.8%** - | 98.5% | 0.7060 | 1 (1058) |
| H=15 (300s) | Causal Rollout | 98.1% | 98.1% | 100.0% | **99.0%** - | 98.1% | 0.0193 | 29 (0) |
| H=15 (300s) | Model A (Telemetry Only) | 98.1% | 98.7% | 75.9% | **85.8%** - | 98.7% | 0.2333 | 15 (354) |
| H=15 (300s) | Model B (Telemetry + Forecast) | 98.1% | 98.8% | 76.8% | **86.4%** [85.0, 87.8] | 98.8% | 0.2357 | 14 (341) |

---

## 3. Early Warning Lead Time Analysis

Evaluated strictly on true positive alerts issued **BEFORE** degradation onset (t_alert < t_onset):

| Split | Horizon | Model | Early Warning Coverage | Mean Lead Time (sec) | Detection After Onset Count |
|---|---|---|---|---|---|
| `standard_test` | H=1 (20s) | Model A (Telemetry Only) | 53.9% (7/13) | **22.9s** (±7.0s) | 252 |
| `standard_test` | H=1 (20s) | Model B (Telemetry + Forecast) | 30.8% (4/13) | **20.0s** (±0.0s) | 252 |
| `standard_test` | H=5 (100s) | Model A (Telemetry Only) | 61.5% (8/13) | **52.5s** (±19.8s) | 220 |
| `standard_test` | H=5 (100s) | Model B (Telemetry + Forecast) | 61.5% (8/13) | **52.5s** (±28.2s) | 220 |
| `standard_test` | H=15 (300s) | Model A (Telemetry Only) | 61.5% (8/13) | **125.0s** (±8.7s) | 105 |
| `standard_test` | H=15 (300s) | Model B (Telemetry + Forecast) | 61.5% (8/13) | **125.0s** (±8.7s) | 105 |
| `capacity_loss_test` | H=1 (20s) | Model A (Telemetry Only) | 4.0% (1/25) | **20.0s** (±0.0s) | 463 |
| `capacity_loss_test` | H=1 (20s) | Model B (Telemetry + Forecast) | 4.0% (1/25) | **20.0s** (±0.0s) | 463 |
| `capacity_loss_test` | H=5 (100s) | Model A (Telemetry Only) | 24.0% (6/25) | **76.7s** (±24.3s) | 363 |
| `capacity_loss_test` | H=5 (100s) | Model B (Telemetry + Forecast) | 8.0% (2/25) | **100.0s** (±0.0s) | 363 |
| `capacity_loss_test` | H=15 (300s) | Model A (Telemetry Only) | 100.0% (25/25) | **149.6s** (±34.5s) | 113 |
| `capacity_loss_test` | H=15 (300s) | Model B (Telemetry + Forecast) | 100.0% (25/25) | **148.8s** (±34.5s) | 113 |
| `boundary_slowdown_test` | H=1 (20s) | Model A (Telemetry Only) | 0.0% (0/25) | **0.0s** (±0.0s) | 474 |
| `boundary_slowdown_test` | H=1 (20s) | Model B (Telemetry + Forecast) | 0.0% (0/25) | **0.0s** (±0.0s) | 474 |
| `boundary_slowdown_test` | H=5 (100s) | Model A (Telemetry Only) | 12.0% (3/25) | **93.3s** (±9.4s) | 374 |
| `boundary_slowdown_test` | H=5 (100s) | Model B (Telemetry + Forecast) | 8.0% (2/25) | **140.0s** (±40.0s) | 374 |
| `boundary_slowdown_test` | H=15 (300s) | Model A (Telemetry Only) | 100.0% (25/25) | **139.2s** (±38.6s) | 124 |
| `boundary_slowdown_test` | H=15 (300s) | Model B (Telemetry + Forecast) | 100.0% (25/25) | **140.0s** (±38.8s) | 124 |
| `surge_test` | H=1 (20s) | Model A (Telemetry Only) | 0.0% (0/25) | **0.0s** (±0.0s) | 468 |
| `surge_test` | H=1 (20s) | Model B (Telemetry + Forecast) | 0.0% (0/25) | **0.0s** (±0.0s) | 468 |
| `surge_test` | H=5 (100s) | Model A (Telemetry Only) | 16.0% (4/25) | **140.0s** (±24.5s) | 369 |
| `surge_test` | H=5 (100s) | Model B (Telemetry + Forecast) | 4.0% (1/25) | **160.0s** (±0.0s) | 369 |
| `surge_test` | H=15 (300s) | Model A (Telemetry Only) | 100.0% (25/25) | **144.0s** (±32.5s) | 119 |
| `surge_test` | H=15 (300s) | Model B (Telemetry + Forecast) | 100.0% (25/25) | **144.0s** (±32.5s) | 119 |
