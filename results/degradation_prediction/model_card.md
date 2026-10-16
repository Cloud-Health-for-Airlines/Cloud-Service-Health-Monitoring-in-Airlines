# BACCP Degradation Prediction Model Card

## Model Details
- **Architecture**: Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`) with Isotonic Probability Calibration.
- **Primary Checkpoint**: Model B (Telemetry + Upstream Workload Forecasts).
- **Horizons**: $H=1$ (20s), $H=5$ (100s), $H=15$ (300s).
- **Feature Dimensionality**: 35 versioned features (`baccp-degradation-features-v1.4.0`).
- **Training Provenance**: Fitted on 120 synthetic whole incidents (master seed 142857) with causal grid alignment.

## Performance Summary (Clean Holdout: standard_test)
| Horizon | Sample Count | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier Score | ECE (10-bin) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=1 (20s)** | 3250 | 0.5000 | 0.8500 | 0.6296 | 0.6280 | 0.9905 | 0.0061 | 0.0101 |
| **H=5 (100s)** | 2750 | 0.4351 | 0.9050 | 0.5877 | 0.8152 | 0.9695 | 0.0323 | 0.0534 |
| **H=15 (300s)** | 1500 | 0.4230 | 0.9456 | 0.5845 | 0.8660 | 0.9411 | 0.1060 | 0.1515 |

## Intended Use & Deployment Scope
- **Target Application**: Predictive early warning and advisory alerting for cloud airline microservices (`reservations`, `crew`, `baggage`, `boundary-gateway`).
- **Out of Scope**: Unattended autonomous actuation on legacy-core (which lacks internal instrumentation).
- **Caveat**: All holdout evaluations reflect synthetic trajectory datasets; real-world production performance requires verification against live OpenTelemetry telemetry.
