# BACCP Workload Forecasting Module

Reproducible per-service incoming workload forecasting for hybrid airline cloud systems.

```
+----------------------------------------------------------------------------------------------------+
|                                BACCP WORKLOAD FORECASTING FLOW                                     |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    Service History Sequence [y_0, ..., y_t] (Offered Workload in Requests/Second)                 |
|                                       │                                                            |
|                                       ▼                                                            |
|    Ordered 20-Feature Schema (Lags: 0..5, 9, 14; Rolling Stats; Cycle Phase sin/cos)               |
|                                       │                                                            |
|                        +--------------+--------------+                                             |
|                        │                             │                                             |
|                        ▼                             ▼                                             |
|          Multi-Horizon Forecasters             Conformal Interval Calibrator                       |
|          - H=1 (20s)                           - Finite-Sample Quantiles                           |
|          - H=5 (100s)                          - Fitted on Calibration Split                       |
|          - H=15 (300s)                         - Out-of-Distribution Shift Detection               |
|          - Auto Fallback (GB -> Ridge AR)                    │                                     |
|                        │                                     │                                     |
|                        +--------------+----------------------+                                     |
|                                       │                                                            |
|                                       ▼                                                            |
|    Forecasting Output Payload:                                                                     |
|    - Point Forecast (RPS)                                                                          |
|    - Conformal 90% Prediction Interval [lower, upper]                                              |
|    - Candidate Forecasts (GB, Ridge, Persistence, Seasonal Naive)                                  |
|    - Signed Capacity Margin (capacity - forecast)                                                  |
|    - Available Headroom: max(0, capacity_margin)                                                   |
|    - Overload: max(0, -capacity_margin)                                                            |
|    - Reliability Metadata & Distribution Shift Warnings                                            |
+----------------------------------------------------------------------------------------------------+
```

---

## Quickstart & Reproduction Commands

All commands should be executed from the repository root with `PYTHONPATH=ai-models`:

### 1. Data Generation
Generate the 250-incident synthetic dataset (with standard, diagnostic surge, and locked fresh surge splits) and split manifest:
```bash
PYTHONPATH=ai-models python3 -m workload_forecasting.generate_workload_data --seed 42
```

### 2. Model Training & Calibration
Fit Persistence, Seasonal Naive, Ridge AR, and Gradient Boosting across all 5 services and 3 horizons, track training domain bounds, and calibrate 90% conformal intervals:
```bash
PYTHONPATH=ai-models python3 -m workload_forecasting.train
```

### 3. Master Evaluation Suite
Evaluate models against Standard Held-Out Test, Diagnostic Surge, and Locked Fresh Surge splits, compute whole-incident bootstrap 95% CIs, and generate diagnostic figures:
```bash
PYTHONPATH=ai-models python3 -m workload_forecasting.evaluate
```

### 4. Automated Tests
Run the comprehensive 11-test suite:
```bash
pytest -v ai-models/tests/test_workload_forecasting.py
```

### 5. Interactive Python Inference
```python
from ai_models.workload_forecasting import get_forecaster

forecaster = get_forecaster()

# Historical offered RPS for reservations (last 20 observations, 20s intervals)
sample_history = [42.1, 43.5, 44.0, 42.8, 45.2, 46.0, 44.9, 45.8, 47.1, 46.5,
                  45.0, 44.2, 45.6, 46.8, 47.5, 48.2, 49.0, 48.5, 49.2, 50.1]

# Forecast H=5 steps (100 seconds ahead) using recommended auto_fallback policy
# Explicit timestamp_sec anchors the micro-cycle phase across variable buffer lengths
result = forecaster.predict(
    service_id="reservations",
    history=sample_history,
    horizon=5,
    model_name="auto_fallback",
    timestamp_sec=600.0,
)

print(f"Selected Model:  {result['selected_model']}")
print(f"Point Forecast:  {result['point_forecast_rps']} RPS")
print(f"90% Interval:    [{result['prediction_interval_90']['lower_rps']} - {result['prediction_interval_90']['upper_rps']}] RPS")
print(f"Capacity Margin: {result['capacity_metrics']['capacity_margin_rps']} RPS")
print(f"Headroom:        {result['capacity_metrics']['available_headroom_rps']} RPS")
print(f"Overload:        {result['capacity_metrics']['overload_rps']} RPS")
print(f"Candidate Preds: {result['candidate_forecasts']}")
print(f"Reliability:     {result['reliability_metadata']['reliability_status']}")
```

---

## Directory & Artifact Layout

- **Data Generator:** [`ai-models/workload_forecasting/generate_workload_data.py`](./generate_workload_data.py)
- **Data Contract:** [`ai-models/workload_forecasting/data_contract.py`](./data_contract.py)
- **Feature Extraction:** [`ai-models/workload_forecasting/features.py`](./features.py)
- **Model Implementations:** [`ai-models/workload_forecasting/models.py`](./models.py)
- **Training Pipeline:** [`ai-models/workload_forecasting/train.py`](./train.py)
- **Evaluation Pipeline:** [`ai-models/workload_forecasting/evaluate.py`](./evaluate.py)
- **Inference Client:** [`ai-models/workload_forecasting/inference.py`](./inference.py)
- **Model Card & Design Doc:** [`ai-models/workload_forecasting/model_card.md`](./model_card.md)
- **Generated Dataset:** `ai-models/data/workload_forecasting/workload_dataset.json`
- **Saved Checkpoints:** `ai-models/weights/workload_forecasting/workload_models_bundle.pkl`
- **Evaluation Metrics & Plots:** `results/workload_forecasting/`
