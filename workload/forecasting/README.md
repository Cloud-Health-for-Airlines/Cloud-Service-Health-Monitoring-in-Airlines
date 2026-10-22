# BACCP Airline Passenger Demand Forecasting Component

## 1. Overview & Objective

The **Airline Demand Forecasting Component** forecasts commercial airline passenger volume and flight activity across future prediction horizons, translating those forecasts into calibrated, service-specific transaction arrival rates (TPS) and resource utilization profiles for the BACCP simulation and cloud monitoring pipeline.

```
+-------------------------------------------------------------+
|    Historical Airline Demand Data                           |
|    (U.S. BTS TranStats T-100 & On-Time Performance)         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|    Chronological Train (70%) / Val (15%) / Test (15%)       |
|    (Strict Non-Shuffled Temporal Sequencing)                |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|    Forecasting Models                                       |
|    - LastValueBaseline (Persistence)                        |
|    - SeasonalNaiveBaseline (Weekly Seasonality s=7)         |
|    - LightweightAutoRegressiveForecaster (Ridge Regression) |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|    Forecast Evaluation (MAE, RMSE, sMAPE, MAPE)             |
|    & 90% Conformal Uncertainty Intervals                    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|    DemandToWorkloadMapper (Calibrated Mapping)              |
|    Passenger Demand -> Service TPS -> Google Cluster Trace  |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|    BACCP Pipeline & Cloud Orchestrator                      |
+-------------------------------------------------------------+
```

---

## 2. Dataset Provenance & Usage Conditions

- **Source:** U.S. Department of Transportation, Bureau of Transportation Statistics (BTS), Office of Airline Information, TranStats.
- **Tables:** Reporting Carrier On-Time Performance (scheduled departures) and T-100 Domestic Market (passenger enplanements).
- **License / Usage Conditions:** **Public Domain** (Work of the United States Government, 17 U.S.C. § 105). Free for public and commercial research with no access fees or credentials required.
- **Bundled Sample Path:** [`workload/datasets/sample_data/bts_daily_passenger_demand.csv`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/workload/datasets/sample_data/bts_daily_passenger_demand.csv)
- **Temporal Coverage:** 730 consecutive calendar days (2 full annual cycles from 2024-01-01 to 2025-12-31).

### Domain Distinction: Passengers vs Flights vs API Requests
This component adheres to strict conceptual boundaries:
- **Enplaned Passengers** represents human travelers boarding flights (macro passenger demand).
- **Scheduled / Actual Flights** represents physical aircraft legs operated.
- **API Transactions (TPS)** represents digital microservice calls initiated by web and mobile applications.
Passenger counts are **not** interchangeable with API requests; they are related through calibrated operational ratios (search-to-book ratios, check-in windows, and checked baggage per passenger).

---

## 3. Modeling Methodology & Chronological Splitting

### Chronological Splits (Zero Future-Data Leakage)
To prevent temporal data leakage, observations are partitioned strictly by calendar sequence:
- **Training Set (70%):** Days 1–511 (2024-01-01 to 2025-05-25)
- **Validation Set (15%):** Days 512–620 (2025-05-26 to 2025-09-11)
- **Held-Out Test Set (15%):** Days 621–730 (2025-09-12 to 2025-12-31)

### Feature Formulation
Features are constructed strictly using backward-looking past data:
1. **Autoregressive Lags:** $y(t-1), y(t-2), y(t-3), y(t-7), y(t-14)$
2. **Rolling Statistics:** 7-day and 14-day backward rolling mean (computed strictly over past values)
3. **Calendar Fourier Terms:** $\sin(2\pi \cdot \text{dow}/7)$, $\cos(2\pi \cdot \text{dow}/7)$, $\sin(2\pi \cdot (\text{month}-1)/12)$, $\cos(2\pi \cdot (\text{month}-1)/12)$
4. **Standardization:** Feature scaler fit exclusively on the training split.

---

## 4. Evaluation Results

Evaluated over the held-out 110-day test set ($H = 110$ days):

| Model | MAE (Passengers) | RMSE (Passengers) | sMAPE (%) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| `LastValueBaseline` | 293,134 | 349,147 | 13.70% | 14.15% |
| `SeasonalNaiveBaseline (s=7)` | 196,448 | 236,314 | 9.12% | 9.21% |
| **`LightweightAutoRegressiveForecaster (Ridge)`** | **117,605** | **177,399** | **5.83%** | **5.71%** |

### Benchmark Relative Improvements
- **vs Last-Value Baseline:** **59.9% MAE reduction**, **49.2% RMSE reduction**
- **vs Seasonal Naive Baseline:** **40.1% MAE reduction**, **24.9% RMSE reduction**

Artifacts are preserved in:
- Structured JSON: [`results/forecasting/demand_forecast_evaluation.json`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/results/forecasting/demand_forecast_evaluation.json)
- Summary Report: [`results/forecasting/demand_forecast_evaluation.md`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/results/forecasting/demand_forecast_evaluation.md)

---

## 5. Calibrated Demand-to-Workload Mapping

The [`DemandToWorkloadMapper`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/workload/forecasting/workload_mapper.py) converts forecasted daily passenger volume ($D_{\text{forecast}}$) into service-specific arrival rates and resource requirements:

1. **Demand Multiplier:**
   $$M = \frac{D_{\text{forecast}}}{D_{\text{reference}}} \quad (D_{\text{reference}} = 2,150,000 \text{ daily passengers})$$
2. **Service Request Arrival Rates (TPS):**
   - **`reservations`**: $\text{Base TPS} \times M \times \text{Diurnal Factor}(t)$
   - **`flight-search`**: $\text{Reservations TPS} \times 15.0$ (calibrated 15:1 search-to-book ratio)
   - **`payment`**: $\text{Reservations TPS} \times 0.43$
   - **`check-in`**: $\text{Base TPS} \times M \times \text{Diurnal Factor}(t)$
   - **`baggage`**: $\text{Base TPS} \times M \times \text{Diurnal Factor}(t)$
   - **`crew`**: $\text{Base TPS} \times M \times \text{Diurnal Factor}(t)$
   - **`boundary-gateway`**: Sum of legacy-bound forwarded calls
   - **`legacy-core`**: Mainframe capacity ceiling (75 TPS limit)
3. **Computing Resource Demands:**
   - Parameterized by [`GoogleClusterTraceAdapter`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/workload/datasets/google_cluster.py) scheduling class empirical distributions, keeping public passenger demand and computing resource traces conceptually decoupled.

---

## 6. Reproduction & Test Execution

Run the complete forecasting test suite:
```bash
python3 -m unittest workload/tests/test_demand_forecasting.py -v
```

All 10 tests verify:
- Timestamp validation, duplicate resolution, and missing value interpolation
- Chronological splitting with zero future leakage
- Baseline execution (`LastValue` and `SeasonalNaive`)
- ML forecaster outperforming naive baselines
- Error metric computations (MAE, RMSE, sMAPE, MAPE)
- Output schema validity with 90% uncertainty intervals
- Calibrated demand-to-workload mapping and end-to-end cloud orchestrator integration
