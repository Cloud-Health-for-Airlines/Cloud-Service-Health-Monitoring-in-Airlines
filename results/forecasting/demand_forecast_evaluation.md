# BACCP Airline Demand Forecasting Evaluation Report

**Generated:** 2026-10-22T09:23:31.615898+00:00  
**Target Column:** `enplaned_passengers`  
**Dataset Coverage:** 2024-01-01 00:00:00 to 2025-12-30 00:00:00 (730 total observations)  

---

## 1. Chronological Splits (Zero Future-Data Leakage)

- **Train Set (70%):** 510 days (2024-01-01 00:00:00 -> 2025-05-24 00:00:00)
- **Validation Set (15%):** 110 days (2025-05-25 00:00:00 -> 2025-09-11 00:00:00)
- **Test Set (15%):** 110 days (2025-09-12 00:00:00 -> 2025-12-30 00:00:00)

---

## 2. Model Performance Comparison

| Model | MAE (Passengers) | RMSE (Passengers) | sMAPE (%) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| `LastValueBaseline` | 293,134.38 | 349,146.86 | 13.7% | 15.17% |
| `SeasonalNaiveBaseline (s=7)` | 196,447.7 | 236,314.3 | 9.12% | 9.76% |
| **`LightweightAutoRegressiveForecaster (Ridge)`** | **117,605.34** | **177,399.17** | **5.83%** | **5.54%** |

### Relative Improvements:
- **vs Last-Value Baseline:** MAE improved by **59.88%**, RMSE improved by **49.19%**
- **vs Seasonal Naive:** MAE improved by **40.13%**, RMSE improved by **24.93%**

---

## 3. Data Provenance & Methodology

1. **Dataset Origin:** U.S. Bureau of Transportation Statistics (BTS) TranStats Reporting Carrier On-Time Performance & T-100 Segment Data (Public Domain).
2. **Feature Formulation:** AutoRegressive lags ($t-1, t-2, t-3, t-7, t-14$), strictly backward rolling statistics, and day-of-week/month trigonometric Fourier terms.
3. **Calibrated Mapping to Workload:** Forecasts drive simulation parameters via `DemandToWorkloadMapper`, translating passenger volume into service request arrivals (search, reservations, payment, check-in, crew, baggage).
