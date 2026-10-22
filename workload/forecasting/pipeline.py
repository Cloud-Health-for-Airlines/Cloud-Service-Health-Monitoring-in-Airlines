"""Master training, evaluation, and reporting pipeline for airline passenger demand forecasting.

Executes:
1. Ingestion & integrity validation of BTS historical time series.
2. Chronological train (70%) / validation (15%) / test (15%) splitting.
3. Training & evaluation of baselines (LastValue and SeasonalNaive).
4. Training & evaluation of LightweightAutoRegressiveForecaster (Ridge).
5. Error metrics computation (MAE, RMSE, sMAPE, MAPE).
6. Artifact saving in results/forecasting/ with structured JSON and Markdown summaries.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .dataset import create_chronological_splits, load_airline_demand_dataset
from .models import (
    ForecastResult,
    LastValueBaseline,
    LightweightAutoRegressiveForecaster,
    SeasonalNaiveBaseline,
    compute_mae,
    compute_mape,
    compute_rmse,
    compute_smape,
)

logger = logging.getLogger("baccp.workload.forecasting.pipeline")

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "forecasting"


class AirlineDemandForecastingPipeline:
    """Coordinates end-to-end model training, baseline benchmarking, and artifact generation."""

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        target_column: str = "enplaned_passengers",
        results_dir: Optional[Union[str, Path]] = None,
    ):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR

    def run_pipeline(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        alpha: float = 1.0,
    ) -> Dict[str, Any]:
        """Execute complete forecasting experiment and return comparative results."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load & validate historical demand dataset
        df = load_airline_demand_dataset(
            path=self.dataset_path,
            target_column=self.target_column,
        )

        # 2. Chronological splitting
        train_df, val_df, test_df = create_chronological_splits(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # 3. Fit & evaluate Last-Value Baseline
        baseline_last = LastValueBaseline()
        baseline_last.fit(train_df, target_column=self.target_column)
        last_val_forecast = baseline_last.predict(
            horizon_steps=len(test_df),
            history=pd.concat([train_df, val_df]),
        )
        y_test = test_df[self.target_column].values
        last_val_metrics = {
            "mae": round(compute_mae(y_test, np.array(last_val_forecast.forecast_values)), 2),
            "rmse": round(compute_rmse(y_test, np.array(last_val_forecast.forecast_values)), 2),
            "smape": round(compute_smape(y_test, np.array(last_val_forecast.forecast_values)), 2),
            "mape": round(compute_mape(y_test, np.array(last_val_forecast.forecast_values)), 2),
        }

        # 4. Fit & evaluate Seasonal-Naive Baseline (weekly cycle s=7)
        baseline_seasonal = SeasonalNaiveBaseline(season_length=7)
        baseline_seasonal.fit(train_df, target_column=self.target_column)
        seasonal_forecast = baseline_seasonal.predict(
            horizon_steps=len(test_df),
            history=pd.concat([train_df, val_df]),
        )
        seasonal_metrics = {
            "mae": round(compute_mae(y_test, np.array(seasonal_forecast.forecast_values)), 2),
            "rmse": round(compute_rmse(y_test, np.array(seasonal_forecast.forecast_values)), 2),
            "smape": round(compute_smape(y_test, np.array(seasonal_forecast.forecast_values)), 2),
            "mape": round(compute_mape(y_test, np.array(seasonal_forecast.forecast_values)), 2),
        }

        # 5. Fit & evaluate Trained Lightweight Autoregressive Model
        model = LightweightAutoRegressiveForecaster(alpha=alpha)
        model.fit(
            train_df=train_df,
            val_df=val_df,
            target_column=self.target_column,
        )
        trained_forecast = model.predict(
            horizon_steps=len(test_df),
            history=pd.concat([train_df, val_df]),
        )
        trained_metrics = {
            "mae": round(compute_mae(y_test, np.array(trained_forecast.forecast_values)), 2),
            "rmse": round(compute_rmse(y_test, np.array(trained_forecast.forecast_values)), 2),
            "smape": round(compute_smape(y_test, np.array(trained_forecast.forecast_values)), 2),
            "mape": round(compute_mape(y_test, np.array(trained_forecast.forecast_values)), 2),
        }

        # 6. Comparison summary
        comparison = {
            "pipeline_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "target_variable": self.target_column,
            "total_observations": len(df),
            "chronological_splits": {
                "train_records": len(train_df),
                "train_start": str(train_df["date"].iloc[0]),
                "train_end": str(train_df["date"].iloc[-1]),
                "val_records": len(val_df),
                "val_start": str(val_df["date"].iloc[0]),
                "val_end": str(val_df["date"].iloc[-1]),
                "test_records": len(test_df),
                "test_start": str(test_df["date"].iloc[0]),
                "test_end": str(test_df["date"].iloc[-1]),
            },
            "models_evaluated": {
                "LastValueBaseline": last_val_metrics,
                "SeasonalNaiveBaseline_s7": seasonal_metrics,
                "LightweightAutoRegressiveForecaster_Ridge": trained_metrics,
            },
            "improvement_vs_last_value": {
                "mae_reduction_pct": round(
                    ((last_val_metrics["mae"] - trained_metrics["mae"]) / last_val_metrics["mae"]) * 100.0, 2
                ),
                "rmse_reduction_pct": round(
                    ((last_val_metrics["rmse"] - trained_metrics["rmse"]) / last_val_metrics["rmse"]) * 100.0, 2
                ),
            },
            "improvement_vs_seasonal_naive": {
                "mae_reduction_pct": round(
                    ((seasonal_metrics["mae"] - trained_metrics["mae"]) / seasonal_metrics["mae"]) * 100.0, 2
                ),
                "rmse_reduction_pct": round(
                    ((seasonal_metrics["rmse"] - trained_metrics["rmse"]) / seasonal_metrics["rmse"]) * 100.0, 2
                ),
            },
            "sample_test_forecasts": trained_forecast.to_dict()["forecast_values"][:14],
        }

        # 7. Persist evaluation artifacts
        json_path = self.results_dir / "demand_forecast_evaluation.json"
        json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

        # Markdown report
        md_content = f"""# BACCP Airline Demand Forecasting Evaluation Report

**Generated:** {comparison['pipeline_timestamp']}  
**Target Column:** `{self.target_column}`  
**Dataset Coverage:** {comparison['chronological_splits']['train_start']} to {comparison['chronological_splits']['test_end']} ({len(df)} total observations)  

---

## 1. Chronological Splits (Zero Future-Data Leakage)

- **Train Set (70%):** {comparison['chronological_splits']['train_records']} days ({comparison['chronological_splits']['train_start']} -> {comparison['chronological_splits']['train_end']})
- **Validation Set (15%):** {comparison['chronological_splits']['val_records']} days ({comparison['chronological_splits']['val_start']} -> {comparison['chronological_splits']['val_end']})
- **Test Set (15%):** {comparison['chronological_splits']['test_records']} days ({comparison['chronological_splits']['test_start']} -> {comparison['chronological_splits']['test_end']})

---

## 2. Model Performance Comparison

| Model | MAE (Passengers) | RMSE (Passengers) | sMAPE (%) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| `LastValueBaseline` | {last_val_metrics['mae']:,} | {last_val_metrics['rmse']:,} | {last_val_metrics['smape']}% | {last_val_metrics['mape']}% |
| `SeasonalNaiveBaseline (s=7)` | {seasonal_metrics['mae']:,} | {seasonal_metrics['rmse']:,} | {seasonal_metrics['smape']}% | {seasonal_metrics['mape']}% |
| **`LightweightAutoRegressiveForecaster (Ridge)`** | **{trained_metrics['mae']:,}** | **{trained_metrics['rmse']:,}** | **{trained_metrics['smape']}%** | **{trained_metrics['mape']}%** |

### Relative Improvements:
- **vs Last-Value Baseline:** MAE improved by **{comparison['improvement_vs_last_value']['mae_reduction_pct']}%**, RMSE improved by **{comparison['improvement_vs_last_value']['rmse_reduction_pct']}%**
- **vs Seasonal Naive:** MAE improved by **{comparison['improvement_vs_seasonal_naive']['mae_reduction_pct']}%**, RMSE improved by **{comparison['improvement_vs_seasonal_naive']['rmse_reduction_pct']}%**

---

## 3. Data Provenance & Methodology

1. **Dataset Origin:** U.S. Bureau of Transportation Statistics (BTS) TranStats Reporting Carrier On-Time Performance & T-100 Segment Data (Public Domain).
2. **Feature Formulation:** AutoRegressive lags ($t-1, t-2, t-3, t-7, t-14$), strictly backward rolling statistics, and day-of-week/month trigonometric Fourier terms.
3. **Calibrated Mapping to Workload:** Forecasts drive simulation parameters via `DemandToWorkloadMapper`, translating passenger volume into service request arrivals (search, reservations, payment, check-in, crew, baggage).
"""
        md_path = self.results_dir / "demand_forecast_evaluation.md"
        md_path.write_text(md_content, encoding="utf-8")

        logger.info(f"Saved forecasting evaluation artifacts to {self.results_dir}")
        return comparison
