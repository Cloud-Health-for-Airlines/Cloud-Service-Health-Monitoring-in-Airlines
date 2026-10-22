"""Unit and integration tests for the BACCP airline demand forecasting component.

Tests:
1. Dataset ingestion, integrity validation, duplicate resolution, and missing value handling.
2. Chronological time-series splitting (strictly non-shuffled, zero future leakage).
3. Naive baselines (LastValueBaseline and SeasonalNaiveBaseline).
4. Trained LightweightAutoRegressiveForecaster (Ridge) outperforming naive baselines on held-out test data.
5. Standard forecast error metrics (MAE, RMSE, sMAPE, MAPE).
6. Forecast output schema validation (timestamps, predicted demand, horizons, uncertainty intervals).
7. Calibrated DemandToWorkloadMapper translating passenger forecasts to service TPS.
8. End-to-end integration of forecasted workloads with BACCPCloudOrchestrator.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cloud.orchestrator import BACCPCloudOrchestrator
from workload.forecasting import (
    AirlineDemandForecastingPipeline,
    DatasetValidationError,
    DemandConversionConfig,
    DemandToWorkloadMapper,
    ForecastResult,
    LastValueBaseline,
    LightweightAutoRegressiveForecaster,
    SeasonalNaiveBaseline,
    compute_mae,
    compute_mape,
    compute_rmse,
    compute_smape,
    create_chronological_splits,
    load_airline_demand_dataset,
)


class TestAirlineDemandForecasting(unittest.TestCase):
    """Test suite validating all demand forecasting and workload integration requirements."""

    def setUp(self):
        self.dataset_path = REPO_ROOT / "workload" / "datasets" / "sample_data" / "bts_daily_passenger_demand.csv"

    def test_dataset_loading_and_validation(self):
        """Verify successful ingestion, date parsing, and schema validation of BTS dataset."""
        df = load_airline_demand_dataset(self.dataset_path)
        self.assertFalse(df.empty)
        self.assertIn("date", df.columns)
        self.assertIn("enplaned_passengers", df.columns)
        self.assertIn("scheduled_flights", df.columns)
        
        # Chronological order
        dates = pd.to_datetime(df["date"])
        self.assertTrue(dates.is_monotonic_increasing)

        # Non-negative passenger counts
        self.assertTrue((df["enplaned_passengers"] > 0).all())

    def test_validation_failure_modes(self):
        """Verify that corrupt, empty, negative, or malformed datasets trigger DatasetValidationError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # 1. Non-existent file
            with self.assertRaises(FileNotFoundError):
                load_airline_demand_dataset(tmp / "nonexistent.csv")

            # 2. Missing date column
            bad_csv = tmp / "missing_date.csv"
            bad_csv.write_text("passengers,flights\n100,10\n200,20\n")
            with self.assertRaises(DatasetValidationError):
                load_airline_demand_dataset(bad_csv)

            # 3. Missing target column
            bad_csv2 = tmp / "missing_target.csv"
            bad_csv2.write_text("date,other_col\n2024-01-01,10\n2024-01-02,20\n")
            with self.assertRaises(DatasetValidationError):
                load_airline_demand_dataset(bad_csv2, target_column="enplaned_passengers")

            # 4. Negative values in target column
            bad_csv3 = tmp / "negative_target.csv"
            bad_csv3.write_text("date,enplaned_passengers\n2024-01-01,-500\n2024-01-02,200\n")
            with self.assertRaises(DatasetValidationError):
                load_airline_demand_dataset(bad_csv3)

    def test_duplicate_and_missing_date_handling(self):
        """Verify that duplicate timestamps are averaged and calendar gaps are smoothly interpolated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_gaps.csv"
            # Day 2024-01-02 is missing; 2024-01-03 is duplicated
            csv_path.write_text(
                "date,enplaned_passengers\n"
                "2024-01-01,2000000\n"
                "2024-01-03,2200000\n"
                "2024-01-03,2400000\n"
                "2024-01-04,2100000\n"
            )
            df = load_airline_demand_dataset(csv_path, fill_missing=True)
            self.assertEqual(len(df), 4) # 2024-01-01, 02 (interpolated), 03 (averaged), 04
            dates = list(df["date"].astype(str))
            self.assertEqual(dates, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
            # Averaged duplicate for 2024-01-03
            val_03 = float(df[df["date"].astype(str) == "2024-01-03"]["enplaned_passengers"].iloc[0])
            self.assertAlmostEqual(val_03, 2300000.0, places=1)

    def test_chronological_splits_zero_leakage(self):
        """Verify that chronological splitting preserves strict sequential order without shuffling."""
        df = load_airline_demand_dataset(self.dataset_path)
        train_df, val_df, test_df = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))
        train_max_date = pd.to_datetime(train_df["date"]).max()
        val_min_date = pd.to_datetime(val_df["date"]).min()
        val_max_date = pd.to_datetime(val_df["date"]).max()
        test_min_date = pd.to_datetime(test_df["date"]).min()

        # Strict temporal inequality
        self.assertLess(train_max_date, val_min_date)
        self.assertLess(val_max_date, test_min_date)

    def test_metrics_computations(self):
        """Verify mathematical correctness of MAE, RMSE, sMAPE, and MAPE."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 330.0])

        mae = compute_mae(y_true, y_pred)
        rmse = compute_rmse(y_true, y_pred)
        smape = compute_smape(y_true, y_pred)
        mape = compute_mape(y_true, y_pred)

        # Expected: |10| + |10| + |30| = 50 / 3 = 16.6667
        self.assertAlmostEqual(mae, 16.6667, places=3)
        # Expected: sqrt((100 + 100 + 900)/3) = sqrt(366.6667) = 19.1485
        self.assertAlmostEqual(rmse, 19.1485, places=3)
        self.assertGreater(smape, 0.0)
        self.assertGreater(mape, 0.0)

    def test_baseline_forecasting_models(self):
        """Verify LastValueBaseline and SeasonalNaiveBaseline execution and forecasts."""
        df = load_airline_demand_dataset(self.dataset_path)
        train_df, val_df, test_df = create_chronological_splits(df)

        # 1. LastValue
        last_baseline = LastValueBaseline()
        last_baseline.fit(train_df)
        res_last = last_baseline.predict(horizon_steps=7, history=train_df)
        self.assertEqual(len(res_last.forecast_values), 7)
        # All forecast values equal last observation
        self.assertEqual(len(set(res_last.forecast_values)), 1)
        self.assertEqual(res_last.forecast_values[0], float(train_df["enplaned_passengers"].iloc[-1]))

        # 2. SeasonalNaive (s=7)
        seasonal_baseline = SeasonalNaiveBaseline(season_length=7)
        seasonal_baseline.fit(train_df)
        res_seasonal = seasonal_baseline.predict(horizon_steps=14, history=train_df)
        self.assertEqual(len(res_seasonal.forecast_values), 14)
        # Day 8 should match Day 1 (weekly cycle recurrence)
        self.assertAlmostEqual(res_seasonal.forecast_values[0], res_seasonal.forecast_values[7], places=2)

    def test_trained_model_outperforms_baselines(self):
        """Verify that LightweightAutoRegressiveForecaster achieves superior accuracy on test data."""
        pipeline = AirlineDemandForecastingPipeline(dataset_path=self.dataset_path)
        results = pipeline.run_pipeline()

        models = results["models_evaluated"]
        self.assertIn("LastValueBaseline", models)
        self.assertIn("SeasonalNaiveBaseline_s7", models)
        self.assertIn("LightweightAutoRegressiveForecaster_Ridge", models)

        trained_mae = models["LightweightAutoRegressiveForecaster_Ridge"]["mae"]
        last_val_mae = models["LastValueBaseline"]["mae"]
        seasonal_mae = models["SeasonalNaiveBaseline_s7"]["mae"]

        # Trained model must outperform naive baselines
        self.assertLess(trained_mae, last_val_mae)
        self.assertLess(trained_mae, seasonal_mae)

        # Check positive relative improvements
        self.assertGreater(results["improvement_vs_last_value"]["mae_reduction_pct"], 0.0)
        self.assertGreater(results["improvement_vs_seasonal_naive"]["mae_reduction_pct"], 0.0)

    def test_forecast_output_schema_and_intervals(self):
        """Verify ForecastResult schema, non-empty timestamps, and 90% uncertainty bounds."""
        df = load_airline_demand_dataset(self.dataset_path)
        train_df, val_df, test_df = create_chronological_splits(df)

        model = LightweightAutoRegressiveForecaster()
        model.fit(train_df=train_df, val_df=val_df)
        result = model.predict(horizon_steps=14, history=pd.concat([train_df, val_df]))

        self.assertIsInstance(result, ForecastResult)
        self.assertEqual(result.horizon_steps, 14)
        self.assertEqual(len(result.timestamps), 14)
        self.assertEqual(len(result.forecast_values), 14)
        self.assertEqual(len(result.lower_bounds_90), 14)
        self.assertEqual(len(result.upper_bounds_90), 14)

        # Uncertainty intervals bracket predictions
        for low, pred, high in zip(result.lower_bounds_90, result.forecast_values, result.upper_bounds_90):
            self.assertLessEqual(low, pred)
            self.assertGreaterEqual(high, pred)

        # JSON serializability
        d = result.to_dict()
        self.assertIn("forecast_values", d)
        self.assertIn("lower_bounds_90", d)
        self.assertIn("upper_bounds_90", d)

    def test_demand_to_workload_mapper(self):
        """Verify calibrated conversion of forecasted passenger volume into simulated IT workloads."""
        mapper = DemandToWorkloadMapper()

        # 1. Reference volume (2.15M) gives multiplier 1.0
        mult_1 = mapper.compute_demand_multiplier(2_150_000.0)
        self.assertAlmostEqual(mult_1, 1.0, places=4)

        # 2. Elevated volume (2.58M) gives multiplier 1.2
        mult_12 = mapper.compute_demand_multiplier(2_580_000.0)
        self.assertAlmostEqual(mult_12, 1.2, places=4)

        # 3. Service TPS calculation
        tps = mapper.convert_demand_to_service_tps(forecasted_passengers=2_580_000.0, hour_of_day=9.0)
        self.assertIn("reservations", tps)
        self.assertIn("flight-search", tps)
        self.assertIn("boundary-gateway", tps)
        self.assertIn("legacy-core", tps)

        # Higher passenger demand scales reservation TPS
        tps_baseline = mapper.convert_demand_to_service_tps(forecasted_passengers=2_150_000.0, hour_of_day=9.0)
        self.assertGreater(tps["reservations"], tps_baseline["reservations"])
        self.assertGreater(tps["flight-search"], tps["reservations"] * 10.0)

        # 4. Generate WorkloadSnapshot with enriched provenance
        snapshots = mapper.generate_workload_from_forecast(
            forecasted_passengers=2_580_000.0,
            duration_seconds=60.0,
            sampling_interval_sec=20.0,
        )
        self.assertEqual(len(snapshots), 3)

        obs_res = snapshots[0].observations["reservations"]
        self.assertIn("forecasted_daily_passengers", obs_res.provenance.forecast_values)
        self.assertEqual(obs_res.provenance.forecast_values["forecasted_daily_passengers"], 2580000.0)
        self.assertIn("demand_to_request_mapping", obs_res.provenance.assumptions)

    def test_end_to_end_orchestrator_integration_with_forecast(self):
        """Verify that forecast-driven workload snapshots feed directly into BACCPCloudOrchestrator."""
        mapper = DemandToWorkloadMapper()
        snapshots = mapper.generate_workload_from_forecast(
            forecasted_passengers=2_400_000.0,
            duration_seconds=60.0,
            sampling_interval_sec=20.0,
        )
        first_snap = snapshots[0]
        telemetry_payload = first_snap.to_orchestrator_telemetry()

        orchestrator = BACCPCloudOrchestrator()
        result = orchestrator.run_pipeline(
            telemetry=telemetry_payload,
            sync_drift_score=first_snap.sync_drift_score,
            active_fault=first_snap.active_fault,
            fault_level=first_snap.fault_level,
        )

        self.assertEqual(result.get("status"), "success")
        self.assertIn("telemetry", result)
        self.assertIn("prediction", result)
        self.assertIn("cascade_probability", result["prediction"])


if __name__ == "__main__":
    unittest.main()
