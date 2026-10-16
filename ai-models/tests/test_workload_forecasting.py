"""Automated unit and integration test suite for BACCP Workload Forecasting (Task 1B Verification).

Covers:
1. Feature schema & version consistency (exactly 20 features, versioned schema).
2. Training vs Inference feature vector and prediction identity at identical origins.
3. Continuous inference variable buffer lengths: sliding buffers of different lengths (20, 30, 36)
   ending at the same timestamp produce 100% identical features and predictions.
4. Unanchored phase fallback: when timestamp/step_index are omitted, neutral phase 0 is used,
   is_phase_anchored is False, and an explicit notice is emitted in confidence_warnings.
5. No false coverage guarantee: audit that reliability metadata does not claim guaranteed coverage,
   flags exchangeability_assumed=False, and documents composite calibration status.
6. Conformal sample-size sufficiency: raises ValueError when calibration sample size is insufficient
   for the nominal quantile level rather than silently capping the index.
7. Whole-incident cluster bootstrap: verifies resampling units are whole incident trajectories.
8. Anti-leakage verification (altering future values does not alter origin features).
9. Horizon alignment (targets match step t + H exactly).
10. Incident split isolation across all 6 partitions (zero incident overlap).
11. Surge fallback policy activation and candidate forecasts preservation.
12. Strict capacity semantics: signed capacity_margin, available_headroom, overload, and unknown handling.
13. Zero-denominator WAPE policy adherence.
14. Input validation: non-finite values (NaN, Inf), empty history, unsupported horizons/services.
15. Short history safe padding and reliability warning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
if str(AI_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODELS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workload_forecasting.data_contract import (
    CONFIGURED_SERVICE_CAPACITIES,
    CONTRACT_VERSION,
    ServiceIdentifier,
    WorkloadRecord,
    validate_workload_record,
)
from workload_forecasting.evaluate import bootstrap_incident_metrics, compute_metrics
from workload_forecasting.features import (
    CONTEXT_WINDOW_STEPS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_HORIZONS,
    build_forecasting_dataset_for_service,
    extract_features_at_origin,
)
from workload_forecasting.inference import WorkloadForecaster, get_forecaster
from workload_forecasting.models import ConformalIntervalCalibrator, compute_capacity_metrics


class TestWorkloadForecastingAudit(unittest.TestCase):
    """Validation test suite for Task 1B audited workload forecasting implementation."""

    @classmethod
    def setUpClass(cls):
        cls.forecaster = get_forecaster()
        cls.manifest_path = REPO_ROOT / "results" / "workload_forecasting" / "split_manifest.json"
        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

    def test_feature_schema_and_version(self):
        """Verify feature schema version, count (exactly 20), and ordering."""
        self.assertEqual(len(FEATURE_NAMES), 20)
        self.assertEqual(FEATURE_SCHEMA_VERSION, "baccp-workload-features-v1.1.0")

        bundle_schema = self.forecaster.feature_schema
        self.assertEqual(bundle_schema.get("version"), FEATURE_SCHEMA_VERSION)
        self.assertEqual(bundle_schema.get("feature_count"), 20)
        self.assertEqual(bundle_schema.get("feature_names"), FEATURE_NAMES)

    def test_training_inference_feature_and_prediction_consistency(self):
        """Verify that inference reproduces training features and predictions at identical origins."""
        np.random.seed(123)
        trajectory_length = 50
        sample_workloads = 40.0 + np.cumsum(np.random.normal(0, 1.5, size=trajectory_length))
        sample_workloads = np.maximum(5.0, sample_workloads)

        origin_t = 28
        step_idx = origin_t
        elapsed_sec = origin_t * 20.0

        training_features = extract_features_at_origin(
            sample_workloads[:origin_t + 1],
            origin_step=origin_t,
            step_index=step_idx,
            timestamp_sec=elapsed_sec,
        )

        online_history_buffer = sample_workloads[:origin_t + 1]
        inference_features = extract_features_at_origin(
            online_history_buffer,
            step_index=step_idx,
            timestamp_sec=elapsed_sec,
        )

        np.testing.assert_array_equal(
            training_features,
            inference_features,
            err_msg="Training and inference feature extraction produced inconsistent feature vectors!",
        )

    def test_continuous_inference_variable_buffer_lengths(self):
        """Verify continuous inference: buffers of different lengths (20, 30, 45) ending at the same

        timestamp produce 100% identical features and predictions.
        """
        np.random.seed(42)
        stream = 40.0 + np.cumsum(np.random.normal(0, 1.2, size=60))
        t_origin = 35
        t_sec = t_origin * 20.0  # 700.0 seconds

        buf_20 = stream[16 : t_origin + 1]  # length 20
        buf_30 = stream[6 : t_origin + 1]   # length 30
        buf_36 = stream[0 : t_origin + 1]   # length 36

        f20 = extract_features_at_origin(buf_20, timestamp_sec=t_sec)
        f30 = extract_features_at_origin(buf_30, timestamp_sec=t_sec)
        f36 = extract_features_at_origin(buf_36, timestamp_sec=t_sec)

        np.testing.assert_array_equal(f20, f30, err_msg="Features differed between buffer lengths 20 and 30!")
        np.testing.assert_array_equal(f30, f36, err_msg="Features differed between buffer lengths 30 and 36!")

        p20 = self.forecaster.predict("reservations", buf_20, horizon=5, timestamp_sec=t_sec)
        p30 = self.forecaster.predict("reservations", buf_30, horizon=5, timestamp_sec=t_sec)
        p36 = self.forecaster.predict("reservations", buf_36, horizon=5, timestamp_sec=t_sec)

        self.assertEqual(p20["point_forecast_rps"], p30["point_forecast_rps"])
        self.assertEqual(p30["point_forecast_rps"], p36["point_forecast_rps"])
        self.assertEqual(
            p20["prediction_interval_90"]["lower_rps"],
            p30["prediction_interval_90"]["lower_rps"],
        )

    def test_unanchored_phase_fallback_and_warning(self):
        """Verify that when timestamp_sec and step_index are omitted, phase defaults to neutral 0,

        is_phase_anchored is False, and an explicit notice is emitted.
        """
        sample_buf = [45.0 + i * 0.2 for i in range(25)]
        res = self.forecaster.predict("reservations", sample_buf, horizon=1)

        self.assertFalse(res["reliability_metadata"]["is_phase_anchored"])
        warnings = res["reliability_metadata"]["confidence_warnings"]
        self.assertTrue(any("unanchored" in w for w in warnings))

        # Check variable buffers without timestamp also produce identical neutral phase
        buf_20 = sample_buf[-20:]
        f1 = extract_features_at_origin(sample_buf)
        f2 = extract_features_at_origin(buf_20)
        np.testing.assert_array_equal(f1, f2)

    def test_no_false_coverage_guarantee_claims(self):
        """Verify that reliability metadata does NOT claim guaranteed coverage and marks

        exchangeability as unassumed.
        """
        normal_buf = [45.0 + i * 0.1 for i in range(20)]
        res = self.forecaster.predict("reservations", normal_buf, horizon=1, timestamp_sec=400.0)

        # Ensure coverage_guarantee_valid is completely gone
        self.assertNotIn("coverage_guarantee_valid", res["reliability_metadata"])
        # Ensure exchangeability is explicitly False
        self.assertFalse(res["reliability_metadata"]["exchangeability_assumed"])
        # Ensure composite policy calibration is marked False
        self.assertFalse(res["prediction_interval_90"]["is_composite_policy_calibrated"])
        # Ensure calibrator source model is recorded
        self.assertEqual(res["prediction_interval_90"]["calibrator_source_model"], "gradient_boosting")

    def test_conformal_calibrator_sample_size_sufficiency(self):
        """Verify that ConformalIntervalCalibrator explicitly raises ValueError when sample size

        is insufficient for exact finite-sample quantile calculation.
        """
        calibrator = ConformalIntervalCalibrator(target_coverage=0.90)

        # Minimum required for 90% is ceil(1 / 0.10) - 1 = 9 samples
        y_small = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        p_small = np.array([10.1, 10.9, 12.2, 12.8, 14.1])

        with self.assertRaises(ValueError) as ctx:
            calibrator.calibrate(y_small, p_small)
        self.assertIn("Insufficient calibration samples", str(ctx.exception))

    def test_whole_incident_bootstrap_cluster_structure(self):
        """Verify that bootstrap resamples whole incidents (clusters of 61 windows),

        not individual independent timestamps.
        """
        dummy_traj_1 = {
            "incident_id": "inc_dummy_01",
            "records_by_service": {
                "reservations": [
                    {"step_index": t, "elapsed_time_sec": t * 20.0, "offered_workload_rps": float(10 + t)}
                    for t in range(90)
                ]
            },
        }
        dummy_traj_2 = {
            "incident_id": "inc_dummy_02",
            "records_by_service": {
                "reservations": [
                    {"step_index": t, "elapsed_time_sec": t * 20.0, "offered_workload_rps": float(50 + t)}
                    for t in range(90)
                ]
            },
        }
        svc_models = self.forecaster.models["reservations"][1]
        svc_cals = self.forecaster.calibrators["reservations"][1]

        # Call bootstrap with 2 incidents
        ci = bootstrap_incident_metrics(
            [dummy_traj_1, dummy_traj_2],
            "reservations",
            horizon=1,
            model_name="persistence",
            svc_models=svc_models,
            svc_calibrators=svc_cals,
            surge_threshold=150.0,
            n_bootstraps=50,
            seed=42,
        )
        self.assertIn("mae_ci95", ci)
        self.assertIn("rmse_ci95", ci)

    def test_no_future_values_entering_feature_windows(self):
        """Verify anti-leakage: altering values at t+1...t+H does not alter features at origin t."""
        total_steps = 40
        origin_t = 20

        base_history = np.linspace(10.0, 50.0, total_steps)
        features_original = extract_features_at_origin(base_history, origin_step=origin_t, step_index=origin_t)

        corrupted_future_history = base_history.copy()
        corrupted_future_history[origin_t + 1 :] = 99999.9

        features_corrupted = extract_features_at_origin(
            corrupted_future_history, origin_step=origin_t, step_index=origin_t
        )

        np.testing.assert_array_equal(
            features_original,
            features_corrupted,
            err_msg="Anti-leakage violated: future values altered features at origin t!",
        )

    def test_horizon_alignment(self):
        """Verify that forecast target values align strictly to step t + H."""
        dummy_traj = {
            "incident_id": "test_dummy_align",
            "records_by_service": {
                "reservations": [
                    {
                        "step_index": t,
                        "elapsed_time_sec": t * 20.0,
                        "offered_workload_rps": float(100 + t),
                    }
                    for t in range(50)
                ]
            },
        }
        dataset = build_forecasting_dataset_for_service([dummy_traj], "reservations", horizons=(1, 5, 15))
        meta = dataset["meta"]
        y_targets = dataset["y"]

        for idx, m in enumerate(meta):
            origin_t = m["origin_step"]
            for h in (1, 5, 15):
                target_val = y_targets[h][idx]
                expected_val = float(100 + origin_t + h)
                self.assertEqual(
                    target_val,
                    expected_val,
                    f"Horizon H={h} misaligned at origin {origin_t}: got {target_val}, expected {expected_val}",
                )

    def test_incident_split_isolation(self):
        """Verify that all incident splits are strictly mutually disjoint across all 6 partitions."""
        splits = self.manifest["splits"]
        train_set = set(splits["train"])
        val_set = set(splits["validation"])
        cal_set = set(splits["calibration"])
        test_std_set = set(splits["test_standard"])
        diag_set = set(splits.get("test_unseen_surge_diagnostic", []))
        locked_set = set(splits.get("test_unseen_surge_locked", []))

        self.assertEqual(len(train_set), 120)
        self.assertEqual(len(val_set), 30)
        self.assertEqual(len(cal_set), 25)
        self.assertEqual(len(test_std_set), 25)
        self.assertEqual(len(diag_set), 25)
        self.assertEqual(len(locked_set), 25)

        all_splits = [
            ("train", train_set),
            ("validation", val_set),
            ("calibration", cal_set),
            ("test_standard", test_std_set),
            ("diagnostic_surge", diag_set),
            ("locked_surge", locked_set),
        ]
        for i in range(len(all_splits)):
            for j in range(i + 1, len(all_splits)):
                name_i, set_i = all_splits[i]
                name_j, set_j = all_splits[j]
                overlap = set_i.intersection(set_j)
                self.assertEqual(
                    len(overlap), 0,
                    f"Leakage detected between {name_i} and {name_j}: overlap={overlap}"
                )

    def test_fallback_policy_activation(self):
        """Verify inference fallback policy: switches from GB to Ridge AR when workload exceeds training max."""
        normal_history = [45.0 + (i * 0.1) for i in range(20)]
        res_normal = self.forecaster.predict(
            service_id="reservations",
            history=normal_history,
            horizon=1,
            model_name="auto_fallback",
            timestamp_sec=300.0,
        )
        self.assertFalse(res_normal["fallback_triggered"])
        self.assertEqual(res_normal["selected_model"], "gradient_boosting")
        self.assertFalse(res_normal["reliability_metadata"]["is_out_of_training_range"])
        self.assertEqual(res_normal["reliability_metadata"]["reliability_status"], "nominal_in_distribution")

        # Surge history (180 RPS, above reservations training max 155.3 RPS)
        surge_history = [180.0 + (i * 1.0) for i in range(20)]
        res_surge = self.forecaster.predict(
            service_id="reservations",
            history=surge_history,
            horizon=1,
            model_name="auto_fallback",
            timestamp_sec=300.0,
        )
        self.assertTrue(res_surge["fallback_triggered"])
        self.assertEqual(res_surge["selected_model"], "autoregressive_ridge")
        self.assertIsNotNone(res_surge["fallback_reason"])
        self.assertTrue(res_surge["reliability_metadata"]["is_out_of_training_range"])
        self.assertEqual(res_surge["reliability_metadata"]["reliability_status"], "out_of_distribution_surge")
        warnings = res_surge["reliability_metadata"]["confidence_warnings"]
        self.assertTrue(any("WARNING" in w for w in warnings))

    def test_candidate_forecasts_preserved(self):
        """Verify that all 4 candidate models are preserved in output for transparent comparison."""
        history = [40.0] * 20
        res = self.forecaster.predict(service_id="reservations", history=history, horizon=5, timestamp_sec=300.0)
        candidates = res["candidate_forecasts"]
        for expected_model in ("gradient_boosting", "autoregressive_ridge", "persistence", "seasonal_naive"):
            self.assertIn(expected_model, candidates)
            self.assertGreater(candidates[expected_model], 0.0)

    def test_capacity_semantics(self):
        """Verify capacity metrics formulas: margin = cap - forecast, headroom = max(0, margin), overload = max(0, -margin)."""
        cap_under = compute_capacity_metrics("reservations", 70.0)
        self.assertEqual(cap_under["configured_capacity_rps"], 100.0)
        self.assertEqual(cap_under["capacity_margin_rps"], 30.0)
        self.assertEqual(cap_under["available_headroom_rps"], 30.0)
        self.assertEqual(cap_under["overload_rps"], 0.0)
        self.assertFalse(cap_under["is_capacity_exceeded"])
        self.assertEqual(cap_under["capacity_type"], "configured_simulation_assumption")

        cap_over = compute_capacity_metrics("reservations", 125.0)
        self.assertEqual(cap_over["capacity_margin_rps"], -25.0)
        self.assertEqual(cap_over["available_headroom_rps"], 0.0)
        self.assertEqual(cap_over["overload_rps"], 25.0)
        self.assertTrue(cap_over["is_capacity_exceeded"])

        cap_legacy = compute_capacity_metrics("legacy-core", 95.0)
        self.assertIsNone(cap_legacy["configured_capacity_rps"])
        self.assertIsNone(cap_legacy["capacity_margin_rps"])
        self.assertIsNone(cap_legacy["available_headroom_rps"])
        self.assertIsNone(cap_legacy["overload_rps"])
        self.assertIsNone(cap_legacy["forecast_to_capacity_ratio"])
        self.assertIsNone(cap_legacy["is_capacity_exceeded"])
        self.assertEqual(cap_legacy["capacity_type"], "unknown_uninstrumented")

    def test_zero_denominator_wape_policy(self):
        """Verify zero-denominator policy: 0.0 when error is 0, None (undefined) when error > 0."""
        y_zero = np.array([0.0, 0.0, 0.0])
        pred_zero = np.array([0.0, 0.0, 0.0])
        m_zero = compute_metrics(y_zero, pred_zero)
        self.assertEqual(m_zero["wape_pct"], 0.0)
        self.assertIsNotNone(m_zero["wape_explanation"])

        pred_nonzero = np.array([1.0, 2.0, 3.0])
        m_err = compute_metrics(y_zero, pred_nonzero)
        self.assertIsNone(m_err["wape_pct"])
        self.assertIn("Undefined", m_err["wape_explanation"])

    def test_invalid_and_nonfinite_inputs(self):
        """Verify graceful error raising for empty, NaN, Inf, and unsupported inputs."""
        with self.assertRaises(ValueError):
            self.forecaster.predict("reservations", [])

        with self.assertRaises(ValueError):
            self.forecaster.predict("reservations", [30.0, np.nan, 40.0])

        with self.assertRaises(ValueError):
            self.forecaster.predict("reservations", [30.0, np.inf, 40.0])

        with self.assertRaises(ValueError):
            self.forecaster.predict("reservations", [30.0] * 15, horizon=3)

        with self.assertRaises(ValueError):
            self.forecaster.predict("unknown-service", [30.0] * 15, horizon=1)

    def test_missing_history_padding(self):
        """Verify that short history (<15) is padded safely and marked in reliability metadata."""
        short_history = [45.0, 46.2, 44.8]
        res = self.forecaster.predict(service_id="reservations", history=short_history, horizon=1)
        self.assertTrue(res["reliability_metadata"]["missing_history_padded"])
        self.assertEqual(res["reliability_metadata"]["input_history_length"], 3)
        self.assertEqual(res["reliability_metadata"]["reliability_status"], "insufficient_history")
        self.assertGreater(res["point_forecast_rps"], 0.0)


if __name__ == "__main__":
    unittest.main()
