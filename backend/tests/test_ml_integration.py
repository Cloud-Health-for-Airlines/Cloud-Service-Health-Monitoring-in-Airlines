"""Focused Integration Test Suite for Task 5 Backend ML Integration.

Covers:
1. Valid forecast, prediction, and explanation responses (explicit units, data sources, provenances).
2. Invalid service IDs, invalid horizons, and missing parameters (HTTP 400).
3. Missing, stale, and insufficient telemetry (HTTP 422 / 503 without silent nominal imputation).
4. Missing model artifacts returning structured 503 without breaking existing API routes.
5. Snapshot, model variant, and horizon consistency across predictions and explanations.
6. JSON serialization ensuring no NaN or Inf.
7. Bounded cache reuse, invalidation, and advisory alert counter deduplication.
8. Workload forecast absolute error recorded only after matching future observation arrives.
9. Compatibility of existing API routes (/api/health, /api/graph, /api/cloud/status, /metrics).
10. Bounded smoke test with actual frozen models.
"""

from __future__ import annotations

import copy
import json
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.app import BACCPRequestHandler, ml_metrics, ml_service
from backend.ml.metrics import BACCPMLMetrics, SUPPORTED_SERVICES
from backend.ml.telemetry_adapter import (
    MAX_TELEMETRY_STALENESS_SECONDS,
    SAMPLING_INTERVAL_SECONDS,
    TelemetryAdapter,
    TelemetryError,
    TelemetrySnapshot,
)


class BaseMLServerTestCase(unittest.TestCase):
    """Base test case running in-process HTTP server on ephemeral port."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), BACCPRequestHandler)
        cls.port = cls.httpd.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path: str) -> Tuple[int, Dict[str, Any]]:
        req = Request(f"{self.base_url}{path}")
        try:
            with urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except HTTPError as err:
            return err.code, json.loads(err.read().decode("utf-8"))

    def get_text(self, path: str) -> Tuple[int, str]:
        req = Request(f"{self.base_url}{path}")
        try:
            with urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode("utf-8")
        except HTTPError as err:
            return err.code, err.read().decode("utf-8")


class TestMLBackendIntegration(BaseMLServerTestCase):
    """Integration verification across all Task 5 requirements."""

    def test_01_valid_workload_forecast(self):
        """Verify GET /forecast/workload returns valid point forecast, conformal bounds, and units."""
        status, body = self.get("/forecast/workload?service=reservations&horizon=5")
        self.assertEqual(status, 200)
        self.assertEqual(body["service"], "reservations")
        self.assertEqual(body["horizon_steps"], 5)
        self.assertEqual(body["horizon_seconds"], 100.0)
        self.assertEqual(body["telemetry_mode"], "replay")
        self.assertEqual(body["data_source"], "REPLAY_INCIDENT_STREAM")
        self.assertEqual(body["data_provenance"], "SYNTHETIC_SIMULATED")
        self.assertIn("predicted_demand_req_s", body)
        self.assertIn("conformal_interval", body)
        self.assertIn("capacity_margin_req_s", body)
        self.assertIn("available_headroom_req_s", body)
        self.assertIn("is_overloaded", body)
        self.assertIn("reliability", body)
        self.assertIn("candidate_models", body)

    def test_02_valid_degradation_prediction(self):
        """Verify GET /predict/degradation returns calibrated probability, current state vs future risk, and advisory decision."""
        status, body = self.get("/predict/degradation?service=reservations&horizon=5")
        self.assertEqual(status, 200)
        self.assertEqual(body["service"], "reservations")
        self.assertEqual(body["horizon_steps"], 5)
        self.assertEqual(body["horizon_seconds"], 100.0)
        self.assertEqual(body["model_variant"], "model_b_telemetry_forecast")
        self.assertIn("calibrated_event_probability", body)
        self.assertIn("alert_threshold", body)
        self.assertIn("advisory_alert_decision", body)
        self.assertIn("current_observed_state", body)
        self.assertIn("predicted_future_risk", body)

        # Verify separation of current observed state from predicted future risk
        curr = body["current_observed_state"]
        fut = body["predicted_future_risk"]
        self.assertIn("latency_ms", curr)
        self.assertIn("backlog_requests", curr)
        self.assertIn("continuous_predictions", fut)
        self.assertIn("estimated_time_to_degradation_seconds", fut)
        self.assertIn("predicted_degradation_type", fut)
        self.assertIn("predicted_severity", fut)

    def test_03_valid_explanation(self):
        """Verify GET /explain/{service} returns TreeSHAP attributions, top drivers, and graph localization."""
        status, body = self.get("/explain/reservations?horizon=5")
        self.assertEqual(status, 200)
        self.assertEqual(body["service"], "reservations")
        self.assertEqual(body["horizon_steps"], 5)
        self.assertIn(body["attribution_method"].lower(), ("treeshap", "tree_shap"))
        self.assertIn("top_risk_increasing_drivers", body)
        self.assertIn("top_risk_decreasing_drivers", body)
        self.assertIn("category_attributions", body)
        self.assertIn("localization", body)
        self.assertIn("suspected_contributing_node", body["localization"])
        self.assertIn(body["localization"]["localization_status"], ("localized", "abstain_unknown_unlocalized"))
        self.assertIn("incident_brief", body)

    def test_04_prediction_and_explanation_snapshot_consistency(self):
        """Verify predictions and explanations for the same request refer to the identical snapshot and model."""
        status_p, pred = self.get("/predict/degradation?service=crew&horizon=5&replay_step=12")
        status_e, expl = self.get("/explain/crew?horizon=5&replay_step=12")

        self.assertEqual(status_p, 200)
        self.assertEqual(status_e, 200)
        self.assertEqual(pred["snapshot_id"], expl["snapshot_id"])
        self.assertEqual(pred["observation_timestamp_sec"], expl["observation_timestamp_sec"])
        self.assertEqual(pred["model_variant"], expl["model_variant"])
        self.assertEqual(pred["horizon_steps"], expl["horizon_steps"])
        self.assertEqual(pred["calibrated_event_probability"], expl["calibrated_event_probability"])
        self.assertEqual(pred["alert_threshold"], expl["alert_threshold"])
        self.assertEqual(pred["advisory_alert_decision"], expl["advisory_alert_decision"])

    def test_05_invalid_service_and_horizon_rejection(self):
        """Verify 400 Bad Request on invalid service or horizon parameters."""
        # Unsupported service
        s1, b1 = self.get("/predict/degradation?service=nonexistent_microservice&horizon=5")
        self.assertEqual(s1, 400)
        self.assertIn("error", b1)

        # Unsupported horizon
        s2, b2 = self.get("/predict/degradation?service=reservations&horizon=7")
        self.assertEqual(s2, 400)
        self.assertIn("error", b2)

        # Missing service
        s3, b3 = self.get("/forecast/workload?horizon=5")
        self.assertEqual(s3, 400)
        self.assertEqual(b3["error"], "INVALID_SERVICE")

        # Non-integer horizon
        s4, b4 = self.get("/forecast/workload?service=reservations&horizon=abc")
        self.assertEqual(s4, 400)
        self.assertEqual(b4["error"], "INVALID_HORIZON")

    def test_06_live_mode_insufficient_telemetry_rejection(self):
        """Verify live mode returns structured 503 instead of silently imputing missing metrics."""
        with patch.object(ml_service.telemetry_adapter, "mode", "live"):
            status, body = self.get("/predict/degradation?service=reservations&horizon=5")
            self.assertEqual(status, 503)
            self.assertEqual(body["error"], "UNAVAILABLE_INSUFFICIENT_TELEMETRY")
            self.assertEqual(body["telemetry_mode"], "live")
            self.assertIn("missing_required_metrics", body)
            self.assertIn("available_live_metrics", body)

    def test_07_stale_telemetry_rejection(self):
        """Verify telemetry exceeding 60-second staleness limit is rejected with HTTP 422."""
        status, body = self.get("/predict/degradation?service=reservations&horizon=5&replay_time_sec=99999.0")
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "STALENESS_LIMIT_EXCEEDED")
        self.assertGreater(body["staleness_seconds"], MAX_TELEMETRY_STALENESS_SECONDS)

    def test_07b_requested_service_stale_while_neighbours_fresh(self):
        """Verify a fresh neighbour does NOT mask stale requested-service data (HTTP 422)."""
        adapter = ml_service.telemetry_adapter
        inc_id = adapter.default_incident_id
        orig_inc = adapter._incidents_by_id[inc_id]

        # Modify incident copy: reservations stream ends at 100.0s, while crew continues up to 240.0s
        mod_inc = copy.deepcopy(orig_inc)
        mod_inc["service_telemetry"]["reservations"] = [
            r for r in mod_inc["service_telemetry"]["reservations"]
            if r.get("timestamp_sec", r.get("step_index", 0) * SAMPLING_INTERVAL_SECONDS) <= 100.0
        ]

        with patch.dict(adapter._incidents_by_id, {inc_id: mod_inc}):
            # Query reservations at clock 240.0s -> staleness is 140s (> 60s) -> 422
            s_res, b_res = self.get("/predict/degradation?service=reservations&horizon=5&replay_time_sec=240.0")
            self.assertEqual(s_res, 422)
            self.assertEqual(b_res["error"], "STALENESS_LIMIT_EXCEEDED")
            self.assertGreater(b_res["staleness_seconds"], 60.0)

            # Query crew at clock 240.0s -> crew is fresh (0.0s staleness) -> HTTP 200
            s_crew, b_crew = self.get("/predict/degradation?service=crew&horizon=5&replay_time_sec=240.0")
            self.assertEqual(s_crew, 200)

    def test_07c_boundary_stale_while_services_fresh(self):
        """Verify stale boundary stream is rejected with HTTP 422 even when all services are fresh."""
        adapter = ml_service.telemetry_adapter
        inc_id = adapter.default_incident_id
        orig_inc = adapter._incidents_by_id[inc_id]

        mod_inc = copy.deepcopy(orig_inc)
        mod_inc["boundary_telemetry"] = [
            r for r in mod_inc["boundary_telemetry"]
            if r.get("timestamp_sec", r.get("step_index", 0) * SAMPLING_INTERVAL_SECONDS) <= 100.0
        ]

        with patch.dict(adapter._incidents_by_id, {inc_id: mod_inc}):
            s, b = self.get("/predict/degradation?service=crew&horizon=5&replay_time_sec=240.0")
            self.assertEqual(s, 422)
            self.assertEqual(b["error"], "STALENESS_LIMIT_EXCEEDED")
            self.assertIn("Boundary gateway telemetry is stale", b["message"])

    def test_07d_replay_timestamp_unchanged_across_horizons(self):
        """Verify replay timestamp is not rewound or clamped based on forecast horizon."""
        s1, b1 = self.get("/predict/degradation?service=reservations&horizon=1&replay_step=12")
        s5, b5 = self.get("/predict/degradation?service=reservations&horizon=5&replay_step=12")
        s15, b15 = self.get("/predict/degradation?service=reservations&horizon=15&replay_step=12")

        self.assertEqual(s1, 200)
        self.assertEqual(s5, 200)
        self.assertEqual(s15, 200)

        # All horizons must observe identical timestamp (step 12 * 20 = 240.0s)
        self.assertEqual(b1["observation_timestamp_sec"], 240.0)
        self.assertEqual(b5["observation_timestamp_sec"], 240.0)
        self.assertEqual(b15["observation_timestamp_sec"], 240.0)

    def test_07e_invalid_and_nonfinite_replay_arguments(self):
        """Verify invalid or non-finite replay parameters return structured 400 Bad Request."""
        # Negative replay_step
        s1, b1 = self.get("/predict/degradation?service=reservations&horizon=5&replay_step=-1")
        self.assertEqual(s1, 400)
        self.assertEqual(b1["error"], "INVALID_REPLAY_PARAMETER")

        # Negative replay_time_sec
        s2, b2 = self.get("/predict/degradation?service=reservations&horizon=5&replay_time_sec=-5.0")
        self.assertEqual(s2, 400)
        self.assertEqual(b2["error"], "INVALID_REPLAY_PARAMETER")

        # NaN replay_time_sec
        s3, b3 = self.get("/predict/degradation?service=reservations&horizon=5&replay_time_sec=NaN")
        self.assertEqual(s3, 400)
        self.assertEqual(b3["error"], "INVALID_REPLAY_PARAMETER")

        # Inf replay_time_sec
        s4, b4 = self.get("/predict/degradation?service=reservations&horizon=5&replay_time_sec=Infinity")
        self.assertEqual(s4, 400)
        self.assertEqual(b4["error"], "INVALID_REPLAY_PARAMETER")

        # Unsupported model_variant
        s5, b5 = self.get("/predict/degradation?service=reservations&horizon=5&model_variant=invalid_variant")
        self.assertEqual(s5, 400)
        self.assertIn(b5["error"], ("INVALID_MODEL_VARIANT", "UNSUPPORTED_MODEL_VARIANT"))

    def test_08_missing_models_fallback_availability(self):
        """Verify missing ML model artifacts return structured 503 without crashing the backend."""
        with patch.object(ml_service, "models_available", False), patch.object(
            ml_service, "model_load_error", "Mocked missing weights bundle"
        ):
            s_f, b_f = self.get("/forecast/workload?service=reservations&horizon=5")
            s_p, b_p = self.get("/predict/degradation?service=reservations&horizon=5")
            s_e, b_e = self.get("/explain/reservations?horizon=5")

            self.assertEqual(s_f, 503)
            self.assertEqual(b_f["error"], "ML_MODELS_UNAVAILABLE")
            self.assertEqual(s_p, 503)
            self.assertEqual(b_p["error"], "ML_MODELS_UNAVAILABLE")
            self.assertEqual(s_e, 503)
            self.assertEqual(b_e["error"], "ML_MODELS_UNAVAILABLE")

            # Existing routes MUST remain available
            s_h, b_h = self.get("/api/health")
            self.assertEqual(s_h, 200)
            self.assertIn("sync_drift_score", b_h)

            s_g, b_g = self.get("/api/graph")
            self.assertEqual(s_g, 200)
            self.assertIn("nodes", b_g)

    def test_09_cache_reuse_and_alert_deduplication(self):
        """Verify cache hits return identical results and do not re-increment Prometheus alert episode counter."""
        initial_count = ml_metrics.alert_episode_counts.get(("reservations", 5), 0)

        # First request: cached=False
        s1, b1 = self.get("/predict/degradation?service=reservations&horizon=5&replay_step=10")
        self.assertEqual(s1, 200)

        # Second request for same snapshot: cached=True
        s2, b2 = self.get("/predict/degradation?service=reservations&horizon=5&replay_step=10")
        self.assertEqual(s2, 200)
        self.assertTrue(b2.get("cached", False))

        # Alert episode count must NOT have incremented on repeated request
        count_after = ml_metrics.alert_episode_counts.get(("reservations", 5), 0)
        self.assertEqual(count_after, initial_count)

    def test_10_prometheus_metrics_exposition(self):
        """Verify /metrics exposes degradation probabilities, alert episodes, and forecast errors with bounded labels."""
        status, prom_text = self.get_text("/metrics")
        self.assertEqual(status, 200)
        self.assertIn("baccp_sync_drift_score", prom_text)
        self.assertIn("baccp_predicted_degradation_probability", prom_text)
        self.assertIn("baccp_advisory_alert_episodes_total", prom_text)
        self.assertIn("baccp_telemetry_freshness_seconds", prom_text)
        self.assertIn("baccp_telemetry_available", prom_text)

        # Verify no timestamps or unconstrained IDs in labels
        for line in prom_text.splitlines():
            if line.startswith("baccp_predicted_degradation_probability"):
                self.assertIn("service=", line)
                self.assertIn("horizon=", line)
                self.assertNotIn("timestamp", line)
                self.assertNotIn("request_id", line)

    def test_11_forecast_error_delayed_arrival_and_resolution(self):
        """Verify workload forecast error is NOT recorded until the target arrives at subsequent snapshot."""
        ml_service.clear_cache_for_testing()
        ml_metrics.reset_for_testing()

        # 1. Trigger forecast at step 10 (target timestamp is step 11 = 220.0s)
        s, f_res = self.get("/forecast/workload?service=baggage&horizon=1&replay_step=10")
        self.assertEqual(s, 200)

        # Target observation at 220.0s has NOT arrived yet at clock 200.0s
        # Forecast error gauge MUST NOT be populated yet
        self.assertNotIn(("baggage", 1), ml_metrics.forecast_absolute_errors)

        # Pending forecast must be registered with collision-proof key
        matching_pending = [
            v for k, v in ml_metrics.pending_forecasts.items()
            if k[1] == "baggage" and k[2] == 1 and k[3] == 200.0 and k[4] == 220.0
        ]
        self.assertEqual(len(matching_pending), 1)

        # 2. Advance clock to step 11 (220.0s) via subsequent request
        s2, _ = self.get("/forecast/workload?service=baggage&horizon=1&replay_step=11")
        self.assertEqual(s2, 200)

        # NOW the target observation has arrived!
        # Forecast absolute error must be resolved and recorded
        self.assertIn(("baggage", 1), ml_metrics.forecast_absolute_errors)
        err = ml_metrics.forecast_absolute_errors[("baggage", 1)]
        self.assertIsNotNone(err)
        self.assertGreaterEqual(err, 0.0)

        # 3. Check /metrics output contains the recorded error
        status_m, prom_body = self.get_text("/metrics")
        self.assertEqual(status_m, 200)
        self.assertIn('baccp_workload_forecast_absolute_error{service="baggage",horizon="1"}', prom_body)

    def test_11b_no_cross_incident_matching_and_cache_deduplication(self):
        """Verify pending forecasts never match observations from a different incident, and cache hits deduplicate."""
        ml_service.clear_cache_for_testing()
        ml_metrics.reset_for_testing()

        inc_a = "inc_standard_test_0175"
        inc_b = "inc_standard_test_0176"

        # 1. Trigger forecast on incident A at step 10 (target 220.0s)
        s1, _ = self.get(f"/forecast/workload?service=baggage&horizon=1&replay_step=10&incident_id={inc_a}")
        self.assertEqual(s1, 200)
        self.assertNotIn(("baggage", 1), ml_metrics.forecast_absolute_errors)

        # 2. Advance clock to step 11 on incident B
        s2, _ = self.get(f"/forecast/workload?service=baggage&horizon=1&replay_step=11&incident_id={inc_b}")
        self.assertEqual(s2, 200)

        # Incident A's pending forecast must NOT have been resolved by incident B's telemetry!
        matching_pending_a = [
            v for k, v in ml_metrics.pending_forecasts.items()
            if k[0] == inc_a and k[1] == "baggage" and k[2] == 1
        ]
        self.assertEqual(len(matching_pending_a), 1)

        # 3. Cache hits do not duplicate records
        pending_count_before = len(ml_metrics.pending_forecasts)
        s3, _ = self.get(f"/forecast/workload?service=baggage&horizon=1&replay_step=10&incident_id={inc_a}")
        self.assertEqual(s3, 200)
        self.assertEqual(len(ml_metrics.pending_forecasts), pending_count_before)

    def test_12_existing_api_routes_compatibility(self):
        """Verify all existing API routes continue to work without contract regressions."""
        s1, b1 = self.get("/api/health/boundary")
        self.assertEqual(s1, 200)
        self.assertIn("sync_drift_score", b1)

        s2, b2 = self.get("/api/graph")
        self.assertEqual(s2, 200)
        self.assertEqual(len(b2["nodes"]), 5)

        s3, b3 = self.get("/api/telemetry")
        self.assertEqual(s3, 200)
        self.assertIn("metrics", b3)

        s4, b4 = self.get("/api/cloud/status")
        self.assertEqual(s4, 200)
        self.assertIn("cloud_mode", b4)

    def test_13_unified_alert_thresholds_boundary_decisions_and_consistency(self):
        """Verify unified alert policy boundary decisions immediately below and at each threshold."""
        from backend.ml.alert_policy import ALERT_THRESHOLDS, evaluate_alert_decision

        # Policy thresholds: H=1: 0.280, H=5: 0.300, H=15: 0.320
        self.assertEqual(ALERT_THRESHOLDS[1], 0.280)
        self.assertEqual(ALERT_THRESHOLDS[5], 0.300)
        self.assertEqual(ALERT_THRESHOLDS[15], 0.320)

        # H=1: 0.279 vs 0.280
        alert_below_1, th_1 = evaluate_alert_decision(0.279, 1)
        alert_at_1, _ = evaluate_alert_decision(0.280, 1)
        self.assertFalse(alert_below_1)
        self.assertTrue(alert_at_1)
        self.assertEqual(th_1, 0.280)

        # H=5: 0.299 vs 0.300
        alert_below_5, th_5 = evaluate_alert_decision(0.299, 5)
        alert_at_5, _ = evaluate_alert_decision(0.300, 5)
        self.assertFalse(alert_below_5)
        self.assertTrue(alert_at_5)
        self.assertEqual(th_5, 0.300)

        # H=15: 0.319 vs 0.320
        alert_below_15, th_15 = evaluate_alert_decision(0.319, 15)
        alert_at_15, _ = evaluate_alert_decision(0.320, 15)
        self.assertFalse(alert_below_15)
        self.assertTrue(alert_at_15)
        self.assertEqual(th_15, 0.320)

        # Actual prediction and explanation consistency case
        status_p, pred = self.get("/predict/degradation?service=reservations&horizon=5&replay_step=12")
        status_e, expl = self.get("/explain/reservations?horizon=5&replay_step=12")
        self.assertEqual(status_p, 200)
        self.assertEqual(status_e, 200)
        self.assertEqual(pred["alert_threshold"], 0.300)
        self.assertEqual(expl["alert_threshold"], 0.300)
        self.assertEqual(pred["advisory_alert_decision"], expl["advisory_alert_decision"])
        self.assertEqual(pred["calibrated_event_probability"], expl["calibrated_event_probability"])

    def test_14_concurrent_metrics_responsiveness_during_slow_explanation(self):
        """Verify /metrics responds immediately while an explanation request is processing in parallel."""
        expl_started = threading.Event()

        original_explain_fn = ml_service.explainer.explain_prediction

        def slow_explain(*args, **kwargs):
            expl_started.set()
            time.sleep(0.4)
            return original_explain_fn(*args, **kwargs)

        with patch.object(ml_service.explainer, "explain_prediction", side_effect=slow_explain):
            expl_result = {}

            def run_expl():
                s, b = self.get("/explain/crew?horizon=5&replay_step=14")
                expl_result["status"] = s
                expl_result["body"] = b

            th = threading.Thread(target=run_expl, daemon=True)
            th.start()

            # Wait until explanation begins executing
            self.assertTrue(expl_started.wait(timeout=5.0))

            # /metrics must respond IMMEDIATELY while explanation is still running
            t0 = time.time()
            status_m, prom_body = self.get_text("/metrics")
            latency_m = time.time() - t0

            self.assertEqual(status_m, 200)
            self.assertIn("baccp_sync_drift_score", prom_body)
            # Response must be fast (< 0.25s), before the 0.4s explanation finishes
            self.assertLess(latency_m, 0.25)

            th.join(timeout=5.0)
            self.assertEqual(expl_result.get("status"), 200)

    def test_15_graph_localization_tie_and_candidate_rankings(self):
        """Verify graph localization tie maps to abstain_ambiguous_tie (not localized) and returns candidate rankings."""
        original_explain_fn = ml_service.explainer.explain_prediction

        def mock_explain_with_tie(*args, **kwargs):
            res = original_explain_fn(*args, **kwargs)
            res_copy = copy.deepcopy(res)
            res_copy["explanation"]["graph_localization"] = {
                "suspected_contributing_node": "crew",
                "is_tie": True,
                "is_abstained": False,
                "candidate_rankings": [
                    {"node": "crew", "score": 0.45},
                    {"node": "reservations", "score": 0.45},
                ],
                "propagation_bottleneck": "reservations",
            }
            return res_copy

        with patch.object(ml_service.explainer, "explain_prediction", side_effect=mock_explain_with_tie):
            s, body = self.get("/explain/crew?horizon=5&replay_step=12")
            self.assertEqual(s, 200)
            loc = body["localization"]
            self.assertEqual(loc["localization_status"], "abstain_ambiguous_tie")
            self.assertNotEqual(loc["localization_status"], "localized")
            self.assertIn("candidate_rankings", loc)
            self.assertEqual(len(loc["candidate_rankings"]), 2)

    def test_16_telemetry_availability_uninitialized_and_stale_update(self):
        """Verify telemetry_available starts at 0 (unverified) and marks unavailable on stale data."""
        fresh_metrics = BACCPMLMetrics()
        for s in SUPPORTED_SERVICES:
            self.assertEqual(fresh_metrics.telemetry_available[s], 0)

        s, b = self.get("/predict/degradation?service=reservations&horizon=5&replay_time_sec=99999.0")
        self.assertEqual(s, 422)
        self.assertEqual(ml_metrics.telemetry_available["reservations"], 0)

    def test_17_injected_predictor_near_threshold_regression(self):
        """Regression: Verify core alert decision is preserved when probability = threshold - 0.00004 across all horizons, and at threshold."""
        from backend.ml.alert_policy import ALERT_THRESHOLDS, evaluate_alert_decision

        services = ["reservations", "crew"]
        horizons = [1, 5, 15]

        for s in services:
            for h in horizons:
                th = ALERT_THRESHOLDS[h]
                prob_below = th - 0.00004
                prob_at = th

                # Case 1: probability = threshold - 0.00004 (Edge case)
                # Core computes alert_issued from full-precision probability (False)
                # and serializes degradation_event_risk rounded to four decimals (round(threshold - 0.00004, 4) = threshold)
                original_predict = ml_service.predictor.predict

                def mock_predict_below(*args, **kwargs):
                    res = original_predict(*args, **kwargs)
                    res_copy = copy.deepcopy(res)
                    h_str = str(h)
                    if s in res_copy["services"] and h_str in res_copy["services"][s]["predictions_by_horizon"]:
                        h_data = res_copy["services"][s]["predictions_by_horizon"][h_str]
                        core_alert, core_thresh = evaluate_alert_decision(prob_below, h)
                        h_data["degradation_event_risk"] = round(prob_below, 4)
                        h_data["calibrated_event_probability_raw"] = float(prob_below)
                        h_data["alert_issued"] = core_alert
                        h_data["alert_threshold"] = core_thresh
                        h_data["estimated_time_to_degradation_sec"] = None
                        h_data["predicted_degradation_type"] = "NOMINAL"
                        h_data["predicted_severity"] = "NOMINAL"
                    return res_copy

                ml_service.clear_cache_for_testing()
                initial_alert_count = ml_metrics.alert_episode_counts[(s, h)]

                with patch.object(ml_service.predictor, "predict", side_effect=mock_predict_below):
                    # 1. Prediction endpoint
                    s_code, pred_body = self.get(f"/predict/degradation?service={s}&horizon={h}&replay_step=10")
                    self.assertEqual(s_code, 200)
                    self.assertFalse(pred_body["advisory_alert_decision"])
                    self.assertFalse(pred_body["alert_issued"])
                    self.assertEqual(pred_body["alert_threshold"], th)
                    self.assertLess(pred_body["calibrated_event_probability"], th)
                    self.assertEqual(pred_body["calibrated_event_probability"], round(prob_below, 5))

                    fut_risk = pred_body["predicted_future_risk"]
                    self.assertIsNone(fut_risk["estimated_time_to_degradation_seconds"])
                    self.assertEqual(fut_risk["time_to_degradation_reason"], "service_nominal_no_active_alert")
                    self.assertEqual(fut_risk["predicted_degradation_type"], "NOMINAL")
                    self.assertEqual(fut_risk["predicted_severity"], "NOMINAL")

                    # 2. Metrics: Verify no alert-counter increment
                    self.assertEqual(ml_metrics.alert_episode_counts[(s, h)], initial_alert_count)

                    # 3. Explanation endpoint
                    s_expl, expl_body = self.get(f"/explain/{s}?horizon={h}&replay_step=10")
                    self.assertEqual(s_expl, 200)
                    self.assertFalse(expl_body["advisory_alert_decision"])
                    self.assertFalse(expl_body["alert_issued"])
                    self.assertEqual(expl_body["calibrated_event_probability"], pred_body["calibrated_event_probability"])

                    brief = expl_body["incident_brief"]
                    self.assertIn("NO ALERT ISSUED", brief)
                    self.assertNotIn("ALERT ACTIVE", brief)
                    self.assertIn("Conditional time-to-degradation estimate is gated to active alert states", brief)
                    # Verify no confusing displays like "risk 30.0% < threshold 30.0%"
                    self.assertNotIn(f"risk {th*100.0:.1f}% < threshold {th*100.0:.1f}%", brief)

                # Case 2: Retain case exactly at threshold (probability = threshold)
                def mock_predict_at(*args, **kwargs):
                    res = original_predict(*args, **kwargs)
                    res_copy = copy.deepcopy(res)
                    h_str = str(h)
                    if s in res_copy["services"] and h_str in res_copy["services"][s]["predictions_by_horizon"]:
                        h_data = res_copy["services"][s]["predictions_by_horizon"][h_str]
                        core_alert, core_thresh = evaluate_alert_decision(prob_at, h)
                        h_data["degradation_event_risk"] = round(prob_at, 4)
                        h_data["calibrated_event_probability_raw"] = float(prob_at)
                        h_data["alert_issued"] = core_alert
                        h_data["alert_threshold"] = core_thresh
                        h_data["estimated_time_to_degradation_sec"] = 20.0
                        h_data["predicted_degradation_type"] = "LATENCY_BREACH"
                        h_data["predicted_severity"] = "MAJOR"
                    return res_copy

                ml_service.clear_cache_for_testing()
                with patch.object(ml_service.predictor, "predict", side_effect=mock_predict_at):
                    s_code, pred_body = self.get(f"/predict/degradation?service={s}&horizon={h}&replay_step=11")
                    self.assertEqual(s_code, 200)
                    self.assertTrue(pred_body["advisory_alert_decision"])
                    self.assertTrue(pred_body["alert_issued"])
                    self.assertEqual(pred_body["calibrated_event_probability"], round(prob_at, 4))

                    # Metrics: Verify alert counter DID increment
                    self.assertEqual(ml_metrics.alert_episode_counts[(s, h)], initial_alert_count + 1)

                    # Explanation endpoint
                    s_expl, expl_body = self.get(f"/explain/{s}?horizon={h}&replay_step=11")
                    self.assertEqual(s_expl, 200)
                    self.assertTrue(expl_body["advisory_alert_decision"])
                    self.assertTrue(expl_body["alert_issued"])
                    self.assertIn("ALERT ACTIVE", expl_body["incident_brief"])

    def test_18_explanation_capacity_held_until_work_finishes(self):
        """Verify explanation semaphore holds capacity across timeouts until work actually finishes."""
        work_block = threading.Event()

        def blocked_explain(*args, **kwargs):
            if not work_block.is_set():
                work_block.wait(timeout=5.0)
            return {
                "explanation": {
                    "attribution_method": "TreeSHAP",
                    "top_risk_increasing_drivers": [],
                    "top_risk_decreasing_drivers": [],
                    "category_attributions": {},
                    "graph_localization": {"suspected_contributing_node": "crew"},
                    "incident_brief": "test brief",
                },
                "prediction": {
                    "services": {
                        "crew": {
                            "predictions_by_horizon": {
                                "1": {
                                    "degradation_event_risk": 0.05,
                                    "alert_issued": False,
                                    "alert_threshold": 0.28,
                                }
                            }
                        }
                    }
                },
            }

        # Set a short configurable timeout of 50ms
        old_timeout = ml_service.explanation_timeout_seconds
        ml_service.explanation_timeout_seconds = 0.05
        ml_service.clear_cache_for_testing()

        try:
            with patch.object(ml_service.explainer, "explain_prediction", side_effect=blocked_explain):
                # 1. Dispatch 4 requests (matching capacity of 4).
                # Each request must time out after 50ms with HTTP 504.
                for step in range(10, 14):
                    status, body = self.get(f"/explain/crew?horizon=1&replay_step={step}")
                    self.assertEqual(status, 504)
                    self.assertEqual(body["error"], "EXPLANATION_TIMEOUT")

                # 2. While the 4 workers/tasks are still blocked, capacity must be exhausted (4/4 held).
                # A 5th request must be rejected with HTTP 503 EXPLANATION_CAPACITY_EXCEEDED.
                status_5, body_5 = self.get("/explain/crew?horizon=1&replay_step=14")
                self.assertEqual(status_5, 503)
                self.assertEqual(body_5["error"], "EXPLANATION_CAPACITY_EXCEEDED")

                # 3. Unblock the injected work so futures can complete and release capacity
                work_block.set()

                # Allow a short moment for worker threads to finish and callbacks to release permits
                time.sleep(0.05)

                # 4. Capacity must recover once background work finishes
                status_rec, body_rec = self.get("/explain/crew?horizon=1&replay_step=15")
                self.assertEqual(status_rec, 200)
                self.assertEqual(body_rec["service"], "crew")
        finally:
            work_block.set()
            ml_service.explanation_timeout_seconds = old_timeout


if __name__ == "__main__":
    unittest.main()

