"""Integration tests for airline demand forecasting and workload generation with the BACCP pipeline.

Validates the full data flow:
Historical BTS Demand Data
    ↓
Demand Forecasting
    ↓
Forecasted Demand
    ↓
Airline Workload Generator
    ↓
Workload / Resource-Demand Observations
    ↓
BACCP Cloud Telemetry, SageMaker Prediction & Risk Evaluation
    ↓
Lambda Circuit Breaker Mitigation & SNS Alerts

Covers:
1. REST API endpoints: GET/POST /api/forecast, POST /api/workload/generate, POST /api/forecast/pipeline.
2. Metric publishing to CloudWatch buffer (without requiring live AWS credentials).
3. Distributed trace segment recording in X-Ray buffer.
4. Nominal scenario (low risk, circuit remains CLOSED).
5. Cascade failure scenario (fault injected, high risk, circuit actuation OPEN/THROTTLED, SNS alert emitted).
"""

from __future__ import annotations

from http.server import HTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from urllib.request import Request, urlopen

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.app import BACCPRequestHandler
from backend.cloud.cloudwatch import CloudWatchPublisher
from backend.cloud.config import AWSConfig
from backend.cloud.lambda_handler import breaker_manager
from backend.cloud.orchestrator import BACCPCloudOrchestrator
from backend.cloud.xray import XRayTraceRecorder
from workload.forecasting import DemandToWorkloadMapper


class TestForecastCloudPipelineIntegration(unittest.TestCase):
    """Integration test suite connecting forecasting and workload generation to cloud pipeline."""

    @classmethod
    def setUpClass(cls):
        """Start local HTTP server in a background thread."""
        cls.httpd = HTTPServer(("127.0.0.1", 0), BACCPRequestHandler)
        cls.port = cls.httpd.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        """Shutdown HTTP server."""
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        # Reset breaker state before each test
        breaker_manager.apply_action("CLOSED", 0.0, "Reset to closed", caller="test")

    def _get(self, path: str):
        req = Request(f"{self.base_url}{path}")
        with urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = Request(f"{self.base_url}{path}", data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def test_get_forecast_endpoint(self):
        """Verify GET /api/forecast returns evaluation metrics, chronological splits, and forecast series."""
        res = self._get("/api/forecast")
        self.assertIn("target_variable", res)
        self.assertIn("models_evaluated", res)
        self.assertIn("chronological_splits", res)
        self.assertIn("sample_test_forecasts", res)
        self.assertEqual(res["target_variable"], "enplaned_passengers")

    def test_post_forecast_endpoint(self):
        """Verify POST /api/forecast generates fresh forecasts and comparative metrics."""
        res = self._post("/api/forecast", {"horizon_steps": 14})
        self.assertIn("models_evaluated", res)
        models = res["models_evaluated"]
        self.assertIn("LightweightAutoRegressiveForecaster_Ridge", models)
        self.assertIn("mae", models["LightweightAutoRegressiveForecaster_Ridge"])

    def test_post_workload_generate_from_forecast(self):
        """Verify POST /api/workload/generate with forecasted_passengers produces calibrated observations."""
        payload = {
            "scenario_name": "forecast_driven_test",
            "forecasted_passengers": 2_450_000.0,
            "duration_seconds": 60.0,
            "sampling_interval_sec": 20.0,
            "random_seed": 42,
        }
        res = self._post("/api/workload/generate", payload)
        self.assertEqual(res["scenario"], "forecast_driven_test")
        self.assertEqual(res["snapshot_count"], 3)
        self.assertEqual(len(res["snapshots"]), 3)

        snap_0 = res["snapshots"][0]
        self.assertIn("observations", snap_0)
        obs_res = snap_0["observations"]["reservations"]
        
        # Verify fields and provenance preservation
        self.assertEqual(obs_res["service_id"], "reservations")
        self.assertGreater(obs_res["request_rate"], 20.0)
        self.assertIn("provenance", obs_res)
        self.assertEqual(
            obs_res["provenance"]["forecast_values"]["forecasted_daily_passengers"],
            2_450_000.0
        )

    def test_post_workload_generate_custom_spike(self):
        """Verify POST /api/workload/generate without forecast produces valid spike scenario."""
        payload = {
            "scenario_name": "spike_api_test",
            "duration_seconds": 80.0,
            "sampling_interval_sec": 20.0,
            "spike_start_sec": 20.0,
            "spike_magnitude": 3.0,
            "random_seed": 42,
        }
        res = self._post("/api/workload/generate", payload)
        self.assertEqual(res["snapshot_count"], 4)
        snap_2 = res["snapshots"][2]  # at t=40s (inside sustained spike plateau)
        self.assertGreater(snap_2["observations"]["reservations"]["request_rate"], 50.0)

    def test_full_pipeline_endpoint_nominal(self):
        """Verify POST /api/forecast/pipeline executes the entire integrated pipeline under nominal conditions."""
        payload = {
            "forecasted_passengers": 2_150_000.0,
            "scenario_name": "nominal_forecast_pipeline",
            "duration_seconds": 60.0,
            "sampling_interval_sec": 20.0,
        }
        res = self._post("/api/forecast/pipeline", payload)
        self.assertEqual(res["status"], "success")
        self.assertIn("forecast", res)
        self.assertIn("workload", res)
        self.assertIn("pipeline_execution", res)

        pipe = res["pipeline_execution"]
        self.assertEqual(pipe["status"], "success")
        self.assertIn("telemetry", pipe)
        self.assertIn("prediction", pipe)
        self.assertIn("risk_evaluation", pipe)
        self.assertEqual(pipe["risk_evaluation"]["recommended_action"], "CLOSED")
        self.assertFalse(pipe["risk_evaluation"]["is_critical"])

    def test_full_pipeline_cascade_mitigation_under_fault(self):
        """Verify that when a chaos fault is present, forecasted workload triggers circuit breaker & alerts."""
        orchestrator = BACCPCloudOrchestrator()
        
        # Run forecast-driven pipeline with high network delay fault injected
        res = orchestrator.run_forecast_workload_pipeline(
            forecasted_passengers=2_600_000.0,
            scenario_name="critical_forecast_pipeline",
            duration_seconds=120.0,
            sampling_interval_sec=20.0,
            active_fault="network-delay",
            fault_level="high",  # 1500ms delay in profiles.json
        )

        self.assertEqual(res["status"], "success")
        pipe = res["pipeline_execution"]
        
        # Telemetry should reflect elevated sync-drift
        drift = pipe["telemetry"]["sync_drift_score"]
        self.assertGreaterEqual(drift, 45.0)

        # Risk evaluation should recommend mitigation
        risk = pipe["risk_evaluation"]
        self.assertIn(risk["recommended_action"], {"THROTTLE", "OPEN"})
        
        # Mitigation was executed via Lambda client
        self.assertIn("mitigation", pipe)
        self.assertIn(pipe["mitigation"]["status"], {"success", "simulated", "applied"})

        # SNS alert was published
        self.assertIn("sns_alert", pipe)
        self.assertIsNotNone(pipe["sns_alert"])
        self.assertEqual(pipe["sns_alert"]["severity"], "CRITICAL")
        self.assertIn(pipe["sns_alert"]["mitigation_status"], {"THROTTLE", "OPEN"})

    def test_cloudwatch_and_xray_recording(self):
        """Verify that CloudWatch metric buffer and X-Ray trace recorder capture telemetry locally."""
        mock_config = AWSConfig(cloud_mode="local")
        cw = CloudWatchPublisher(aws_config=mock_config)
        xray = XRayTraceRecorder(aws_config=mock_config)
        orchestrator = BACCPCloudOrchestrator(cloudwatch=cw, xray=xray)

        # Confirm non-live / local mode
        self.assertFalse(cw.is_live)
        self.assertFalse(xray.is_live)

        initial_metrics_count = len(cw.published_buffer)
        initial_traces_count = len(xray.recorded_traces)

        # Run integrated pipeline
        res = orchestrator.run_forecast_workload_pipeline(
            forecasted_passengers=2_200_000.0,
            scenario_name="telemetry_test",
        )
        self.assertEqual(res["status"], "success")

        # Verify CloudWatch buffered metrics
        self.assertGreater(len(cw.published_buffer), initial_metrics_count)
        metric_names = {m["MetricName"] for m in cw.published_buffer}
        self.assertIn("BoundarySyncDrift", metric_names)
        self.assertIn("CascadeProbability", metric_names)

        # Verify X-Ray recorded distributed traces including demand_forecasting and workload_generation
        self.assertGreater(len(xray.recorded_traces), initial_traces_count)
        trace_names = {t["name"] for t in xray.recorded_traces}
        self.assertTrue(any("demand_forecasting" in name for name in trace_names))
        self.assertTrue(any("workload_generation" in name for name in trace_names))


if __name__ == "__main__":
    unittest.main()
