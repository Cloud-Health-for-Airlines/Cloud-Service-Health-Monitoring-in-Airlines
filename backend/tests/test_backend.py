"""Unit test suite for BACCP Backend REST API and AWS Cloud Integration.

Covers:
- Cloud configuration (local vs AWS mode, default values, env overrides)
- CloudWatch metric publishing (all 8 metrics)
- X-Ray context tracing (all 6 operations, graceful failure handling)
- SageMaker model adapter (input/output contract, conformal bounds, local fallback)
- Lambda mitigation client (structured payload, local simulation, breaker updates)
- SNS alert publisher (8 required attributes, local buffering)
- Cloud orchestrator end-to-end pipeline
- Canonical graph and boundary health computation
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from backend.cloud.cloudwatch import CloudWatchPublisher
from backend.cloud.config import AWSConfig, BackendConfig
from backend.cloud.lambda_handler import CircuitBreakerManager, LambdaMitigationClient, lambda_handler
from backend.cloud.orchestrator import BACCPCloudOrchestrator
from backend.cloud.sagemaker import SageMakerCascadePredictor
from backend.cloud.sns import SNSPublisher
from backend.cloud.xray import XRayTraceRecorder, generate_segment_id, generate_trace_id


class TestCloudConfiguration(unittest.TestCase):
    """Test cloud configuration loading, mode resolution, and security defaults."""

    def test_default_config_local_mode(self):
        cfg = BackendConfig()
        self.assertEqual(cfg.port, 8000)
        self.assertEqual(cfg.aws.cloud_mode, "local")
        self.assertEqual(cfg.aws.mode, "local")
        self.assertFalse(cfg.aws.is_live)
        self.assertEqual(cfg.aws.region_name, "us-east-1")
        self.assertEqual(cfg.aws.cloudwatch_namespace, "BACCP/AirlineCloudHealth")

    def test_custom_aws_config_attributes(self):
        custom_cfg = AWSConfig(
            cloud_mode="aws",
            region_name="eu-west-1",
            cloudwatch_namespace="BACCP/CustomTest",
            sagemaker_endpoint_name="test-sagemaker-model",
            lambda_breaker_function="test-breaker-lambda",
            sns_topic_arn="arn:aws:sns:eu-west-1:111122223333:test-topic",
            xray_enabled=True,
        )
        self.assertEqual(custom_cfg.cloud_mode, "aws")
        self.assertTrue(custom_cfg.is_live)
        self.assertEqual(custom_cfg.region_name, "eu-west-1")
        self.assertEqual(custom_cfg.cloudwatch_namespace, "BACCP/CustomTest")
        self.assertTrue(custom_cfg.xray_enabled)


class TestCloudWatchAdapter(unittest.TestCase):
    """Test CloudWatch metric publishing across all 8 required BACCP metrics."""

    def setUp(self):
        self.cw = CloudWatchPublisher(aws_config=AWSConfig(cloud_mode="local"))

    def test_local_mode_initialization(self):
        self.assertFalse(self.cw.is_live)
        self.assertEqual(self.cw.namespace, "BACCP/AirlineCloudHealth")

    def test_publish_all_eight_metrics(self):
        """Test publishing all 8 required BACCP metrics."""
        metrics = {
            "boundary_health_score": 38.5,
            "cascade_probability": 0.42,
            "prediction_lead_time": 180.0,
            "service_latency": 45.2,
            "error_rate": 1.5,
            "request_rate": 35.0,
            "circuit_breaker_actions": 1.0,
            "active_alerts": 2.0,
        }
        ok = self.cw.publish_baccp_metrics(metrics, dimensions={"Environment": "Testbed"})
        self.assertTrue(ok)
        self.assertEqual(len(self.cw.published_buffer), 8)

        # Verify recent metrics serialization
        recent = self.cw.get_recent_metrics(count=8)
        self.assertEqual(len(recent), 8)
        names = {m["MetricName"] for m in recent}
        expected_names = set(metrics.keys())
        self.assertEqual(names, expected_names)

        # Check units
        unit_map = {m["MetricName"]: m["Unit"] for m in recent}
        self.assertEqual(unit_map["boundary_health_score"], "Percent")
        self.assertEqual(unit_map["service_latency"], "Milliseconds")
        self.assertEqual(unit_map["prediction_lead_time"], "Seconds")
        self.assertEqual(unit_map["cascade_probability"], "None")

    def test_publish_helpers(self):
        self.cw.publish_boundary_sync_drift(55.0)
        self.cw.publish_cascade_probability(probability=0.88, lead_time_sec=25.0, root_node="boundary-gateway")
        self.cw.publish_service_telemetry(service="reservations", latency_ms=18.2, error_rate=0.1, request_rate=22.0)
        self.cw.publish_circuit_breaker_action("THROTTLED", throttle_rate=50.0)
        self.cw.publish_active_alerts(1)
        self.assertGreater(len(self.cw.published_buffer), 5)


class TestXRayTracing(unittest.TestCase):
    """Test AWS X-Ray distributed tracing and operation context managers."""

    def setUp(self):
        self.xray = XRayTraceRecorder(aws_config=AWSConfig(cloud_mode="local", xray_enabled=False))

    def test_id_generation(self):
        trace_id = generate_trace_id()
        self.assertTrue(trace_id.startswith("1-"))
        segment_id = generate_segment_id()
        self.assertEqual(len(segment_id), 16)

    def test_trace_operations_coverage(self):
        """Test tracing across all 6 requested operations."""
        operations = [
            "incoming_api_request",
            "telemetry_processing",
            "graph_retrieval",
            "prediction_request",
            "alert_generation",
            "circuit_breaker_action",
        ]
        for op in operations:
            with self.xray.trace_operation(op, metadata={"test": True}) as segment:
                self.assertEqual(segment["name"], f"baccp:{op}")
                self.assertIn("trace_id", segment)
                self.assertIn("start_time", segment)

        self.assertEqual(len(self.xray.recorded_traces), len(operations))
        for trace in self.xray.recorded_traces:
            self.assertEqual(trace["status"], "ok")
            self.assertGreaterEqual(trace["duration_ms"], 0.0)

    def test_trace_operation_exception_handling(self):
        """Test that exceptions inside trace_operation record error state without crashing tracer."""
        with self.assertRaises(ValueError):
            with self.xray.trace_operation("telemetry_processing") as _:
                raise ValueError("Simulated processing fault")

        last_trace = self.xray.recorded_traces[-1]
        self.assertEqual(last_trace["status"], "error")
        self.assertTrue(last_trace.get("error"))

    def test_cross_boundary_trace_synthesis(self):
        trace = self.xray.record_cross_boundary_trace("reservations", duration_ms=40.0)
        self.assertEqual(trace["name"], "reservations")
        self.assertEqual(trace["annotations"]["generation_type"], "cloud-native")
        gateway = trace["subsegments"][0]
        self.assertEqual(gateway["name"], "boundary-gateway")
        legacy = gateway["subsegments"][0]
        self.assertEqual(legacy["name"], "legacy-core")


class TestSageMakerCascadePredictor(unittest.TestCase):
    """Test SageMaker prediction adapter input/output interface and conformal bounds."""

    def setUp(self):
        self.predictor = SageMakerCascadePredictor(aws_config=AWSConfig(cloud_mode="local"))

    def test_local_fallback_nominal(self):
        """Test nominal conditions produce low probability and long lead time."""
        res = self.predictor.predict_cascade(
            graph_data={"nodes": [], "edges": []},
            sync_drift_score=15.0,
            telemetry_features={"reservations": {"latency_ms": 15.0}},
            boundary_features={"sync_drift_score": 15.0},
        )
        self.assertIn("cascade_probability", res)
        self.assertIn("predicted_failure_location", res)
        self.assertIn("severity", res)
        self.assertIn("lead_time", res)
        self.assertIn("confidence", res)
        self.assertIn("conformal_bounds", res)

        self.assertLess(res["cascade_probability"], 0.25)
        self.assertEqual(res["severity"], "LOW")
        self.assertGreater(res["lead_time"], 180.0)
        self.assertEqual(res["conformal_bounds"]["confidence_level"], 0.90)
        self.assertLessEqual(res["conformal_bounds"]["lower"], res["cascade_probability"])
        self.assertGreaterEqual(res["conformal_bounds"]["upper"], res["cascade_probability"])
        self.assertEqual(res["model_metadata"]["mode"], "local-analytical")

    def test_local_fallback_critical_fault(self):
        """Test high drift and injected fault produce high probability and critical alert."""
        res = self.predictor.predict_cascade(
            graph_data={},
            sync_drift_score=88.0,
            active_fault="connection-drop",
            fault_level="high",
        )
        self.assertGreater(res["cascade_probability"], 0.80)
        self.assertIn(res["severity"], ("HIGH", "CRITICAL"))
        self.assertLess(res["lead_time"], 60.0)
        self.assertEqual(res["predicted_failure_location"], "boundary-gateway")


class TestLambdaCircuitBreaker(unittest.TestCase):
    """Test Lambda integration client, structured mitigation payload, and simulation."""

    def setUp(self):
        self.manager = CircuitBreakerManager()
        self.manager.apply_action("CLOSED", 0.0, "Test reset")
        self.client = LambdaMitigationClient(aws_config=AWSConfig(cloud_mode="local"))

    def test_structured_mitigation_payload_and_simulation(self):
        result = self.client.invoke_mitigation(
            affected_service="reservations",
            gateway="boundary-gateway",
            cascade_probability=0.88,
            severity="CRITICAL",
            recommended_action="OPEN",
            sync_drift_score=82.5,
            lead_time=35.0,
        )

        self.assertEqual(result["statusCode"], 200)
        self.assertTrue(result["simulated"])
        self.assertIn("Simulated local mitigation", result["note"])

        # Check payload fields
        payload = result["payload"]
        self.assertEqual(payload["event_type"], "cascade_mitigation")
        self.assertEqual(payload["affected_service"], "reservations")
        self.assertEqual(payload["gateway"], "boundary-gateway")
        self.assertEqual(payload["cascade_probability"], 0.88)
        self.assertEqual(payload["severity"], "CRITICAL")
        self.assertEqual(payload["recommended_action"], "OPEN")
        self.assertIn("timestamp", payload)

        # Check breaker state updated
        status = self.manager.get_status()
        self.assertEqual(status["state"], "OPEN")
        self.assertEqual(status["throttle_rate"], 1.0)

        # Check history recorded
        history = self.client.get_invocation_history()
        self.assertGreater(len(history), 0)


class TestSNSPublisher(unittest.TestCase):
    """Test SNS alert publisher and high-severity predictive cascade alert schema."""

    def setUp(self):
        self.sns = SNSPublisher(aws_config=AWSConfig(cloud_mode="local"))

    def test_publish_cascade_alert_attributes(self):
        alert = self.sns.publish_cascade_alert(
            severity="HIGH",
            cascade_probability=0.76,
            affected_service="reservations",
            gateway="boundary-gateway",
            predicted_failure_location="boundary-gateway",
            lead_time=58.0,
            mitigation_status="THROTTLED",
            extra_details="Test cascade explanation",
        )

        # Verify all 8 required fields
        self.assertEqual(alert["severity"], "HIGH")
        self.assertEqual(alert["cascade_probability"], 0.76)
        self.assertEqual(alert["affected_service"], "reservations")
        self.assertEqual(alert["boundary/gateway"], "boundary-gateway")
        self.assertEqual(alert["gateway"], "boundary-gateway")
        self.assertEqual(alert["predicted_failure_location"], "boundary-gateway")
        self.assertEqual(alert["lead_time"], 58.0)
        self.assertEqual(alert["mitigation_status"], "THROTTLED")
        self.assertIn("timestamp", alert)

        # Verify buffering
        recent = self.sns.get_recent_alerts(count=1)
        self.assertEqual(len(recent), 1)
        self.assertTrue(recent[0]["simulated"])


class TestCloudOrchestrator(unittest.TestCase):
    """Test BACCP Cloud Orchestrator end-to-end pipeline coordination."""

    def setUp(self):
        self.orchestrator = BACCPCloudOrchestrator(aws_config=AWSConfig(cloud_mode="local"))

    def test_pipeline_nominal_flow(self):
        """Under low drift, pipeline predicts nominal and does not mitigate."""
        report = self.orchestrator.run_pipeline(sync_drift_score=14.0)
        self.assertEqual(report["status"], "success")
        self.assertEqual(report["risk_evaluation"]["recommended_action"], "CLOSED")
        self.assertIsNone(report.get("sns_alert"))
        self.assertGreaterEqual(report["duration_ms"], 0.0)

    def test_pipeline_critical_flow(self):
        """Under high drift / fault, pipeline triggers Lambda mitigation and SNS alert."""
        report = self.orchestrator.run_pipeline(
            sync_drift_score=85.0,
            active_fault="network-delay",
            fault_level="high",
        )
        self.assertEqual(report["status"], "success")
        self.assertIn(report["risk_evaluation"]["recommended_action"], ("THROTTLE", "OPEN"))
        self.assertIsNotNone(report.get("mitigation"))
        self.assertIsNotNone(report.get("sns_alert"))
        self.assertIn("prediction", report)
        self.assertIn("distributed_trace_id", report)


class TestCanonicalGraphAndHealth(unittest.TestCase):
    """Test API helper functions."""

    def test_canonical_graph_structure(self):
        from backend.api.app import load_canonical_graph
        graph = load_canonical_graph()
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)
        node_ids = {n["id"] for n in graph["nodes"]}
        expected_nodes = {"reservations", "crew", "baggage", "boundary-gateway", "legacy-core"}
        self.assertEqual(node_ids, expected_nodes)

        node_types = {n["id"]: n["node_type"] for n in graph["nodes"]}
        self.assertEqual(node_types["reservations"], "cloud-native")
        self.assertEqual(node_types["boundary-gateway"], "boundary-gateway")
        self.assertEqual(node_types["legacy-core"], "legacy")

    def test_boundary_health_computation(self):
        from backend.api.app import compute_boundary_health, simulated_state
        simulated_state["active_fault"] = None
        health_baseline = compute_boundary_health()
        self.assertEqual(health_baseline["status"], "HEALTHY")
        self.assertLess(health_baseline["sync_drift_score"], 20.0)

        # Simulate fault
        simulated_state["active_fault"] = "connection-drop"
        simulated_state["fault_level"] = "high"
        health_fault = compute_boundary_health()
        self.assertIn(health_fault["status"], ("DEGRADED", "CRITICAL"))
        self.assertGreater(health_fault["sync_drift_score"], 60.0)

        # Clear
        simulated_state["active_fault"] = None
        simulated_state["fault_level"] = None


class TestRESTServerEndpoints(unittest.TestCase):
    """Integration test verifying REST endpoints over HTTP."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import HTTPServer
        from backend.api.app import BACCPRequestHandler

        cls.httpd = HTTPServer(("127.0.0.1", 0), BACCPRequestHandler)
        cls.port = cls.httpd.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _get(self, path: str):
        from urllib.request import Request, urlopen
        req = Request(f"{self.base_url}{path}")
        with urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: dict):
        from urllib.request import Request, urlopen
        data = json.dumps(payload).encode("utf-8")
        req = Request(f"{self.base_url}{path}", data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def test_get_cloud_status(self):
        res = self._get("/api/cloud/status")
        self.assertIn("cloud_mode", res)
        self.assertIn("cloudwatch", res)
        self.assertIn("xray", res)
        self.assertIn("sagemaker", res)
        self.assertIn("lambda", res)
        self.assertIn("sns", res)

    def test_get_boundary_health(self):
        res = self._get("/api/health/boundary")
        self.assertIn("sync_drift_score", res)
        self.assertIn("status", res)
        self.assertIn("formula", res)

    def test_post_predict(self):
        res = self._post("/api/predict", {"sync_drift_score": 75.0, "active_fault": "network-delay"})
        self.assertIn("cascade_probability", res)
        self.assertIn("predicted_failure_location", res)
        self.assertIn("lead_time", res)
        self.assertIn("conformal_bounds", res)
        self.assertGreater(res["cascade_probability"], 0.60)

    def test_post_circuit_breaker_action(self):
        res = self._post("/api/circuit-breaker/action", {
            "action": "THROTTLE",
            "throttle_rate": 0.45,
            "reason": "Test mitigation from integration test",
        })
        self.assertEqual(res["status"], "applied")
        self.assertIn("action_result", res)
        self.assertTrue(res["action_result"]["simulated"])

    def test_post_pipeline_run(self):
        res = self._post("/api/pipeline/run", {"sync_drift_score": 15.0})
        self.assertEqual(res["status"], "success")
        self.assertIn("duration_ms", res)
        self.assertIn("telemetry", res)
        self.assertIn("prediction", res)


if __name__ == "__main__":
    unittest.main()

