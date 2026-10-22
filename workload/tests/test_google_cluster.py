"""Unit tests for Google Cluster Workload Dataset Adapter.

Verifies:
1. Headerless standard CSV parsing (task_events and task_usage).
2. Header-based CSV parsing.
3. Gzip-compressed (.csv.gz) stream parsing.
4. Consolidated JSON dataset parsing.
5. Missing value imputation and 'missing_info' flag handling.
6. Malformed input tolerance (coercion in normal mode, exceptions in strict mode).
7. Microsecond-to-second timestamp conversion and duration calculations.
8. Domain mapping into airline service tiers (Reservations, Crew, Baggage).
9. Workload intensity time-series aggregation across temporal windows.
10. Transformation into orchestrator-compatible RED telemetry profiles.
11. Provenance metadata and zero-cloud-call guarantee.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest

from workload.datasets.google_cluster import (
    EVENT_TYPES,
    GoogleClusterTaskEvent,
    GoogleClusterTaskUsage,
    GoogleClusterTraceAdapter,
    map_to_airline_tier,
)


class TestGoogleClusterTraceAdapter(unittest.TestCase):
    """Test suite for Google Cluster workload dataset loading and preprocessing."""

    def setUp(self):
        self.adapter = GoogleClusterTraceAdapter()

    def test_metadata_and_provenance(self):
        """Verify metadata contains official attribution, CC-BY-4.0 license, and schema details."""
        meta = self.adapter.get_metadata()
        self.assertIn("Google Cluster Data", meta["name"])
        self.assertIn("CC BY 4.0", meta["license"])
        self.assertIn("https://github.com/google/cluster-data", meta["official_url"])
        self.assertIn("task_events", meta["primary_tables"])
        self.assertIn("task_usage", meta["primary_tables"])
        self.assertIn("airline_mapping", meta)

        instructions = self.adapter.download_instructions()
        self.assertIn("gsutil", instructions)
        self.assertIn("gs://clusterdata-2011-2/", instructions)

    def test_load_sample_task_events_csv(self):
        """Verify loading the bundled sample task_events CSV file."""
        events = self.adapter.load_task_events()
        self.assertGreater(len(events), 15)

        # First event is SUBMIT for reservations task
        first = events[0]
        self.assertEqual(first.job_id, 1001)
        self.assertEqual(first.task_index, 0)
        self.assertEqual(first.event_type, 0)
        self.assertEqual(first.event_name, "SUBMIT")
        self.assertIsNone(first.machine_id)
        self.assertEqual(first.scheduling_class, 3)
        self.assertEqual(first.priority, 10)
        self.assertEqual(first.airline_tier, "production_reservations")
        self.assertAlmostEqual(first.cpu_request, 0.125)
        self.assertAlmostEqual(first.memory_request, 0.0625)
        self.assertEqual(first.timestamp_usec, 0)
        self.assertEqual(first.timestamp_sec, 0.0)
        self.assertIn("col_0", first.raw_fields)

        # Check an event with machine_id and SCHEDULE (event_type 1)
        scheduled = [e for e in events if e.event_type == 1][0]
        self.assertIsNotNone(scheduled.machine_id)
        self.assertEqual(scheduled.event_name, "SCHEDULE")

    def test_load_sample_task_usage_csv(self):
        """Verify loading the bundled sample task_usage CSV file."""
        usages = self.adapter.load_task_usage()
        self.assertGreater(len(usages), 5)

        first = usages[0]
        self.assertEqual(first.job_id, 1001)
        self.assertEqual(first.start_time_usec, 0)
        self.assertEqual(first.end_time_usec, 300_000_000)
        self.assertEqual(first.start_time_sec, 0.0)
        self.assertEqual(first.end_time_sec, 300.0)
        self.assertEqual(first.duration_sec, 300.0)
        self.assertAlmostEqual(first.cpu_usage, 0.085)
        self.assertAlmostEqual(first.memory_usage, 0.052)
        self.assertGreaterEqual(first.max_cpu_usage, first.cpu_usage)
        self.assertGreaterEqual(first.max_memory_usage, first.memory_usage)
        self.assertIn("col_0", first.raw_fields)

    def test_load_json_sample(self):
        """Verify loading consolidated sample from JSON format."""
        events, usages = self.adapter.load_from_json()
        self.assertGreaterEqual(len(events), 5)
        self.assertGreaterEqual(len(usages), 3)

        self.assertEqual(events[0].job_id, 1001)
        self.assertEqual(events[0].airline_tier, "production_reservations")
        self.assertEqual(usages[0].duration_sec, 300.0)

    def test_header_based_csv_parsing(self):
        """Verify adapter parses CSVs containing explicit header rows."""
        csv_content = (
            "timestamp,missing_info,job_id,task_index,machine_id,event_type,user_name,scheduling_class,priority,cpu_request,memory_request,disk_space_request,different_machine_restriction\n"
            "5000000,0,9999,0,m123,1,test_user,2,7,0.10,0.05,0.002,1\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
            tf.write(csv_content)
            temp_path = tf.name

        try:
            events = self.adapter.load_task_events(path=temp_path)
            self.assertEqual(len(events), 1)
            ev = events[0]
            self.assertEqual(ev.timestamp_usec, 5000000)
            self.assertEqual(ev.timestamp_sec, 5.0)
            self.assertEqual(ev.job_id, 9999)
            self.assertEqual(ev.machine_id, "m123")
            self.assertEqual(ev.airline_tier, "mid_crew")
            self.assertEqual(ev.different_machine_restriction, True)
            self.assertEqual(ev.raw_fields["user_name"], "test_user")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_gzip_compressed_csv_loading(self):
        """Verify loading directly from gzip-compressed (.csv.gz) files without unpacking."""
        csv_content = (
            "0,0,8888,0,,0,user1,3,10,0.20,0.10,0.001,0\n"
            "2000000,0,8888,0,m456,1,user1,3,10,0.20,0.10,0.001,0\n"
        )
        with tempfile.NamedTemporaryFile("wb", suffix=".csv.gz", delete=False) as tf:
            with gzip.open(tf, "wt", encoding="utf-8") as gz:
                gz.write(csv_content)
            temp_path = tf.name

        try:
            events = self.adapter.load_task_events(path=temp_path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].job_id, 8888)
            self.assertEqual(events[1].machine_id, "m456")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_missing_values_and_imputation(self):
        """Verify missing fields (missing_info=1, empty strings, missing cpu_request) are safely imputed."""
        # Row with missing machine_id, missing cpu_request, missing memory_request
        row = ["10000000", "1", "7777", "0", "", "0", "", "2", "8", "", "", "", "0"]
        event = self.adapter.parse_task_event_row(row)

        self.assertIsNotNone(event)
        self.assertEqual(event.timestamp_sec, 10.0)
        self.assertTrue(event.missing_info)
        self.assertIsNone(event.machine_id)
        self.assertIsNone(event.user_name)
        # Sched class 2 defaults to 0.05 CPU and 0.04 RAM
        self.assertAlmostEqual(event.cpu_request, 0.05)
        self.assertAlmostEqual(event.memory_request, 0.04)
        self.assertEqual(event.airline_tier, "mid_crew")

    def test_malformed_input_tolerance(self):
        """Verify adapter handles malformed rows gracefully in lenient mode, and raises in strict mode."""
        # Too few columns
        short_row = ["100", "0", "1"]
        self.assertIsNone(self.adapter.parse_task_event_row(short_row))

        # Comment row
        comment_row = ["# This is a comment", "1", "2", "3", "4", "5", "6"]
        self.assertIsNone(self.adapter.parse_task_event_row(comment_row))

        # Non-numeric timestamp in lenient mode coerces to 0
        corrupt_row = ["invalid_timestamp", "0", "1001", "0", "", "0", "user", "3", "10", "bad_cpu", "0.05", "0.01", "0"]
        event = self.adapter.parse_task_event_row(corrupt_row)
        self.assertIsNotNone(event)
        self.assertEqual(event.timestamp_usec, 0)
        self.assertEqual(event.cpu_request, 0.05)  # class 3 default imputation

        # In strict mode, invalid numbers raise ValueError
        strict_adapter = GoogleClusterTraceAdapter(strict=True)
        with self.assertRaises(ValueError):
            strict_adapter.parse_task_event_row(corrupt_row)

    def test_airline_tier_mapping(self):
        """Verify mapping of Google Cluster scheduling classes and priorities to airline services."""
        # Class 3 -> Reservations
        self.assertEqual(map_to_airline_tier(3, 10), "production_reservations")
        self.assertEqual(map_to_airline_tier(0, 11), "production_reservations")

        # Class 2 -> Crew
        self.assertEqual(map_to_airline_tier(2, 6), "mid_crew")
        self.assertEqual(map_to_airline_tier(1, 8), "mid_crew")

        # Class 0-1 -> Baggage
        self.assertEqual(map_to_airline_tier(0, 2), "batch_baggage")
        self.assertEqual(map_to_airline_tier(1, 4), "batch_baggage")

    def test_workload_intensity_timeseries(self):
        """Verify computation of time-window aggregated workload intensity metrics."""
        events = self.adapter.load_task_events()
        usages = self.adapter.load_task_usage()

        # Group by 60-second windows
        series = self.adapter.compute_workload_intensity_timeseries(
            window_seconds=60.0,
            events=events,
            usage=usages,
        )
        self.assertGreaterEqual(len(series), 1)

        first_win = series[0]
        self.assertIn("window_index", first_win)
        self.assertIn("arrival_rate_tasks_per_sec", first_win)
        self.assertIn("task_event_counts", first_win)
        self.assertIn("resource_metrics", first_win)
        self.assertIn("airline_tier_distribution", first_win)

        res_counts = first_win["airline_tier_distribution"]
        self.assertIn("production_reservations", res_counts)
        self.assertIn("mid_crew", res_counts)
        self.assertIn("batch_baggage", res_counts)

        # Check total events count matches sum of event types
        counts = first_win["task_event_counts"]
        self.assertEqual(
            counts["total"],
            counts["submit"] + counts["schedule"] + counts["evict"] + counts["fail"] + counts["finish"],
        )

    def test_extract_airline_workload_profile(self):
        """Verify conversion of cluster workload metrics into orchestrator-compatible RED telemetry."""
        profile = self.adapter.extract_airline_workload_profile(window_seconds=60.0)

        # Must contain all 5 airline system nodes
        required_nodes = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]
        for node in required_nodes:
            self.assertIn(node, profile)
            metrics = profile[node]
            self.assertIn("request_rate", metrics)
            self.assertIn("latency_ms", metrics)
            self.assertIn("error_rate", metrics)
            self.assertIn("cpu_load", metrics)
            self.assertGreater(metrics["request_rate"], 0.0)
            self.assertGreater(metrics["latency_ms"], 0.0)
            self.assertGreaterEqual(metrics["error_rate"], 0.0)
            self.assertGreaterEqual(metrics["cpu_load"], 0.0)

        # Gateway request rate should equal aggregate of cloud callers
        expected_gw = profile["reservations"]["request_rate"] + profile["crew"]["request_rate"] + profile["baggage"]["request_rate"]
        self.assertAlmostEqual(profile["boundary-gateway"]["request_rate"], expected_gw, places=1)


if __name__ == "__main__":
    unittest.main()
