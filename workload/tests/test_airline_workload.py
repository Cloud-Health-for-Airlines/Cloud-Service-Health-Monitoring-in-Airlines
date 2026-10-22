"""Unit and integration tests for the realistic airline IT workload generator.

Tests:
1. Reproducibility across runs with identical seeds.
2. Schema validity and data provenance separation.
3. Workload spikes (flash sale surge) and gradual ramp dynamics.
4. Diurnal demand variation (morning peak vs overnight trough).
5. Expanded multi-service airline topology (flight search, payment, check-in).
6. Chaos testbed fault injection integration (network-delay, connection-drop, batch-job-stall).
7. End-to-end integration with BACCPCloudOrchestrator and GenerationTypedGraph.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
AI_MODELS_DIR = REPO_ROOT / "ai-models"
if str(AI_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODELS_DIR))

from workload import (
    RealisticAirlineWorkloadGenerator,
    WorkloadScenarioConfig,
    WorkloadObservation,
    WorkloadSnapshot,
    DataProvenance,
    CANONICAL_5_SERVICES,
    EXPANDED_9_SERVICES,
)
from backend.cloud.orchestrator import BACCPCloudOrchestrator


class TestRealisticAirlineWorkloadGenerator(unittest.TestCase):
    """Test suite validating workload generator requirements."""

    def setUp(self):
        self.generator = RealisticAirlineWorkloadGenerator()

    def test_reproducibility(self):
        """Verify that identical random seeds produce 100% numerically identical workloads."""
        cfg_seed42 = WorkloadScenarioConfig(
            scenario_name="reproducibility_test",
            duration_seconds=100.0,
            sampling_interval_sec=20.0,
            random_seed=42,
        )
        run_a = self.generator.generate_workload(cfg_seed42)
        run_b = self.generator.generate_workload(cfg_seed42)

        self.assertEqual(len(run_a), len(run_b))
        for snap_a, snap_b in zip(run_a, run_b):
            self.assertEqual(snap_a.timestamp_sec, snap_b.timestamp_sec)
            self.assertEqual(snap_a.sync_drift_score, snap_b.sync_drift_score)
            for svc_id in CANONICAL_5_SERVICES:
                obs_a = snap_a.observations[svc_id]
                obs_b = snap_b.observations[svc_id]
                self.assertAlmostEqual(obs_a.request_rate, obs_b.request_rate, places=6)
                self.assertAlmostEqual(obs_a.cpu_demand, obs_b.cpu_demand, places=6)
                self.assertAlmostEqual(obs_a.latency_ms, obs_b.latency_ms, places=6)
                self.assertAlmostEqual(obs_a.queue_length, obs_b.queue_length, places=6)

        # Different seed produces different random realizations
        cfg_seed99 = WorkloadScenarioConfig(
            scenario_name="reproducibility_test",
            duration_seconds=100.0,
            sampling_interval_sec=20.0,
            random_seed=99,
        )
        run_c = self.generator.generate_workload(cfg_seed99)
        self.assertNotEqual(
            run_a[0].observations["reservations"].request_rate,
            run_c[0].observations["reservations"].request_rate,
        )

    def test_schema_validity_and_provenance(self):
        """Verify observation fields, valid ranges, and explicit provenance separation."""
        cfg = WorkloadScenarioConfig(
            duration_seconds=60.0,
            sampling_interval_sec=20.0,
            random_seed=123,
        )
        snapshots = self.generator.generate_workload(cfg)
        self.assertGreaterEqual(len(snapshots), 3)

        for snap in snapshots:
            # Snapshot-level assertions
            self.assertIsInstance(snap.iso_timestamp, str)
            self.assertGreaterEqual(snap.sync_drift_score, 0.0)
            self.assertLessEqual(snap.sync_drift_score, 100.0)

            # JSON serializability
            snap_dict = snap.to_dict()
            json_str = json.dumps(snap_dict)
            self.assertIsInstance(json_str, str)

            for svc_id, obs in snap.observations.items():
                self.assertIsInstance(obs, WorkloadObservation)
                self.assertEqual(obs.service_id, svc_id)
                self.assertIn(obs.service_tier, {"cloud-native", "boundary-gateway", "legacy"})
                self.assertGreaterEqual(obs.request_rate, 0.0)
                self.assertGreaterEqual(obs.request_count, 0.0)
                self.assertGreaterEqual(obs.cpu_demand, 0.0)
                self.assertLessEqual(obs.cpu_demand, 1.0)
                self.assertGreaterEqual(obs.memory_demand, 0.0)
                self.assertLessEqual(obs.memory_demand, 1.0)
                self.assertGreaterEqual(obs.queue_length, 0.0)
                self.assertGreaterEqual(obs.latency_ms, 0.0)
                self.assertGreaterEqual(obs.error_rate, 0.0)
                self.assertLessEqual(obs.error_rate, 1.0)

                # Provenance assertions
                prov = obs.provenance
                self.assertIsInstance(prov, DataProvenance)
                self.assertIsInstance(prov.observed_dataset_values, dict)
                self.assertIsInstance(prov.forecast_values, dict)
                self.assertIsInstance(prov.simulated_values, dict)
                self.assertIsInstance(prov.assumptions, dict)
                self.assertGreater(len(prov.observed_dataset_values), 0)
                self.assertGreater(len(prov.forecast_values), 0)
                self.assertGreater(len(prov.simulated_values), 0)
                self.assertGreater(len(prov.assumptions), 0)

    def test_workload_spikes(self):
        """Verify that a sudden demand surge (flash sale / weather rebooking) multiplies arrival rates."""
        cfg = WorkloadScenarioConfig(
            scenario_name="sudden_spike",
            duration_seconds=300.0,
            sampling_interval_sec=20.0,
            spike_start_sec=100.0,
            spike_duration_sec=60.0,
            spike_magnitude=3.5,
            random_seed=42,
        )
        snapshots = self.generator.generate_workload(cfg)
        
        # Step 2: t=40s (pre-spike baseline)
        pre_spike_rate = snapshots[2].observations["reservations"].request_rate
        
        # Step 6: t=120s (inside the spike window [100s, 160s])
        spike_rate = snapshots[6].observations["reservations"].request_rate
        
        # Step 12: t=240s (post-spike baseline)
        post_spike_rate = snapshots[12].observations["reservations"].request_rate

        # Arrival rate during the spike should be at least 3x baseline
        self.assertGreater(spike_rate, pre_spike_rate * 2.8)
        self.assertGreater(spike_rate, post_spike_rate * 2.8)

    def test_gradual_ramp(self):
        """Verify that a gradual ramp increases demand smoothly over the ramp duration."""
        cfg = WorkloadScenarioConfig(
            scenario_name="gradual_ramp",
            duration_seconds=300.0,
            sampling_interval_sec=20.0,
            ramp_start_sec=40.0,
            ramp_duration_sec=180.0,
            ramp_end_factor=2.5,
            random_seed=42,
        )
        snapshots = self.generator.generate_workload(cfg)
        
        # t=20s (before ramp)
        rate_start = snapshots[1].observations["reservations"].request_rate
        # t=260s (after ramp completion)
        rate_end = snapshots[13].observations["reservations"].request_rate

        self.assertGreater(rate_end, rate_start * 2.0)

    def test_diurnal_periods(self):
        """Verify that empirical BTS diurnal curve yields higher morning peak than overnight trough."""
        cfg_peak = WorkloadScenarioConfig(start_hour=9.0, duration_seconds=60.0, sampling_interval_sec=20.0)
        cfg_trough = WorkloadScenarioConfig(start_hour=3.0, duration_seconds=60.0, sampling_interval_sec=20.0)

        peak_snaps = self.generator.generate_workload(cfg_peak)
        trough_snaps = self.generator.generate_workload(cfg_trough)

        peak_rate = peak_snaps[0].observations["reservations"].request_rate
        trough_rate = trough_snaps[0].observations["reservations"].request_rate

        # 09:00 peak should be significantly higher than 03:00 red-eye trough (BTS profile indicates > 5x)
        self.assertGreater(peak_rate, trough_rate * 3.0)

    def test_expanded_airline_services(self):
        """Verify simulating the full 9-service airline IT topology with search-to-book ratios."""
        cfg = WorkloadScenarioConfig(
            services=EXPANDED_9_SERVICES,
            duration_seconds=60.0,
            sampling_interval_sec=20.0,
            random_seed=42,
        )
        snapshots = self.generator.generate_workload(cfg)
        obs = snapshots[0].observations

        # Verify all 9 services exist
        for s in EXPANDED_9_SERVICES:
            self.assertIn(s, obs)

        # In commercial airlines, flight-search read volume is typically 5x-20x the reservation booking volume
        search_rate = obs["flight-search"].request_rate
        res_rate = obs["reservations"].request_rate
        self.assertGreater(search_rate, res_rate * 4.0)

    def test_chaos_fault_injection_integration(self):
        """Verify integration with testbed chaos profiles (network-delay, connection-drop, batch-job-stall)."""
        # 1. Network Delay fault
        cfg_delay = WorkloadScenarioConfig(
            duration_seconds=200.0,
            sampling_interval_sec=20.0,
            active_fault="network-delay",
            fault_level="high",  # 1500ms delay in profiles.json
            fault_start_sec=60.0,
            random_seed=42,
        )
        snaps_delay = self.generator.generate_workload(cfg_delay)
        
        # t=40s (pre-fault): latency normal
        self.assertLess(snaps_delay[2].observations["legacy-core"].latency_ms, 30.0)
        
        # t=100s (fault active): latency > 1000ms, queue depth rises, sync drift rises
        fault_snap = snaps_delay[5]
        self.assertGreater(fault_snap.observations["legacy-core"].latency_ms, 1200.0)
        self.assertGreater(fault_snap.observations["boundary-gateway"].queue_length, 10.0)
        self.assertGreater(fault_snap.sync_drift_score, 45.0)

        # 2. Connection Drop fault
        cfg_drop = WorkloadScenarioConfig(
            duration_seconds=200.0,
            sampling_interval_sec=20.0,
            active_fault="connection-drop",
            fault_level="high",
            fault_start_sec=60.0,
            random_seed=42,
        )
        snaps_drop = self.generator.generate_workload(cfg_drop)
        fault_drop_snap = snaps_drop[5]
        self.assertGreater(fault_drop_snap.observations["legacy-core"].error_rate, 0.50)

    def test_integration_with_baccp_cloud_orchestrator(self):
        """Verify that generated telemetry seamlessly feeds into BACCPCloudOrchestrator."""
        cfg = WorkloadScenarioConfig(
            services=CANONICAL_5_SERVICES,
            duration_seconds=60.0,
            sampling_interval_sec=20.0,
            random_seed=42,
        )
        telemetry_series = self.generator.generate_canonical_telemetry_series(cfg)
        self.assertEqual(len(telemetry_series), 3)

        orchestrator = BACCPCloudOrchestrator()
        step_0 = telemetry_series[0]

        # Execute orchestrator pipeline with generated telemetry
        result = orchestrator.run_pipeline(
            telemetry=step_0["telemetry"],
            sync_drift_score=step_0["sync_drift_score"],
            active_fault=step_0["active_fault"],
            fault_level=step_0["fault_level"],
        )

        self.assertEqual(result.get("status"), "success")
        self.assertIn("telemetry", result)
        self.assertIn("prediction", result)
        self.assertIn("cascade_probability", result["prediction"])

    def test_integration_with_generation_typed_graph(self):
        """Verify that generate_graph_sequence produces PyG/HeteroGraphData compatible tensors."""
        cfg = WorkloadScenarioConfig(
            services=CANONICAL_5_SERVICES,
            duration_seconds=60.0,
            sampling_interval_sec=20.0,
            random_seed=42,
        )
        graph_seq = self.generator.generate_graph_sequence(cfg)
        self.assertEqual(len(graph_seq), 3)

        first_snap = graph_seq[0]
        self.assertIn("flat_x", first_snap)
        self.assertIn("flat_edge_index", first_snap)
        self.assertIn("hetero_snapshot", first_snap)

        flat_x = first_snap["flat_x"]
        self.assertEqual(flat_x.shape[0], 5)  # 5 canonical nodes
        self.assertEqual(flat_x.shape[1], 6)  # 6 features per node


if __name__ == "__main__":
    unittest.main()
