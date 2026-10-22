# BACCP Realistic Airline IT Workload Generator

## 1. Overview & Objective

The **Realistic Airline IT Workload Generator** provides time-dependent, multi-service operational workloads for airline IT systems. It is designed to evaluate:
- Workload management and queue backpressure dynamics
- Dynamic capacity planning and airline passenger demand forecasting
- Boundary State Synchronization Drift $\epsilon(t)$ divergence
- Early cascade prediction using the Heterogeneous RGCN-GRU model
- Cloud mitigation actuation via the PPO reinforcement learning circuit breaker

---

## 2. Architecture & Data Provenance

To ensure scientific rigor, this generator maintains strict boundaries between generic computing infrastructure and empirical airline business operations:

```
+-----------------------------------+        +-----------------------------------+
|     Google Cluster Traces         |        |   Public Airline Datasets (BTS)   |
|     (clusterdata-2011-2)          |        |       (TranStats & FAA ASPM)      |
+-----------------------------------+        +-----------------------------------+
                  |                                            |
                  v                                            v
  [Generic Compute Characteristics]            [Airline Operational Patterns]
  - Empirical CPU utilization distributions    - Empirical 24h diurnal departure curve
  - Empirical memory demand variance           - Day-of-week passenger volume indices
  - Task arrival burstiness & volatility       - High-volume search-to-book ratio (15:1)
                  \                                            /
                   \                                          /
                    v                                        v
                 +----------------------------------------------+
                 |    RealisticAirlineWorkloadGenerator         |
                 +----------------------------------------------+
                                        |
                 +----------------------+-----------------------+
                 |                      |                       |
                 v                      v                       v
          [Traffic Shapers]      [Boundary Queue]      [Chaos Testbed]
          - Normal / Peak        - M/G/1 queue         - network-delay
          - Sudden spikes        - Backpressure        - connection-drop
          - Gradual ramps        - Drift epsilon(t)    - batch-job-stall
                                        |
                                        v
                         +-----------------------------+
                         |     WorkloadObservation     |
                         |  (with Explicit Provenance) |
                         +-----------------------------+
```

### Data Provenance Separation Contract
Every generated observation includes a [`DataProvenance`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/workload/generator/models.py) structure separating:
- **`observed_dataset_values`**: Direct empirical reference points (e.g., Google Cluster task CPU baseline, BTS diurnal multipliers).
- **`forecast_values`**: Forward-projected demand rates and scenario scaling factors.
- **`simulated_values`**: Discrete-event metrics (queue depth, round-trip latency, error rates, $\epsilon(t)$ sync drift).
- **`assumptions`**: Documented system parameters (e.g., search-to-book ratio, mainframe capacity ceilings).

---

## 3. Simulated Airline IT Services

The generator supports both the canonical 5-service topology required by the GNN model, as well as an expanded 9-service suite representing the complete airline digital footprint:

| Service Identifier | Tier | Criticality | Base TPS | Gateway Call? | Characteristics |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `flight-search` | `cloud-native` | Medium | 180.0 | No | High volume browse traffic (15x book ratio), read-heavy, low CPU/req. |
| `reservations` | `cloud-native` | High | 28.0 | Yes (85%) | Core booking engine, ACID PNR ledger synchronization. |
| `payment` | `cloud-native` | High | 12.0 | Yes (50%) | Strict transaction integrity, zero drop tolerance, higher CPU. |
| `check-in` | `cloud-native` | High | 22.0 | Yes (60%) | Departure bank spikes (T-24h to T-1h), seat allocation. |
| `crew` | `cloud-native` | Medium | 14.0 | Yes (70%) | Flight crew pairing, duty tracking, FAA Part 117 compliance. |
| `baggage` | `cloud-native` | Low | 16.0 | Yes (40%) | BSM/BPM baggage scan events, tolerant of brief deferral. |
| `notifications` | `cloud-native` | Low | 35.0 | No | Event-driven flight status and gate change alerts. |
| `boundary-gateway` | `boundary-gateway` | Boundary | 55.0 | N/A | HTTP-to-TCP reverse-proxy, queue buffer, backpressure tracking. |
| `legacy-core` | `legacy` | Legacy | 55.0 | N/A | Mainframe transaction core (CICS/IMS), 75 TPS capacity limit. |

---

## 4. Workload Scenarios & Characteristics

1. **Diurnal Fluctuation (`start_hour`, `day_of_week`)**:
   - Continuous interpolation across empirical 24-hour BTS curves (03:00 red-eye trough vs 09:00 departure bank).
2. **Sudden Demand Spikes (`spike_start_sec`, `spike_magnitude`)**:
   - Models promotional flash sales or severe weather cancellation rebooking surges.
3. **Gradual Demand Ramps (`ramp_start_sec`, `ramp_end_factor`)**:
   - Models holiday travel passenger surges (e.g. Thanksgiving or peak summer ramp).
4. **Chaos Fault Injection (`active_fault`, `fault_level`)**:
   - Direct compatibility with [`testbed/chaos/profiles.json`](file:///Users/bhiwanshusharma/Documents/Cloud_Project/testbed/chaos/profiles.json):
     - `network-delay`: Injects 100ms (low), 500ms (medium), or 1500ms (high) latency.
     - `connection-drop`: Injects 10%, 50%, or 100% TCP connection resets / HTTP 502.
     - `batch-job-stall`: Simulates mainframe stalls pausing completion for 1s, 5s, or 15s.

---

## 5. Usage Example

```python
from workload.generator import (
    RealisticAirlineWorkloadGenerator,
    WorkloadScenarioConfig,
    CANONICAL_5_SERVICES,
)

# 1. Initialize generator (loads Google Cluster traces & BTS demand profile)
generator = RealisticAirlineWorkloadGenerator()

# 2. Configure a flash sale spike scenario
config = WorkloadScenarioConfig(
    scenario_name="flash_sale_spike",
    duration_seconds=300.0,
    sampling_interval_sec=20.0,
    demand_intensity=1.2,
    random_seed=42,
    spike_start_sec=100.0,
    spike_duration_sec=60.0,
    spike_magnitude=3.5,
    services=CANONICAL_5_SERVICES,
)

# 3. Generate workload snapshots
snapshots = generator.generate_workload(config)

for snap in snapshots:
    obs = snap.observations["reservations"]
    print(f"[{snap.iso_timestamp}] Reservations Rate: {obs.request_rate:.1f} TPS, "
          f"CPU: {obs.cpu_demand:.2f}, Latency: {obs.latency_ms:.1f}ms, "
          f"Boundary Drift: {snap.sync_drift_score:.1f}%")
```

---

## 6. Verification & Test Suite

Run the dedicated test suite:
```bash
python3 -m unittest workload/tests/test_airline_workload.py -v
```

All 9 unit and integration tests verify:
- Deterministic reproducibility (`test_reproducibility`)
- Full schema and provenance validity (`test_schema_validity_and_provenance`)
- Sudden spike and gradual ramp dynamics (`test_workload_spikes`, `test_gradual_ramp`)
- Diurnal time-of-day demand differences (`test_diurnal_periods`)
- Expanded 9-service topology (`test_expanded_airline_services`)
- Chaos testbed fault injection (`test_chaos_fault_injection_integration`)
- Integration with BACCP Cloud Orchestrator (`test_integration_with_baccp_cloud_orchestrator`)
- Integration with GenerationTypedGraph (`test_integration_with_generation_typed_graph`)
