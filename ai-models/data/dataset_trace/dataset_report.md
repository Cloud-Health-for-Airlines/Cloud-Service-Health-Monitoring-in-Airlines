# BACCP Trace-Driven Dataset (real workload)

**Generated:** 2026-10-19T13:29:31.965450+00:00  
**Command:** `python ai-models/real_data/build_trace_dataset.py --sources WN:202212 DL:202407`

## Provenance

| Component | Source |
|---|---|
| workload | BTS On-Time + BTS DB1B + Azure Functions 2019 (real) |
| fault_durations_and_types | Google Cloud Service Health |
| cloud_utilisation | Alibaba microservices v2022 NodeMetrics |
| services | simulated (queueing model) - see real_data/trace_simulator.py |

## Sources

| Source | Minutes | Faults | Legacy cap (rps) | Pax/flight (source) | Overloaded min w/o fault | Episodes | Cascades |
|---|---|---|---|---|---|---|---|
| WN:202212 | 44246 | 93 | 40.77 | 88.2 (BTS DB1B 2024Q3) | 1661 | 500 | 250 |
| DL:202407 | 44633 | 88 | 38.42 | 105.1 (BTS DB1B 2024Q3) | 168 | 500 | 250 |

## Labels

- Episodes: **1000**, cascades **500** (50.0%)
- Cascade drivers: {'demand-surge': 109, 'fault': 391}
- Lead time (s): {'mean': 673.02, 'median': 180.0, 'min': 10.0, 'max': 3580.0}
- Split: `event` -> train 400, val 100, test 500

| Fault type | Episodes | Cascades |
|---|---|---|
| none | 386 | 109 |
| connection-drop | 381 | 238 |
| network-delay | 183 | 103 |
| batch-job-stall | 50 | 50 |
