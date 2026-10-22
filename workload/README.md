# BACCP Workload Management, Public Datasets, and Workload Generation

This package implements workload management, public dataset adapters, and realistic airline IT workload generation for the Boundary-Aware Cascade Predictor (BACCP) project.

## Directory Structure

```
workload/
├── datasets/
│   ├── google_cluster.py         # Google Cluster Trace Adapter (clusterdata-2011-2 & 2019a)
│   ├── sample_data/
│   │   ├── google_cluster_task_events_sample.csv
│   │   ├── google_cluster_task_usage_sample.csv
│   │   ├── google_cluster_sample.json
│   │   └── bts_airline_demand_profile.json
│   └── README.md                 # Google cluster dataset documentation & license
├── generator/
│   ├── airline_workload.py       # RealisticAirlineWorkloadGenerator implementation
│   ├── models.py                 # WorkloadObservation, DataProvenance, WorkloadSnapshot
│   ├── patterns.py               # Diurnal curves, demand shapers, service topologies
│   └── README.md                 # Workload generator detailed documentation
├── forecasting/
│   ├── dataset.py                # Historical BTS demand loader, validator, chronological splitter
│   ├── models.py                 # Baselines, AutoRegressive Ridge forecaster, uncertainty intervals
│   ├── pipeline.py               # Training, baseline evaluation, artifact generation
│   ├── workload_mapper.py        # Calibrated mapping from passenger forecast to service TPS
│   └── README.md                 # Demand forecasting documentation & evaluation report
├── tests/
│   ├── test_google_cluster.py    # 11 unit tests for Google trace adapter
│   ├── test_airline_workload.py  # 9 unit & integration tests for workload generator
│   └── test_demand_forecasting.py# 10 unit & integration tests for demand forecasting
└── __init__.py                   # Package exports
```

## Quick Verification

Run all workload tests:
```bash
python3 -m unittest discover -s workload/tests -p "test_*.py" -v
```
All 30 tests pass in ~0.14s without requiring external network access or cloud credentials.
