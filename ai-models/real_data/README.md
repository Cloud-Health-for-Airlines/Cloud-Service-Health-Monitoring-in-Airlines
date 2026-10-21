# `real_data/`: BACCP on real public datasets

Builds the BACCP training and evaluation data from **real** sources instead of AI-generated traces. For the why, read [documentation/REVIEW_RESPONSE_GAP_AND_DATA.md](../../documentation/REVIEW_RESPONSE_GAP_AND_DATA.md). For the dataset survey, read [documentation/REAL_DATASETS.md](../../documentation/REAL_DATASETS.md).

```
BTS On-Time ─┐                                     ┌─► dataset_trace/baccp_dataset.pt  (team's .pt format)
BTS DB1B ────┼─► reservation_workload.py ──┐       │
Azure 2019 ──┘   (per-minute rps/service)  ├─► trace_simulator.py ─► build_trace_dataset.py
Alibaba 2022 ──► autoscaler target/noise ──┤   (queueing + autoscale  │
GCP incidents ─► fault types/durations ────┘    + retries + faults)   └─► train_on_trace.py ─► results/real_data/
```

| File | Purpose |
|---|---|
| `catalog.py` | Every dataset: URL, licence, citation, role |
| `download.py` | Resumable downloader into `data/raw/` (git-ignored) |
| `airline_ops.py` | BTS On-Time + DB1B loaders, disruption-window detector |
| `cloud_traces.py` | Azure Functions + Alibaba loaders |
| `cloud_health.py` | Google Cloud incident history loader and classifier |
| `prepare.py` | Raw → small committed artifacts in `ai-models/data/real_processed/` |
| `reservation_workload.py` | Real flights + passengers → requests/s for search, booking, check-in, rebooking, baggage, crew |
| `trace_simulator.py` | Replays the workload through the 5-service hybrid stack in 10-second steps (faults ramp up gradually or hit abruptly); labels cross-boundary cascades that persist ≥ 30 s |
| `build_trace_dataset.py` | Cuts balanced 60-min episodes; event split (train WN Dec-22 / test DL Jul-24) |
| `train_on_trace.py` | Trains `cascade_predictor` (team code, unchanged) in `early` or `full` mode, plus baselines |
| `report_figures.py` | Figures and sensitivity table |
| `early_warning_diagnostics.py` | Leakage (random vs day vs event split), horizon and precursor-visibility checks |
| `rcaeval.py` | External check on real Train-Ticket telemetry (RCAEval) |
| `walkthrough.py` | Step-by-step terminal demo of the whole pipeline (`--pause` for live reviews) |

## Quickstart

```bash
# Optional: refresh raw data (~1.2 GB). The processed artifacts are already committed.
python ai-models/real_data/download.py
python ai-models/real_data/prepare.py

python ai-models/real_data/build_trace_dataset.py          # ~2 min
python ai-models/real_data/train_on_trace.py --mode early  # early warning (2 min before)
python ai-models/real_data/train_on_trace.py --mode full   # detection
python ai-models/real_data/report_figures.py
python ai-models/real_data/early_warning_diagnostics.py
python ai-models/real_data/walkthrough.py --pause
python -m pytest ai-models/tests/test_real_data.py
```

To add another airline or month (e.g. the Alaska Airlines Oct-2025 outage):

```bash
python ai-models/real_data/download.py bts_ontime --param year=2025 month=10
python ai-models/real_data/prepare.py --carriers AS WN DL
python ai-models/real_data/build_trace_dataset.py --sources WN:202212 DL:202407 AS:202510
```

## Notes for integration

- `trace_*` weights live in `ai-models/weights/trace_early/` and `trace_full/`. The originals are untouched.
- The features use the same normalisation as `backend/cloud/sagemaker.py` (lat/100, rate/1000, err, cpu, one-hot), so these weights can be served by the existing endpoint.
- The older per-sample trainers in `ai-models/evaluation/train_cascade_predictor.py` collapse to a constant predictor on this data (no batching or feature scaling). Use `cascade_predictor/` instead.
