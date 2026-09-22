# BACCP Dataset Generation Report

**Generated:** 2026-09-22T08:22:45.330903+00:00  
**Generation Mode:** `synthetic` (Synthetic Fallback Simulator)  
**Source Command:** `python ai-models/data/generate_dataset.py --mode=synthetic`  

> [!NOTE]
> **Mode Transparency:** As detailed in `IMPLEMENTATION_STATUS.md`, Docker and eBPF kernel privileges are absent on this host. Numbers below were produced via the high-fidelity statistical fallback modeling the exact `testbed/chaos/profiles.json` network latency and queue saturation equations. To reproduce on a Docker/eBPF-enabled Linux runner, use:  
> `python ai-models/data/generate_dataset.py --mode=live`

---

## 1. Summary Statistics

| Metric | Value |
| :--- | :--- |
| **Total Experimental Runs** | **60** |
| **Cascade Outages Incurred** | **35** (58.33%) |
| **Nominal Baseline Runs** | **25** (41.67%) |
| **Mean Warning Lead Time** | **45.3s** (~0.75 minutes) |
| **Median Warning Lead Time** | **40.0s** |
| **Lead Time Range** | 5.0s - 100.0s |

---

## 2. Fault Profile Breakdown

| Fault Profile | Total Injections | Cascades Triggered | Cascade Rate |
| :--- | :---: | :---: | :---: |
| `connection-drop` | 15 | 10 | 66.7% |
| `batch-job-stall` | 15 | 15 | 100.0% |
| `none` | 15 | 0 | 0.0% |
| `network-delay` | 15 | 10 | 66.7% |

---

## 3. Dataset Splits (Split by Run ID to Prevent Temporal Leakage)

- **Train Set (36 runs, 60%):** `[0, 1, 3, 7, 8, 9, 11, 13]...`
- **Validation Set (12 runs, 20%):** `[5, 6, 10, 14, 17, 18, 22, 24, 29, 47, 54, 57]`
- **Test Set (12 runs, 20%):** `[2, 4, 12, 23, 36, 39, 40, 42, 45, 53, 55, 56]`
