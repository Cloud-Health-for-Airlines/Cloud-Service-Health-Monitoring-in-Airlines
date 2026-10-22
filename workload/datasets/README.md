# Google Cluster Workload Traces for Airline IT Cloud Simulation

This module integrates real-world trace data from Google Cluster Workload Traces into the **Boundary-Aware Cascade Predictor (BACCP)** project to enhance resource-utilization modeling and workload-management simulations across airline microservices.

---

## 1. Dataset Provenance & Attribution

- **Official Repository**: [google/cluster-data (GitHub)](https://github.com/google/cluster-data)
- **Trace Versions Supported**:
  - `clusterdata-2011-2`: Canonical 29-day trace from a ~12.5k machine Borg cell (May 2011).
  - `clusterdata-2019a`: Modern 8-cell Borg workload trace (May 2019).
- **License / Terms of Use**:
  - Distributed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.
  - You are free to share and adapt the material with appropriate attribution to Google Inc.
- **Foundational Literature**:
  - Wilkes, J. (2011). *More Google cluster data*. Google research blog.
  - Verma, A. et al. (2015). *Large-scale cluster management at Google with Borg*. EuroSys 2015.
  - Tirmazi, M. et al. (2020). *Borg: the next generation*. EuroSys 2020.

---

## 2. Why Google Cluster Data for Airline IT?

In commercial airline cloud architectures, services operate across distinct priority and latency-sensitivity tiers:
1. **Flight Reservations (`reservations`)**: Latency-critical, revenue-generating, customer-facing transactions requiring high availability and low latency.
2. **Crew Operations (`crew`)**: Mission-critical internal operations with moderate burstiness during flight crew check-ins and schedule reassignments.
3. **Baggage Reconciliation & Batch Analytics (`baggage`)**: Throughput-oriented background workloads that can be throttled or shed during boundary overload.

Google Cluster Data provides realistic, empirical task arrival distributions, priority scheduling tiers, resource over-commitment patterns, and eviction dynamics that mirror enterprise airline cluster behavior.

### Domain Tier Mapping

| Google Cluster `scheduling_class` | Google Cluster `priority` | Airline IT Service | Operational Role |
| :---: | :---: | :--- | :--- |
| **`3`** | **`9 - 11`** | **`reservations`** | Latency-critical production service; protected by PPO circuit breaker. |
| **`2`** | **`6 - 8`** | **`crew`** | Operational mid-tier service; throttled under severe cascade risk. |
| **`0 - 1`** | **`0 - 5`** | **`baggage`** | Batch analytics & logging; first candidate for selective load shedding. |

---

## 3. Schema Reference

### `task_events` Table (Lifecycle Transitions)

Distributed as headerless CSV (or `.csv.gz`):

| Column Index | Field Name | Type | Description |
| :---: | :--- | :---: | :--- |
| **0** | `timestamp` | Integer | Microseconds since trace start. |
| **1** | `missing_info` | Integer | `0` = complete record; `1` = missing information. |
| **2** | `job_id` | Integer | Unique identifier for parent job. |
| **3** | `task_index` | Integer | Zero-based index within parent job. |
| **4** | `machine_id` | String | Machine identifier (empty if event occurs before scheduling). |
| **5** | `event_type` | Integer | `0`: SUBMIT, `1`: SCHEDULE, `2`: EVICT, `3`: FAIL, `4`: FINISH, `5`: KILL, `6`: LOST, `7`: UPDATE_PENDING, `8`: UPDATE_RUNNING. |
| **6** | `user_name` | String | Obfuscated user string hash. |
| **7** | `scheduling_class` | Integer | Latency sensitivity class (`0`: batch, `1`: non-prod, `2`: mid, `3`: production). |
| **8** | `priority` | Integer | Task scheduling priority (`0` lowest to `11` highest). |
| **9** | `cpu_request` | Float | Normalized CPU requested relative to machine capacity $[0, 1]$. |
| **10** | `memory_request` | Float | Normalized memory requested relative to machine capacity $[0, 1]$. |
| **11** | `disk_space_request` | Float | Normalized disk space requested $[0, 1]$. |
| **12** | `diff_machine_constraint` | Boolean | `1` if tasks must run on different machines; `0` otherwise. |

### `task_usage` Table (Periodic Resource Utilization)

Captures resource metrics over 300-second (5-minute) sampling intervals:

| Column Index | Field Name | Type | Description |
| :---: | :--- | :---: | :--- |
| **0** | `start_time` | Integer | Window start timestamp in microseconds. |
| **1** | `end_time` | Integer | Window end timestamp in microseconds. |
| **2** | `job_id` | Integer | Unique identifier for parent job. |
| **3** | `task_index` | Integer | Task index within job. |
| **4** | `machine_id` | String | Identifier of machine running the task. |
| **5** | `cpu_usage` | Float | Mean CPU utilization rate over window $[0, 1]$. |
| **6** | `memory_usage` | Float | Mean canonical memory usage $[0, 1]$. |
| **7** | `assigned_memory_usage` | Float | Assigned memory footprint (including page cache). |
| **8** | `unmapped_page_cache` | Float | Page cache not mapped to user space. |
| **9** | `total_page_cache` | Float | Total page cache footprint. |
| **10** | `disk_io_time` | Float | Fraction of time waiting on disk I/O. |
| **11** | `local_disk_space_usage`| Float | Local disk space consumed. |
| **12** | `max_cpu_usage` | Float | Peak CPU usage observed during window. |
| **13** | `max_memory_usage` | Float | Peak memory usage observed during window. |
| **14** | `max_disk_io_time` | Float | Peak disk I/O wait fraction. |
| **15** | `cpu_rate_1s_sample` | Float | Random 1-second sample of CPU utilization rate. |

---

## 4. Download Procedure

> [!IMPORTANT]
> The full Google Cluster dataset is approximately **40 GB compressed** and over **2 TB uncompressed**. The repository does **NOT** download this dataset automatically during builds or tests. Instead, a lightweight curated sample is bundled in `workload/datasets/sample_data/`.

To acquire the complete production dataset for extensive offline evaluation:

```bash
# 1. Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# 2. Authenticate with Google Cloud
gcloud auth login

# 3. Download a specific partition of clusterdata-2011-2 (e.g., partition 0):
gsutil -m cp gs://clusterdata-2011-2/task_events/part-00000-of-00500.csv.gz ./data/
gsutil -m cp gs://clusterdata-2011-2/task_usage/part-00000-of-00500.csv.gz ./data/

# 4. Or query the 2019 Borg traces on BigQuery:
# Project: google.com:google-cluster-data
# Dataset: clusterdata_2019_a
```

---

## 5. Preprocessing & Normalization Pipeline

The adapter (`GoogleClusterTraceAdapter`) performs the following preprocessing stages:

1. **Format Agnostic Parsing**: Supports uncompressed CSV, gzip-compressed `.csv.gz`, and consolidated JSON files. Auto-detects header presence or falls back to standard Google column index layout.
2. **Timestamp Normalization**: Converts raw microsecond offsets (`timestamp_usec`) to relative seconds (`timestamp_sec = timestamp_usec / 1e6`) and window durations.
3. **Missing Value Imputation**:
   - Empty `machine_id` values (pre-scheduling submissions) are coerced to `None`.
   - Records with `missing_info = 1` are imputed with class-conditional defaults (e.g., $0.05$ CPU for Class 2/3; $0.02$ for Class 0/1).
   - Non-numeric or corrupted values are safely clamped to $0.0$ unless `strict=True` is requested.
4. **Preservation of Raw Fields**: Every record preserves its original row fields in `raw_fields` for complete auditability.
5. **Workload Intensity Aggregation**: `compute_workload_intensity_timeseries(window_seconds)` groups events into time windows, computing arrival velocity, failure/eviction rates, active concurrency, and CPU/memory pressure.
6. **Telemetry Transformation**: `extract_airline_workload_profile()` converts cluster load profiles into RED metrics (`request_rate`, `latency_ms`, `error_rate`, `cpu_load`) matching the input schema of `backend/cloud/orchestrator.py`.

---

## 6. Dataset Limitations

1. **Obfuscation**: Usernames and machine IDs are cryptographically hashed; internal Google application semantics are hidden.
2. **Capacity Normalization**: Resources are scaled to the maximum machine size in the cell rather than physical core counts or absolute gigabytes.
3. **Single Cell Scope**: The 2011 dataset reflects a single homogeneous cluster cell; multi-region latency is modeled via network chaos rather than raw trace timestamps.
