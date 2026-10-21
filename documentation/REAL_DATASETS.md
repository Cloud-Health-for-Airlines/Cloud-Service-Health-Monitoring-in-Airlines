# Real Public Datasets for Airline Cloud-Health Research: Survey and Usage

This is a survey of every public dataset relevant to BACCP: what it is, where to get it, its schema, its licence, and whether and how we use it. The machine-readable version is [`ai-models/real_data/catalog.py`](../ai-models/real_data/catalog.py).
`python ai-models/real_data/download.py --list` prints the catalog.

Legend: ✅ used in the pipeline · 🟡 documented / optional · ❌ considered and rejected (reason given)

---

## A. Airline operations and booking (the *demand* side)

### ✅ BTS Airline On-Time Performance (Reporting Carrier)
- **Publisher:** U.S. DOT, Bureau of Transportation Statistics. TranStats table [`QO_VQ=EFD`](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD). Monthly files from 1987 onward.
- **Download:** `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YYYY}_{M}.zip` (~30 MB per month).
- **Schema (columns we use):** `FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline, Tail_Number, Origin, Dest, CRSDepTime (hhmm local), DepTime, DepDelay, CRSArrTime, ArrDelay, Cancelled, CancellationCode, Diverted, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay`.
- **Licence:** U.S. Government work, public domain.
- **Use:** minute-level schedule per carrier; real disruption windows found automatically (`airline_ops.detect_disruption_windows`: robust z-score on daily cancellation rate). We downloaded **Dec-2022** (Southwest meltdown) and **Jul-2024** (CrowdStrike/Delta). The detector found 47 windows, including WN 22–29 Dec 2022 and DL 19–23 Jul 2024.

### ✅ BTS DB1B: Airline Origin & Destination Survey (Market table)
- **Publisher:** BTS, table [`QO_VQ=EFI`](https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFI). Quarterly, 1993 onward. **A 10% sample of real tickets.** A BTS librarian note says the sample rises to 40% from July 2025; check before using newer quarters.
- **Download:** `https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_{YYYY}_{Q}.zip` (~95 MB). Coupon and Ticket tables are also available.
- **Columns used:** `Year, Quarter, Origin, Dest, TkCarrier, OpCarrier, RPCarrier, Passengers, MktFare, MktDistance`.
- **Use:** passengers per carrier per quarter (×10), divided by departures in a matching on-time month, gives booked journeys per departure. This is the closest thing to a **real airline booking dataset** that is public.

### 🟡 BTS T-100 Segment / Market
Monthly passengers, seats and load factor per route. Useful to convert journeys into per-segment seat occupancy for check-in and baggage. Not yet wired in.

### ❌ Kaggle airline booking datasets
- *Flight Price Prediction* (EaseMyTrip, 300,261 bookings scraped Feb–Mar 2022, has `days_left`). Real-scraped, but needs a Kaggle login and covers Indian routes only. It would be a good source for a **booking lead-time curve** if added manually.
- *customer_booking.csv* (British Airways Forage job simulation, ~50k rows). Provenance undocumented, so rejected for "real data" claims.
- BA passenger *reviews* datasets contain reviews, not bookings.

---

## B. Cloud workload traces (*how the cloud behaves*)

### ✅ Azure Functions Trace 2019 (Azure Public Dataset)
- **Source:** [AzureFunctionsDataset2019.md](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md). Archive: `https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz` (143 MB).
- **Files:**
  - `invocations_per_function_md.anon.d01..d14.csv`: `HashOwner, HashApp, HashFunction, Trigger, 1..1440` (invocations per minute).
  - `function_durations_percentiles.*`
  - `app_memory_percentiles.*`
- **Citation:** Shahrad et al., *Serverless in the Wild*, USENIX ATC 2020.
- **Use:** we take the top 400 HTTP-triggered functions over days 1–7 and keep the 202 that show a day/night cycle (peak/trough ≥ 2, i.e. user-facing). From these we derive:
  - the **diurnal profile** (peak/trough 3.5×), and
  - **autocorrelated burstiness residuals**, averaged over 16 functions to mimic aggregated user traffic.

### 🟡 Azure VM Trace 2017 / 2019 (AzurePublicDatasetV2)
5-minute CPU readings of millions of VMs over 30 days (156 GB compressed). Links are in `AzurePublicDatasetLinksV2.txt`. Citation: Cortez et al., *Resource Central*, SOSP 2017. Too large for the default pipeline. A natural alternative to Alibaba for utilisation statistics.

### 🟡 Azure Functions Invocation Trace 2021
Per-invocation arrival and duration over 2 weeks. Citation: SOSP 2021, *Faster and Cheaper Serverless Computing on Harvested Resources*. Finer-grained alternative to the 2019 trace.

### ✅ Alibaba Cluster Trace: Microservices v2022
- **Source:** [alibaba/clusterdata · cluster-trace-microservices-v2022](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022). 13 days, ~2 TB total.
- **Tables (60 s granularity):**
  - `NodeMetrics(timestamp, nodeid, cpu_utilization, memory_utilization)`, ~665 MB per 12 h
  - `MSMetrics(... msname, msinstanceid, cpu, memory)`, ~350 MB per 30 min
  - `MSRTMCR(... providerrpc_rt/mcr, consumerrpc_rt/mcr, db/mc/mq rt & mcr, http_rt/mcr)`, ~3 GB per hour
  - `CallGraph(traceid, service, rpc_id, um, dm, rpctype, interface, rt)`
- **Citation:** Luo et al., *The Power of Prediction: Microservice Auto Scaling via Workload Learning*, SoCC 2022.
- **Use:** `NodeMetricsUpdate_0` (first 12 h, 1,000 nodes). Median CPU **18.5%**, p90 32%, p99 55%. These set the autoscaler target (p90, clipped to [0.4, 0.8]) and give empirical CPU noise.
- **Caveat:** the 2021 call-graph data has known inconsistencies, and 2022 has fewer ([Huye et al., ICPE 2024](https://research.spec.org/icpe_proceedings/2024/companion/p67.pdf)). Use [CASPER](https://github.com/docc-lab/casper) if reconstructing call graphs.

### 🟡 Alibaba cluster-trace-v2018 / v2017
- v2018: ~4,000 machines over 8 days. `machine_usage.tar.gz` is 1.7 GB; the whole trace is 49 GB. The README asks users to fill a short survey first.
- v2017: ~1,300 machines over 12 h. Introduced co-location of online services with batch jobs.

### 🟡 Google Cluster Workload Trace 2019 (Borg, ClusterData2019)
- 8 Borg cells, May 2019, ~2.4 TiB. **Served only through BigQuery** ([docs](https://github.com/google/cluster-data/blob/master/ClusterData2019.md), CC-BY). Citation: Tirmazi et al., *Borg: the Next Generation*, EuroSys 2020.
- Sample query (needs your own GCP project; the free tier is usually enough for small samples):
  ```sql
  SELECT time, machine_id, average_usage.cpus AS cpu, average_usage.memory AS mem
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage
  WHERE time BETWEEN 600000000 AND 900000000
  LIMIT 100000
  ```
- ClusterData2011 is downloadable from the public GCS bucket `gs://clusterdata-2011-2`.

---

## C. Cloud service health and failures (*when things break*)

### ✅ Google Cloud Service Health incident history
- **Live feed:** `https://status.cloud.google.com/incidents.json` (schema at `/incidents.schema.json`, product catalog at `/products.json`). Fields:
  - `id, begin, end, severity, status_impact, external_desc`
  - `affected_products[{id,title}]`
  - `currently/previously_affected_locations`
  - `updates[]`
- **Problem:** the live feed now keeps only ~7 months (7 incidents when we downloaded it).
- **Fix:** we merge Internet Archive snapshots (Dec-2021, Dec-2022, Dec-2023, Jun-2024, Jun-2025) by incident id, which recovers **902 incidents, Dec-2020 to 2026** (`download.py gcp_incidents_archive`; truncated snapshots are skipped automatically).
- **Use:** fault-type mix (keyword classifier → connection-drop 63%, network-delay 26%, batch-job-stall 11%), severity mix (low 56%, medium 36%, high 9%), and duration distribution (median 191 min, p90 2,199 min; capped at 240 min in simulation).

### ✅ RCAEval (incl. Train-Ticket)
- [Zenodo 14504481](https://zenodo.org/records/14504481), [arXiv 2412.17015](https://arxiv.org/abs/2412.17015v1).
- 735 labelled failure cases (11 fault types) from Online Boutique, Sock Shop and **Train-Ticket**:
  - RE1: metrics only, 375 cases
  - RE2: metrics + logs + traces, 270 cases
  - RE3: code-level faults, 90 cases
- Comes with 15 RCA baselines.
- **Used for:** external validation of the cascade predictor on real benchmark telemetry (RE1-TT, 125 cases). Train-Ticket is the closest open analogue of a reservation stack. Code: `ai-models/real_data/rcaeval.py`; results: `results/real_data/rcaeval_metrics.json`; write-up: REVIEW_RESPONSE_GAP_AND_DATA.md §2.4.
- A related 2025 Train-Ticket dataset has 1,430 validated cases ([arXiv 2510.04711](https://arxiv.org/pdf/2510.04711v1)).

### 🟡 Others
- **AIOps Challenge 2020/2021** datasets (metrics and traces, no logs).
- **Loghub** (system logs).
- **AWS / Azure status histories:** RSS only, ~90 days; no structured archive found.

---

## D. Industry facts used as assumptions (with sources)

| Parameter | Value | Source |
|---|---|---|
| Look-to-book ratio | 1000:1 default (searches → cloud cache, not legacy) | Amadeus via [PhocusWire 2015](https://www.phocuswire.com/Airline-system-look-to-book-ratios-soar-expected-to-go-10x-higher); OAG 10k–20k:1 now ([PhocusWire](https://www.phocuswire.com/news/distribution/look-to-book-ai-rise-search-cost-airlines), [AltexSoft](https://altexsoft.com/blog/look-to-book-ratio-aviation/)) |
| Crew per flight | 5 (2 pilots + ~3 flight attendants) | FAA minimums for narrow-body aircraft |
| Rebooking legacy calls per passenger | 6 (swept 2–10) | Assumption; see [sensitivity.md](../results/real_data/sensitivity.md) |
| Crew ops per disrupted crew member | 12 (4 ops × ~3 downstream legs) | Assumption, motivated by the SkySolver failure mode |
| Mainframe headroom | 1.25 × P99 normal-day demand | Assumption (swept 1.1–1.5) |
| Fault duration cap | 240 min | Assumption (airline-side failover) |
