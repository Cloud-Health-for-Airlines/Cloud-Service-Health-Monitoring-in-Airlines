# Response to Review 1: Research Gap, Real Datasets, and Terminology

This document answers the three points raised in the first project review:

1. *What gap did you find, and how did you arrive at this solution?*
2. *Use real datasets, not only AI-generated data, for the demo.*
3. *Terms to understand:* workload / cluster-workload dataset, Azure dataset, generating reservation workload and how the cloud manages it, simulated airline services, BTS data, airline booking dataset, cloud service-health dataset.

Every number below comes from real public data processed by code in
[`ai-models/real_data/`](../ai-models/real_data/). The detailed dataset survey is in
[REAL_DATASETS.md](REAL_DATASETS.md).

---

## 1. The gap, and how we found it

### 1.1 How we searched

We put three bodies of work side by side and asked what each one leaves out:

| Body of work | What it studies | Typical data | What it leaves out |
|---|---|---|---|
| **Cloud AIOps / failure prediction** (GNN root-cause analysis, microservice anomaly detection) | Faults that spread between cloud microservices | Alibaba / Google / Azure cluster traces, microservice benchmarks (Train-Ticket, Sock Shop, Online Boutique via RCAEval) | No legacy tier and no airline demand. Cluster traces carry no network or latency data ([Measurement-based resource allocation survey, arXiv 2408.09497](https://arxiv.org/pdf/2408.09497)) |
| **Airline operations research** (delay propagation, crew recovery, disruption management) | Flights, crews, aircraft | BTS On-Time, DB1B | Treats the IT systems as if they never fail, even though recent meltdowns were IT-amplified |
| **Industry incident reports** | What actually broke | Post-mortems and news coverage | Not turned into datasets or predictive models |

### 1.2 What the real incidents show

| Incident | Operational trigger | IT element that amplified it | Seen in real BTS data (our loader, no hand-labelling) |
|---|---|---|---|
| **Southwest, Dec 2022** | Winter Storm Elliott | The SkySolver crew-scheduling tool was overwhelmed by the volume of re-assignments ([Senate Commerce SWAPA submission](https://www.commerce.senate.gov/services/files/B8D729EC-5F96-4E8D-A902-F43DA29F2E08), [TravelPulse](https://www.travelpulse.com/news/airlines-airports/what-really-caused-southwest-airlines-holiday-meltdown)) | WN, **22–29 Dec**: peak **77.5%** of flights cancelled, **16,435** cancellations. DL and UA recovered by 26 Dec; WN did not |
| **Delta, Jul 2024** | CrowdStrike / Windows outage | "One of our crew tracking-related tools was affected and unable to effectively process the unprecedented number of changes" ([OPB/NPR](https://opb.org/article/2024/07/31/delta-s-ceo-says-the-crowdstrike-outage-cost-the-airline-500-million-in-5-days)) | DL, **19–23 Jul**: peak **47.4%** cancelled, **5,737** cancellations |
| **Alaska Airlines, Jul and Oct 2025** | None (pure IT) | Primary data-centre hardware failure caused a system-wide ground stop ([CBS](https://www.cbsnews.com/news/alaska-airlines-requests-system-wide-ground-stop-due-to-it-outage), [TechRadar](https://www.techradar.com/pro/security/alaska-airlines-has-grounded-its-fleet-once-again-due-to-a-mystery-it-issue)) | Outside our two downloaded months. Can be added with `download.py bts_ontime --param year=2025 month=10` |

**Pattern:** a surge of operational changes (rebookings, crew re-pairing) hits a system of record that cannot scale, and the failure then spreads.

### 1.3 What the real data adds

- **The cloud side scales easily.** Alibaba's production microservice nodes run at a **median of 18.5% CPU** (p90 32%, 1,000 nodes, 708k samples). The cloud tier has large headroom and autoscales.
- **The legacy side does not.** Mainframe and system-of-record capacity is fixed. In our generated workload, the Southwest meltdown raised *sustained* boundary load by **1.47×**, and by 1.38× for Delta, over normal days ([sensitivity.md](../results/real_data/sensitivity.md)).
- **Autoscaling makes the boundary problem worse.** The more the cloud tier scales out to absorb a surge, the more write traffic it pushes into the fixed-capacity boundary.
- **Demand alone rarely breaks the boundary. Demand plus a degradation does.** The meltdown's raised load stayed under capacity at the default headroom, except in short windows. Combined with a real-style fault (Google Cloud incident mix), it cascades faster. Our dataset shows this: 391 of the 500 cascades are fault-driven and 109 are pure demand surges, most of them during the Southwest meltdown.

### 1.4 The gap statement

> No existing work or dataset links **real airline operational demand** to the **health of the hybrid IT stack** that serves it. In particular, nothing models the boundary between an elastic cloud tier and an inelastic legacy system of record. Cloud AIOps models treat that stack as a homogeneous cloud graph and are evaluated on synthetic or single-benchmark data with random splits. Airline operations research treats IT as infallible. As a result, nobody predicts (with lead time) the cross-boundary cascades that turned weather and software incidents into the Southwest 2022 and Delta 2024 meltdowns, and nobody tests whether such a predictor transfers to a different airline and a different event.

Sub-gaps, each mapped to what BACCP does about it:

| # | Sub-gap | BACCP response |
|---|---|---|
| G1 | No dataset joins airline operations with IT-service health | Trace-driven dataset built from BTS + DB1B + Azure + Alibaba + Google Cloud incidents ([`real_data/`](../ai-models/real_data/)) |
| G2 | Cluster traces model the cloud only: no legacy tier, no network or latency | Generation-typed graph (cloud / boundary-gateway / legacy) with a boundary sync-drift feature ε(t) |
| G3 | AIOps flags failures that are already happening and gives no lead time | Multi-task predictor with a lead-time head, evaluated in an **early-warning** mode (model sees only data from before the cascade) |
| G4 | Evaluations use random splits on one synthetic distribution | **Event split**: train on Southwest Dec-2022, test on Delta Jul-2024 (different airline, different real event) |
| G5 | Prediction is not connected to mitigation | RL circuit breaker (existing work in `circuit-breaker/`), now testable against real demand curves |

### 1.5 How the gap led to this solution

1. *Failures happen at the elastic/inelastic boundary.* So the graph must type nodes by technology generation, not treat everything as one cloud graph. **Hence the generation-typed heterogeneous GNN.**
2. *The legacy tier cannot be instrumented.* So health has to be inferred from the boundary: what is submitted versus what completes. **Hence the drift metric ε(t) and gateway-side eBPF.**
3. *Operators need time to act.* **Hence lead-time prediction and an early-warning evaluation.**
4. *The reviewer asked for real data.* We cannot obtain a real airline's internal telemetry, so we use **trace-driven simulation**, the standard systems-research method. Real demand, real cloud behaviour and real fault statistics drive simulated services, and we say clearly which parts are which (§2.2).

---

## 2. Real datasets now used

### 2.1 Datasets and their role

| Reviewer's term | Dataset we use | Real? | Role in BACCP |
|---|---|---|---|
| **BTS data** | BTS Airline On-Time Performance, Dec-2022 and Jul-2024 (557k + 640k flights) | Yes, U.S. DOT | Minute-by-minute flight schedule. Real cancellations and delays drive rebooking and crew-repair surges. Ground-truth disruption windows |
| **Airline booking dataset** | BTS DB1B Market 2024-Q3 (10% sample of real tickets) | Yes, U.S. DOT | Passengers per departure per carrier (e.g. WN 88.2, DL 105.1 booked journeys per departure) |
| **Azure dataset** | Azure Functions Trace 2019 (Azure Public Dataset) | Yes, Microsoft | Real diurnal shape (peak/trough 3.5× across 202 human-driven HTTP functions) and minute-level burstiness |
| **Cluster workload dataset** | Alibaba Microservices Trace v2022, NodeMetrics | Yes, Alibaba | Real utilisation: sets the autoscaler target and CPU noise |
| **Cloud service-health dataset** | Google Cloud Service Health incident history, live feed plus Internet Archive snapshots (**902 incidents**, Dec-2020 to 2026) | Yes, Google | Real fault-type mix (connection 63%, network 26%, batch 11%), severity mix, and durations (median 3.2 h) |
| (external validation) | RCAEval RE1-TT: 125 fault-injection cases on Train-Ticket, a ~65-pod ticket-booking system | Yes, benchmark deployment (Pham et al. 2025) | Zero-shot test of the trained weights on real service telemetry; see §2.4 |

### 2.2 What is real and what is simulated

| Component | Status |
|---|---|
| When flights happen, when they are cancelled or delayed | **Real** (BTS) |
| How many passengers book them | **Real** (DB1B) |
| Shape and burstiness of cloud request traffic | **Real** (Azure) |
| How hard production clouds run their machines | **Real** (Alibaba) |
| Fault types, severities and durations | **Real** (Google Cloud). Frequency is **oversampled** to get enough positives; duration is capped at 4 h (time-to-failover assumption) |
| How a fault develops | **Assumption**: 70% worsen gradually over 2–20 min (incidents that start as "elevated latency"), 30% are abrupt (hardware / data-centre failures) |
| Requests generated per passenger event (look-to-book 1000:1, rebooking 6 legacy calls per passenger, and so on) | **Assumption**, documented in `WorkloadConfig` and stress-tested in [sensitivity.md](../results/real_data/sensitivity.md) |
| The five airline services and their queueing behaviour | **Simulated** in 10-second steps: Little's law, a fixed mainframe connection pool, lagged autoscaling, retry amplification. A cascade must last ≥ 30 s to count |

### 2.3 Results on real-workload data (honest)

Event split: trained on Southwest Dec-2022 (400 train / 100 val episodes), tested on **Delta Jul-2024** (500 episodes, never seen in training).

Simulator v2: 10-second steps, gradually worsening faults, 30-second persistence. Early warning gives the model the last 2 minutes of 30-second readings before (cascade − 2 min).

| Model | Early warning (2 min before cascade) AUC | Early F1 | Detection (full window) AUC | Detection F1 |
|---|---|---|---|---|
| **BACCP hetero-RGCN** (`cascade_predictor`) | **0.625** | **0.554** | 0.870 | 0.829 |
| Logistic regression (same features) | 0.589 | 0.524 | **0.911** | **0.847** |
| Static drift rule (ε ≥ 45%) | 0.595 | 0.047 (almost never fires in time) | 0.855 | 0.804 |

What this means:

- **Detection works on an unseen airline and event** (AUC 0.87–0.91), but a plain logistic regression on the same features does as well as BACCP or better. The graph model does not yet earn its complexity on this data.
- **Early warning is the hard, open problem.** BACCP ranks best (0.63), but every model is close to chance. The static ε ≥ 45% dashboard threshold almost never fires before the cascade. All of this is far below the ~0.96 reported on AI-generated data. **That drop is the main justification for using real data**: the synthetic benchmark overstated performance.
- **Why early warning is hard here** ([`early_warning_diagnostics.json`](../results/real_data/early_warning_diagnostics.json)):
  - Only **54%** of cascades show *any* precursor 2 minutes before they start (abrupt faults and fast ramps give none).
  - Early in a ramp, a fault that will cascade and one that will not look alike.
  - Accuracy falls as the horizon grows (best simple model: AUC 0.72 at 30 s, about 0.60 at 2 min, below 0.65 at 5 min).
- **Leakage check (important).** A *random* split inside Southwest scores AUC **0.96** at 2 minutes, but splitting by day gives **0.67** and the cross-airline split **0.61**. Overlapping windows of the same fault leak across a random split. This is exactly why we use event/day splits, and why random-split results (including our old synthetic ones) should not be trusted.
- Next steps: more airline-months, per-airline normalisation and threshold calibration, and more history per sample.
- Full outputs: [`results/real_data/cascade_metrics_early.json`](../results/real_data/cascade_metrics_early.json), [`cascade_metrics_full.json`](../results/real_data/cascade_metrics_full.json). Figures: [`real_inputs.png`](../results/real_data/real_inputs.png), [`southwest_dec2022_workload.png`](../results/real_data/southwest_dec2022_workload.png).

### 2.4 External validation on real microservice telemetry (RCAEval Train-Ticket)

The services in §2.3 are simulated. To check that the predictor is not only learning our simulator, we ran it on **RCAEval RE1-TT** ([arXiv 2412.17015](https://arxiv.org/abs/2412.17015), [Zenodo 14590730](https://zenodo.org/records/14590730), CC-BY 4.0). This is real per-second telemetry from Train-Ticket, a ticket-booking microservice system, under 125 injected faults: 5 fault types (cpu, mem, disk, delay, loss) × 5 root-cause services × 5 repetitions.

**How it maps onto BACCP.** Train-Ticket services are grouped by function onto the 5 nodes: order/booking/seat/payment → reservations; route/train/station scheduling → crew; luggage consignment/food/assurance → baggage; the `ts-ui-dashboard` edge gateway → boundary-gateway; the MongoDB/MySQL pods and the account services that front them (auth, user, security) → legacy-core. Features use the unchanged `_node_features` normalisation. The full mapping is in [`rcaeval.py`](../ai-models/real_data/rcaeval.py).

**Labels.** A fault counts as a *cascade* when it spreads beyond its root-cause service: some other service's p90 latency over the 5 minutes after injection is at least 2× its pre-fault median and at least 8 robust z-scores above it. To check this rule for noise, we applied it to two fault-free windows before each injection. It fired in **1 of 125** of those placebo windows. Under this rule 71 of 125 faults cascade: mem 23/25, delay 20/25, loss 17/25, cpu 9/25, disk 2/25. Each case gives two samples of 5 one-minute snapshots: one before injection (label 0) and one after (label = cascade).

| Model (250 windows, 71 positive) | AUC, all windows | F1 @ 0.5 | AUC, post-injection only (cascade vs contained fault) |
|---|---|---|---|
| BACCP, trace-trained `trace_full` (detection) weights, **zero-shot** | 0.822 | 0.440 (precision 1.0, recall 0.28) | 0.704 |
| BACCP, trace-trained `trace_early` weights, **zero-shot** | 0.200 (inverted) | 0.000 | 0.205 |
| Static drift rule (ε ≥ 45%) | 0.500 | 0.000 (never fires) | 0.500 |
| BACCP trained on RCAEval (leave-one-root-service-out CV) | 0.843 | 0.705 | **0.834** |
| Logistic regression trained on RCAEval (same CV, same features) | **0.860** | **0.708** | 0.752 |

What this means:

- **Detection transfers; early warning does not; calibration does not.** The detection weights, trained only on our trace-driven simulator, rank real Train-Ticket windows well above chance (AUC 0.82) with no retraining, but at the 0.5 threshold they catch only 28% of cascades, so the threshold must be recalibrated per deployment. The early-warning weights rank Train-Ticket windows *backwards* (AUC 0.20): what they learned about the last 2 minutes of a ramp does not carry over to a different system with abrupt injected faults.
- **The drift rule does not transfer.** RE1 has no gateway queue depth and no per-call completion counts, so ε can only use gateway latency and errors, and it never reaches 45%. Raw ε ranks windows only slightly better than chance (AUC 0.60).
- **When trained on real data, BACCP holds up under location shift.** When each fault location is held out in turn, BACCP is close to logistic regression on all windows (0.84 vs 0.86). It is clearly better at the harder question of *which* faults will spread beyond their service (0.83 vs 0.75).
- **Caveats.** There are 125 cases, one seed and one system, so differences of about ±0.03 AUC are within noise. Faults in RE1 are injected abruptly, so this tests detection, not early warning. Train-Ticket has no mainframe, and its databases stand in for the legacy core.
- Full output, including per-case labels and the service-to-node mapping: [`results/real_data/rcaeval_metrics.json`](../results/real_data/rcaeval_metrics.json).

---

## 3. Glossary of the terms from the review

**Workload / cluster workload dataset (also "cluster trace").** A log recorded from a real production computing cluster: which jobs, VMs or containers ran, what resources they requested and used (CPU, memory) and when. Researchers use these to model realistic load instead of inventing it. The main public ones are the Google Borg traces (2011, 2019), the Alibaba cluster traces (2017, 2018, microservices 2021/2022, GPU 2020) and the Azure VM traces (2017, 2019).

**Azure dataset (Azure Public Dataset).** Microsoft's collection of real anonymised Azure traces on GitHub: VM utilisation (2017, 2019), Azure Functions serverless invocations (2019, 2021), blob access, and LLM inference traces. We use **Azure Functions 2019**: per-minute invocation counts of production functions over 14 days.

**Generating reservation workload.** Turning *business activity* into *IT requests per second*. An airline's reservation system receives:
- **Searches**: about 1,000 per booking (the *look-to-book ratio*; Amadeus reported ~1000:1 and OAG now reports 10,000:1 or more, see [PhocusWire](https://www.phocuswire.com/Airline-system-look-to-book-ratios-soar-expected-to-go-10x-higher), [PhocusWire 2024](https://www.phocuswire.com/news/distribution/look-to-book-ai-rise-search-cost-airlines)).
- **Bookings**, **check-ins** (24 h to 40 min before departure), **baggage events**, **crew roster updates**, and **rebookings** when flights are cancelled.

We generate each stream from real BTS flights and DB1B passengers ([`reservation_workload.py`](../ai-models/real_data/reservation_workload.py)).

**How the cloud manages that workload.**
- **Caching**: searches are served from a cloud availability cache, as in Amadeus's Google-Cloud availability service with Lufthansa.
- **Autoscaling**: replicas are added when utilisation passes a target (Alibaba data shows targets are low, ~20–30%).
- **Load balancing** and **retries**.
- **Rate limiting / circuit breaking.**

What the cloud *cannot* do is scale the legacy system of record behind it. That is where BACCP focuses.

**Simulated airline services.** Software stand-ins for the airline's reservations, crew, baggage, gateway and mainframe services, used because real airline systems cannot be broken on purpose. We have two levels: the Docker testbed (`testbed/services/`) and the trace-driven queueing simulator. The closest open-source analogue is **Train-Ticket**, a 40+ microservice ticket-booking benchmark used in RCAEval.

**BTS data.** The U.S. Bureau of Transportation Statistics publishes airline data through **TranStats**:
- **On-Time Performance**: every domestic flight with its delays and cancellations.
- **DB1B**: a 10% sample of tickets.
- **T-100**: segment and market traffic.

All are public domain and free to download.

**Airline booking dataset.** Real airline booking records are commercially confidential. The best public real source is **BTS DB1B** (actual tickets, 10% sample, with passengers, fares and itineraries). Kaggle datasets such as "Flight Price Prediction" (EaseMyTrip, 300k scraped Indian bookings with `days_left`) are real-scraped but need a Kaggle login. The widely shared BA `customer_booking.csv` is from a Forage job simulation and its provenance is unclear, so we do not rely on it.

**Cloud service-health dataset.** Records of when cloud services were degraded or down:
- **Status-page histories**: Google Cloud's `incidents.json`, AWS and Azure status pages.
- **AIOps benchmarks** with labelled faults: RCAEval, the AIOps Challenge datasets, Loghub logs.

We use **902 real Google Cloud incidents** for fault realism.

**Other terms worth knowing for the next review:**
- **Trace-driven simulation**: running a simulator with real recorded inputs instead of invented ones.
- **Ground truth**: the real, verified label (here, BTS disruption windows).
- **Data provenance**: where every piece of data came from.
- **Event split / out-of-distribution test**: testing on a different real event than the one trained on.
- **Look-to-book ratio**: searches per booking.
- **Headroom**: spare capacity above normal peak.
- **Retry amplification**: retries multiply load during failures.
- **Early warning vs detection**: predicting before a failure versus noticing it after it starts.

---

## 4. Reproduce

```bash
pip install -r ai-models/requirements.txt matplotlib
python ai-models/real_data/download.py          # ~1.2 GB of real data into data/raw/ (git-ignored)
python ai-models/real_data/prepare.py           # -> ai-models/data/real_processed/ (3 MB, committed)
python ai-models/real_data/build_trace_dataset.py
python ai-models/real_data/train_on_trace.py --mode early
python ai-models/real_data/train_on_trace.py --mode full
python ai-models/real_data/report_figures.py
python ai-models/real_data/early_warning_diagnostics.py   # leakage / horizon / visibility checks
python ai-models/real_data/walkthrough.py --pause         # step-by-step demo of the whole pipeline
python ai-models/real_data/download.py rcaeval  # RCAEval RE1-TT, 280 MB (size checked before download)
python ai-models/real_data/rcaeval.py           # -> results/real_data/rcaeval_metrics.json (~1 min)
python -m pytest ai-models/tests/test_real_data.py ai-models/tests/test_rcaeval.py
```

The processed artifacts are committed, so steps 3–6 run without downloading anything.

## 5. Limitations (stated up front)

- The airline IT services are simulated. No real airline telemetry is public.
- Per-passenger request multipliers are assumptions. The sensitivity table shows that the conclusion (demand alone rarely saturates; demand plus a fault cascades) holds for rebooking multipliers of 2–6, and that pure-demand saturation appears from about 6–8 calls per passenger at 1.25× headroom.
- Azure's diurnal profile is in UTC from an unknown region and is used only as a shape.
- DB1B counts origin–destination journeys, so it under-counts per-segment passengers for connecting itineraries.
- Fault frequency is oversampled relative to real Google Cloud rates so that cascades are frequent enough to learn from.
- How faults develop (70% gradual over 2–20 min, 30% abrupt) is an assumption, not measured.
- The workload is per-minute data interpolated to 10-second steps, so sub-minute bursts are smoothed.
- Results come from one seed and two carrier-months. More months (`download.py bts_ontime --param year=Y month=M`) will tighten the estimates.
