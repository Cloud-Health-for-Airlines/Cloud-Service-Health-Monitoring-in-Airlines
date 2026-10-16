# BACCP Degradation Prediction: Evaluation Protocol & Contract

## 1. Dataset Counts and Provenance
- **Total Incidents**: 275 whole trajectories across 6 operational regimes.
- **Trajectory Duration**: 30 discrete steps per incident ($dt = 20.0\,\text{s}$, duration $= 600.0\,\text{s}$).
- **Split Partitions**:
  - `train`: 120 incidents (43.6%) — used for fitting baseline parameters and ablation models.
  - `val`: 30 incidents (10.9%) — used for operating point tuning and probability calibration.
  - `cal`: 25 incidents (9.1%) — historical holdout used for isotonic temperature calibration.
  - `standard_test`: 25 incidents (9.1%) — clean held-out evaluation cohort.
  - `capacity_loss_test`: 25 incidents (9.1%) — held-out physical capacity reduction stress cohort.
  - `boundary_slowdown_test`: 25 incidents (9.1%) — held-out boundary tunnel latency & sync drift stress cohort.
  - `surge_test`: 25 incidents (9.1%) — held-out workload surge (2.5x traffic bursts) stress cohort.
- **Split Integrity**: All origin windows from an incident strictly reside within the incident's split. No window leakage across splits.
- **Holdout Status**: Test sets have been evaluated in preceding development iterations; they are designated as **previously inspected holdouts** rather than pristine uninspected test sets.

## 2. Operational Definitions & Eligibility
- **Sampling Interval**: $\Delta t = 20.0\,\text{seconds}$.
- **Horizons**:
  - $H=1$: $20\,\text{s}$ forward window (3,250 evaluated origin samples in clean test).
  - $H=5$: $100\,\text{s}$ forward window (2,750 evaluated origin samples in clean test).
  - $H=15$: $300\,\text{s}$ forward window (1,500 evaluated origin samples in clean test).
- **Sustained Degradation Ground Truth**: $\ge 3$ consecutive 20-second steps ($60\,\text{s}$) violating SLO thresholds.
- **Origin Eligibility**:
  - Warmup requirement: $t_{\text{origin}} \ge 3$ ($60\,\text{s}$) to allow rolling mean/max extraction.
  - Forward-window constraint: $t_{\text{origin}} \le T - H - 1$ to ensure full window observability.
  - Censoring: Trailing windows with unresolved breach status (`check_window_label_observability`) are strictly excluded.
- **Current State vs Early Warning Separation**:
  - Current-state detection evaluates accuracy on origins where the service is already in sustained degradation.
  - Early-warning evaluation is restricted strictly to origins where the service is currently nominal, followed by a confirmed onset in $(t, t+H \cdot \Delta t]$. An alert at or after onset does NOT count as an early warning.
- **Actionable Lead Time**: Minimum $20.0\,\text{s}$ (1 step before onset).
- **False Alarm Accounting**: Evaluated on healthy exposure origins with a 100-second ($5 \times 20\text{s}$) deduplication cooldown.

## 3. Evaluated Model Cohort
1. **Static SLO Threshold Alerting**: Triggers alert if current metric exceeds 75% of SLO threshold.
2. **Training-Fitted Z-Score Baseline**: Per-service anomaly baseline fitted on `train`, operating threshold selected on `val`.
3. **Model A**: Telemetry-only gradient boosted model (22 features).
4. **Model B**: Telemetry + Workload Forecast primary model (35 features). Deployed thresholds: $H=1: 0.280$, $H=5: 0.300$, $H=15: 0.320$.
5. **Model C**: Temporal comparator GRU model ($4 \times 17$ features).

## 4. Priority-Weighted Macro Aggregation
Three cloud domains are combined using architectural priority weights:
$$\text{Metric}_{\text{macro}} = \frac{1.5 \cdot M_{\text{reservations}} + 1.0 \cdot M_{\text{crew}} + 0.5 \cdot M_{\text{baggage}}}{3.0}$$
Boundary Gateway and Legacy Core are reported separately as infrastructure interconnect components.
