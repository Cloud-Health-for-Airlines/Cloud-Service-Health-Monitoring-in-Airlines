# BACCP ML Inference & Telemetry API Documentation

Version: `baccp-api-v1.5.0`  
Protocol: HTTP REST / JSON  
Default Port: `8000`

---

## 1. Overview & Architecture

The BACCP Backend API integrates three core machine learning modules:
1. **Per-Service Workload Forecasting** (`/forecast/workload`)
2. **Degradation & Failure Risk Prediction** (`/predict/degradation`)
3. **Transparent Attributions & Root-Cause Localization** (`/explain/{service}`)
4. **Prometheus Observability Metrics** (`/metrics`)

### Telemetry Adapter & Mode Separation

The backend features an explicit **Telemetry Adapter** (`backend/ml/telemetry_adapter.py`) supporting two modes:
- **`replay` (Default / Demonstration)**: Feeds verified incident history from saved incident trajectories (`degradation_dataset.json`). Enforces a configurable replay clock and strictly checks that telemetry staleness does not exceed `MAX_TELEMETRY_STALENESS_SECONDS = 60.0s`. Every response identifies `telemetry_mode: "replay"` and `data_source: "REPLAY_INCIDENT_STREAM"`.
- **`live`**: Connects to actual backend/eBPF telemetry. If live application metrics (e.g. measured latency percentiles, queue depths, error rates) are unavailable, returns structured HTTP 503 (`UNAVAILABLE_INSUFFICIENT_TELEMETRY`) rather than silently filling nominal baselines.

### Live Telemetry Gaps Documentation
- **eBPF Probe Coverage**: The active eBPF probe captures `tcp_connection_attempt` counts only. Application-level metrics (measured latency percentiles, queue backlog, completed rates, and error rates) are not instrumented in live containers.
- **Latency Measurement**: Graph edges expose static average latency (`avg_latency_ms`); they do NOT measure or report p95 or p99 percentiles.
- **Legacy Tier**: `legacy-core` is an uninstrumented mainframe backend; internal thread pools and queues cannot be directly scraped.
- **Time-Series Storage**: The local backend lacks an active persistent time-series database (e.g. InfluxDB/Prometheus TSDB) for multi-step historical windowing.

---

## 2. Configuration & Startup

### Environment Variables
Configure via environment variables or a `.env` file:
```bash
# Telemetry Mode: 'replay' (recommended for demo/test) or 'live'
export BACCP_TELEMETRY_MODE=replay

# Binding Configuration
export BACCP_HOST=0.0.0.0
export BACCP_PORT=8000
```

### Server Startup
```bash
# Start server using Python 3
python3 backend/api/app.py --host 0.0.0.0 --port 8000
```

---

## 3. Endpoints Specification

### 3.1. GET `/forecast/workload`

Generates point workload forecasts and 90% conformal prediction intervals for a target microservice.

#### Query Parameters
| Parameter | Type | Required | Description |
|---|---|---|---|
| `service` | string | **Yes** | Target service: `reservations`, `crew`, `baggage`, `boundary-gateway`, `legacy-core` |
| `horizon` | integer | **Yes** | Forecast steps: `1` (20s), `5` (100s), `15` (300s) |
| `replay_step` | integer | No | Telemetry origin step index (replay mode) |
| `replay_time_sec` | float | No | Telemetry origin elapsed seconds (replay mode) |
| `incident_id` | string | No | Saved incident ID to replay (replay mode) |

#### Example cURL
```bash
curl -s "http://127.0.0.1:8000/forecast/workload?service=reservations&horizon=5" | jq
```

#### Example Response (HTTP 200)
```json
{
  "service": "reservations",
  "horizon_steps": 5,
  "horizon_seconds": 100.0,
  "observation_timestamp_sec": 300.0,
  "target_timestamp_sec": 400.0,
  "data_source": "REPLAY_INCIDENT_STREAM",
  "telemetry_mode": "replay",
  "data_provenance": "SYNTHETIC_SIMULATED",
  "incident_id": "inc_standard_test_0175",
  "snapshot_id": "replay:inc_standard_test_0175:step_15:t_300.0",
  "freshness_seconds": 0.0,
  "model_name": "gradient_boosting",
  "model_version": "baccp-workload-v1.1.0",
  "predicted_demand_req_s": 51.39,
  "conformal_interval": {
    "lower": 42.66,
    "upper": 60.12,
    "width": 17.46,
    "coverage_level": 0.90
  },
  "conformal_coverage": 0.90,
  "capacity_margin_req_s": 18.61,
  "available_headroom_req_s": 18.61,
  "is_overloaded": false,
  "candidate_models": {
    "gradient_boosting": 51.39,
    "autoregressive_ridge": 49.82,
    "persistence": 48.0,
    "seasonal_naive": 47.5
  },
  "reliability": {
    "is_stale": false,
    "is_carried_forward": false,
    "warnings": []
  },
  "cached": false
}
```

---

### 3.2. GET `/predict/degradation`

Generates continuous latency/backlog predictions, calibrated forward-window degradation probability, and advisory alert decisions.

#### Query Parameters
| Parameter | Type | Required | Description |
|---|---|---|---|
| `service` | string | **Yes** | Target service: `reservations`, `crew`, `baggage`, `boundary-gateway`, `legacy-core` |
| `horizon` | integer | **Yes** | Forecast steps: `1` (20s), `5` (100s), `15` (300s) |
| `model_variant` | string | No | `model_b_telemetry_forecast` (default), `model_a_telemetry_only`, `model_c_temporal_comparator` |
| `replay_step` | integer | No | Telemetry origin step index (replay mode) |
| `replay_time_sec` | float | No | Telemetry origin elapsed seconds (replay mode) |

#### Example cURL
```bash
curl -s "http://127.0.0.1:8000/predict/degradation?service=reservations&horizon=5" | jq
```

#### Example Response (HTTP 200)
```json
{
  "service": "reservations",
  "horizon_steps": 5,
  "horizon_seconds": 100.0,
  "observation_timestamp_sec": 300.0,
  "target_timestamp_sec": 400.0,
  "data_source": "REPLAY_INCIDENT_STREAM",
  "telemetry_mode": "replay",
  "data_provenance": "SYNTHETIC_SIMULATED",
  "incident_id": "inc_standard_test_0175",
  "snapshot_id": "replay:inc_standard_test_0175:step_15:t_300.0",
  "model_variant": "model_b_telemetry_forecast",
  "model_version": "baccp-degradation-inference-v1.3.0",
  "input_reliability": {
    "status": "SYNTHETIC_SIMULATED",
    "has_history_padding": false,
    "has_carried_forward": false,
    "boundary_stream_missing": false,
    "imputed_fields_count": 0,
    "reliability_warnings": []
  },
  "calibrated_event_probability": 0.0412,
  "alert_threshold": 0.300,
  "advisory_alert_decision": false,
  "alert_issued": false,
  "current_observed_state": {
    "latency_ms": 22.4,
    "backlog_requests": 4.1,
    "error_rate": 0.0,
    "boundary_sync_drift_pct": 12.4,
    "is_currently_degraded": false,
    "current_degradation_type": "nominal",
    "current_severity": "nominal"
  },
  "predicted_future_risk": {
    "continuous_predictions": {
      "latency_ms": 24.1,
      "backlog_requests": 6.2,
      "boundary_sync_drift_pct": 13.0
    },
    "estimated_time_to_degradation_seconds": null,
    "time_to_degradation_reason": "service_nominal_no_active_alert",
    "predicted_degradation_type": "nominal",
    "predicted_severity": "nominal",
    "slo_thresholds": {
      "latency_ms": 100.0,
      "backlog_requests": 50.0,
      "error_rate": 0.05
    },
    "upstream_workload_forecast": {
      "forecast_demand_req_s": 51.39,
      "nominal_capacity_req_s": 70.0,
      "rollout_projected_backlog": 0.0,
      "rollout_projected_latency_ms": 20.0
    }
  },
  "cached": false
}
```

---

### 3.3. GET `/explain/{service}`

Computes exact TreeSHAP feature attributions and counterfactual graph bottleneck localization for the target service prediction.

#### Path & Query Parameters
| Parameter | Type | Required | Description |
|---|---|---|---|
| `service` | string | **Yes** (Path) | Service name: `reservations`, `crew`, `baggage`, `boundary-gateway`, `legacy-core` |
| `horizon` | integer | No (Query) | Forecast steps: `1`, `5` (default), `15` |
| `model_variant` | string | No (Query) | Model variant (default `model_b_telemetry_forecast`) |

#### Example cURL
```bash
curl -s "http://127.0.0.1:8000/explain/reservations?horizon=5" | jq
```

#### Example Response (HTTP 200)
```json
{
  "service": "reservations",
  "horizon_steps": 5,
  "horizon_seconds": 100.0,
  "observation_timestamp_sec": 300.0,
  "data_source": "REPLAY_INCIDENT_STREAM",
  "telemetry_mode": "replay",
  "data_provenance": "SYNTHETIC_SIMULATED",
  "incident_id": "inc_standard_test_0175",
  "snapshot_id": "replay:inc_standard_test_0175:step_15:t_300.0",
  "model_variant": "model_b_telemetry_forecast",
  "model_version": "baccp-degradation-explainer-v1.0.0",
  "calibrated_event_probability": 0.0412,
  "alert_threshold": 0.300,
  "advisory_alert_decision": false,
  "attribution_method": "TreeSHAP",
  "attribution_target": "raw_log_odds_margin",
  "expected_value": -2.8941,
  "output_value": -3.1250,
  "reconstruction_error": 8.88e-16,
  "top_risk_increasing_drivers": [
    {"feature": "wf_forecast_demand", "attribution": 0.1245, "observed_value": 51.39, "reference_mean": 48.0},
    {"feature": "curr_latency_ms", "attribution": 0.0812, "observed_value": 22.4, "reference_mean": 20.0}
  ],
  "top_risk_decreasing_drivers": [
    {"feature": "wf_available_headroom", "attribution": -0.2514, "observed_value": 18.61, "reference_mean": 22.0},
    {"feature": "curr_backlog", "attribution": -0.1840, "observed_value": 4.1, "reference_mean": 8.0}
  ],
  "category_attributions": {
    "workload": 0.092,
    "saturation": -0.435,
    "network_latency": 0.081,
    "boundary_gateway": 0.012,
    "upstream_workload_forecast": -0.126
  },
  "localization": {
    "suspected_contributing_node": "unknown_unlocalized",
    "propagation_bottleneck": "none_identified",
    "localization_status": "abstain_unknown_unlocalized",
    "is_tie": false,
    "is_abstained": true,
    "abstention_reason": "No candidate node intervention produced degradation risk reduction above noise floor (max risk reduction 0.000000 < 0.05).",
    "candidate_rankings": [
      {
        "candidate_node": "reservations",
        "node_generation": "cloud",
        "counterfactual_probability": 0.0,
        "counterfactual_probability_raw": 0.0,
        "signed_risk_delta": 0.0,
        "signed_risk_delta_raw": 0.0,
        "risk_reduction_if_nominal": 0.0,
        "risk_reduction_raw": 0.0,
        "relative_impact_pct": 0.0
      }
    ]
  },
  "incident_brief": "### BACCP INCIDENT EXPLANATION BRIEF: RESERVATIONS [H=5 (100s)]\n- **Affected Service**: reservations (Generation: cloud)\n- **Prediction Origin Timestamp**: 300.0s (t_origin)\n- **Forecast Window**: [300.0s, 400.0s] (H=5 steps / 100s ahead, Target Ts: 400.0s)\n- **Model Variant**: model_b_telemetry_forecast [Schema: baccp-degradation-features-v1.4.0]\n- **Observed SLO State at Origin**: Within Nominal SLO Limits (Observed latency 18.8ms < 100.0ms, backlog 0.0 < 50.0)\n- **Future Predicted Risk**: Degradation Risk: 0.0% (Calibrated event probability: 0.0000)\n- **Alert Decision**: NO ALERT ISSUED (Predicted risk 0.0% < threshold 30.0%)\n- **Conditional Time-to-Onset**: Unavailable: Conditional time-to-degradation estimate is gated to active alert states (alert_issued=False; risk 0.0% < threshold 30.0%).\n- **Predicted Type & Severity**: NOMINAL / NOMINAL\n\n#### Top Supporting Drivers (Model Attribution):\n1. **curr_latency_ms** = 18.76 ms (Ref: 28.8451 ms) | Attribution: +0.2594 log-odds [Network]\n2. **wf_rollout_arrivals** = 4973.2 requests (Ref: 4700.8053 requests) | Attribution: +0.1802 log-odds [Workload]\n3. **curr_completed_rate** = 49.54 req/s (Ref: 40.3428 req/s) | Attribution: +0.159 log-odds [Workload]\n\n#### Graph Localization & Suspected Contributing Origin:\n- **Suspected Contributing Node**: unknown_unlocalized (Generation: unknown)\n- **Propagation Bottleneck**: none_identified\n- **Verified Injected Origin**: UNAVAILABLE\n\n#### Data Provenance & Reliability Limitations:\n- **Telemetry Provenance**: SYNTHETIC_SIMULATED (eBPF captures TCP attempts; application latency/backlog simulated)\n- **Input Reliability Status**: SYNTHETIC_SIMULATED (Warnings: None (Nominal inputs))\n- **Attribution Methodology**: TreeSHAP (Reconstruction Error: 4.44e-15)\n- **Causality Disclaimer**: Attributions measure statistical model sensitivity; they do NOT establish physical causality.\n",
  "methodology_disclaimer": "Feature attributions (Tree SHAP / Integrated Gradients) measure statistical sensitivity of the trained prediction models to inputs. They do NOT constitute proof of physical causality or real-world root cause.",
  "cached": false
}
```

---

### 3.4. GET `/metrics`

Exposes standard Prometheus metrics in text exposition format (version 0.0.4) using thread-safe collectors.

#### Concurrency & Metric Responsiveness
The server runs under `ThreadingHTTPServer`. CPU-heavy explanations are bounded by a semaphore (`max 4 concurrent/queued requests`), ensuring that `/metrics` responds with low latency (< 200ms) even while explanations run in parallel.

#### Example cURL
```bash
curl -s "http://127.0.0.1:8000/metrics"
```

#### Metrics Excerpt
```text
# HELP baccp_predicted_degradation_probability Calibrated failure degradation probability by service and horizon
# TYPE baccp_predicted_degradation_probability gauge
baccp_predicted_degradation_probability{horizon="5",service="reservations"} 0.0412
baccp_predicted_degradation_probability{horizon="5",service="crew"} 0.0210

# HELP baccp_advisory_alert_episodes_total Total deduplicated advisory alert episodes by service and horizon
# TYPE baccp_advisory_alert_episodes_total counter
baccp_advisory_alert_episodes_total{horizon="1",service="reservations"} 0
baccp_advisory_alert_episodes_total{horizon="5",service="reservations"} 0

# HELP baccp_workload_forecast_absolute_error Mean absolute demand error of workload forecast after future observation arrives
# TYPE baccp_workload_forecast_absolute_error gauge
baccp_workload_forecast_absolute_error{horizon="1",service="baggage"} 34.770

# HELP baccp_telemetry_freshness_seconds Elapsed seconds since latest telemetry observation by service
# TYPE baccp_telemetry_freshness_seconds gauge
baccp_telemetry_freshness_seconds{service="reservations"} 0.00

# HELP baccp_telemetry_available Telemetry stream availability status (1=available, 0=unavailable)
# TYPE baccp_telemetry_available gauge
baccp_telemetry_available{service="reservations"} 1
```

---

## 4. Alert Threshold Policy & Provenance

All API endpoints (`/predict/degradation`, `/explain/{service}`, incident briefs, alert-gated timing, and Prometheus alert counters) share one unified alert policy configuration:

| Horizon | Alert Threshold | Target Evaluation Metric |
|---|---|---|
| **$H=1$ (20s)** | `0.280` | High-precision immediate intervention trigger |
| **$H=5$ (100s)** | `0.300` | Balanced actionable early warning ($F_1 = 0.584$, Recall = 0.90) |
| **$H=15$ (300s)** | `0.320` | Conservative long-range planning threshold |

### Provenance Gap Disclosure
These threshold values represent operational policy parameters predeclared during Task 4 evaluation to balance actionable recall against false alarms under asymmetric degradation costs. They were **not** derived from a formal cross-validation hyperparameter grid search and must not be described as mathematically validated optimal thresholds. Model weights and probability calibration remain strictly frozen.

---

## 5. Replay Freshness, Clock Constancy & Forecast Error Timing

### 5.1. Freshness Checks
- Freshness is validated independently per required service and boundary stream:
  $$\text{Freshness} = T_{\text{clock}} - T_{\text{latest}}(\text{service})$$
- If $\text{Freshness} > 60.0\,\text{s}$, the API returns HTTP 422 `STALENESS_LIMIT_EXCEEDED`. A fresh neighbouring service **cannot** mask stale requested-service data.

### 5.2. Clock Constancy
- The requested replay clock timestamp is strictly preserved and remains identical across forecast horizons ($H=1, 5, 15$). Replay times are not rewound or clamped based on horizon.
- Replay times must be finite and non-negative; non-finite or negative values return HTTP 400 `INVALID_REPLAY_PARAMETER`.
- Requests with fewer than 4 historical steps return HTTP 422 `UNAVAILABLE_INSUFFICIENT_TELEMETRY`.

### 5.3. Forecast Error Timing & Incident Isolation
- Workload forecast absolute error is **not** recorded immediately upon forecast generation. The forecast is registered as pending under a collision-proof identity:
  $$(\text{incident\_id}, \text{service}, \text{horizon}, T_{\text{origin}}, T_{\text{target}})$$
- On subsequent snapshot requests where $T_{\text{clock}} \ge T_{\text{target}}$, pending forecasts for the matching incident are resolved using only observations available at that clock.
- Forecasts from one incident are never resolved by observations from another incident.
- Bounded memory (max 500 entries) and 10-minute TTL expiration prevent memory leaks.
- Repeated cache-hit requests do not duplicate pending forecasts or re-record observations.

---

## 6. Error Handling & Edge Cases

| Status Code | Error Code | Description |
|---|---|---|
| `400` | `INVALID_SERVICE` | Missing or unrecognized `service` parameter. |
| `400` | `INVALID_HORIZON` | Missing or non-integer `horizon` parameter. |
| `400` | `UNSUPPORTED_HORIZON` | Horizon not in `[1, 5, 15]`. |
| `400` | `INVALID_REPLAY_PARAMETER` | Non-finite or negative replay time or step. |
| `400` | `INVALID_MODEL_VARIANT` | Unrecognized model variant parameter. |
| `422` | `STALENESS_LIMIT_EXCEEDED` | Observation timestamp is $>60.0\,\text{s}$ behind replay clock. |
| `422` | `UNAVAILABLE_INSUFFICIENT_TELEMETRY` | Telemetry stream lacks required historical context ($<60\text{s}$). |
| `503` | `UNAVAILABLE_INSUFFICIENT_TELEMETRY` | Live mode requested but live source lacks required metrics. |
| `503` | `ML_MODELS_UNAVAILABLE` | Model weights bundle or required dependencies missing on server. |
| `503` | `EXPLANATION_CAPACITY_EXCEEDED` | Explanation queue has reached maximum concurrency limit (4 requests). |
| `504` | `EXPLANATION_TIMEOUT` | Explanation computation exceeded 10.0s execution timeout. |

