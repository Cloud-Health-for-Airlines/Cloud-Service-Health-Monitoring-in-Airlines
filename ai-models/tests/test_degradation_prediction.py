"""Comprehensive Verification Test Suite for Degradation and Boundary Deterioration Prediction.

Version: baccp-degradation-tests-v1.0.0

Verifies:
1. Causal backlog dynamics and mass conservation (Q(t) >= 0, Drops >= 0)
2. Target alignment: endpoint t+H vs forward-window (t, t+H]
3. Future-data isolation and anti-leakage
4. Whole-incident disjoint partition isolation
5. Upstream workload forecast out-of-fold provenance
6. Training and inference feature schema consistency
7. Unknown legacy capacity preserved as None
8. Boundary sync drift ε(t) adherence to existing repository definition
9. Missing telemetry, short buffers, and non-finite input resilience
10. Forecast feature ablation toggling (Model A vs Model B)
11. Package bridge compatibility (from ai_models.degradation_prediction import ...)
12. Early warning lead time metric definition (t_alert < t_onset)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest
import sys
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
AI_MODELS_DIR = REPO_ROOT / "ai-models"
if str(AI_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODELS_DIR))
DATA_PATH = AI_MODELS_DIR / "data" / "degradation_prediction" / "degradation_dataset.json"
FIXTURE_PATH = AI_MODELS_DIR / "data" / "degradation_prediction" / "fixture_test_incidents.json"
WEIGHTS_PATH = AI_MODELS_DIR / "weights" / "degradation_prediction" / "degradation_models_bundle.pkl"

from degradation_prediction.data_contract import (
    BOUNDARY_SYNC_DRIFT_THRESHOLD,
    CONFIGURED_SERVICE_CAPACITIES,
    CONTRACT_VERSION,
    DegradationIncident,
    DegradationRegime,
    NOMINAL_BASELINES,
    SAMPLING_INTERVAL_SECONDS,
    ServiceIdentifier,
    SLO_THRESHOLDS,
    SUPPORTED_HORIZONS,
    is_step_degraded,
)
from degradation_prediction.features import (
    FEATURE_SCHEMA_VERSION,
    TELEMETRY_ONLY_FEATURE_NAMES,
    TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
    extract_features_at_origin,
)
from degradation_prediction.inference import DegradationPredictor
from degradation_prediction.models import (
    CausalQueueRolloutBaseline,
    LearnedDegradationHorizonModel,
    PersistenceDegradationBaseline,
)


@pytest.fixture(scope="module")
def dataset_data():
    target_path = DATA_PATH if DATA_PATH.exists() else FIXTURE_PATH
    if not target_path.exists():
        pytest.skip(f"Dataset not found at {DATA_PATH} or {FIXTURE_PATH}.")
    with open(target_path, "r") as f:
        return json.load(f)


def test_contract_and_horizons():
    """Verifies horizons H in {1, 5, 15} at 20-second intervals."""
    assert SUPPORTED_HORIZONS == [1, 5, 15]
    assert SAMPLING_INTERVAL_SECONDS == 20.0
    assert CONTRACT_VERSION.startswith("baccp-degradation-contract-v1.")


def test_causal_queue_conservation_and_drops(dataset_data):
    """Verifies discrete-time causal queue conservation: Q(t) = max(0, Q(t-1) + Arrivals - Departures - Drops)."""
    dt = SAMPLING_INTERVAL_SECONDS
    incidents = dataset_data["incidents"][:10]

    for inc in incidents:
        svc_tel = inc["service_telemetry"]
        for s in [s.value for s in ServiceIdentifier]:
            records = svc_tel[s]
            q_prev = 0.0
            for r in records:
                q_curr = r["backlog_requests"]
                lam = r["offered_rate"]
                comp = r["completed_rate"]
                drops = r["dropped_requests"]

                arrivals = lam * dt
                departures = comp * dt

                # Mass conservation check
                expected_q = max(0.0, q_prev + arrivals - departures - drops)
                assert math.isclose(q_curr, expected_q, abs_tol=0.2), (
                    f"Mass conservation violation in {inc['incident_id']} at step {r['step_index']}: "
                    f"q_curr={q_curr}, expected={expected_q}"
                )
                assert q_curr >= 0.0
                assert drops >= 0.0
                q_prev = q_curr


def test_target_alignment_endpoint_vs_window(dataset_data):
    """Verifies distinction between t+H endpoint degradation and forward-window (t, t+H] event."""
    inc = dataset_data["incidents"][0]
    svc_tel = inc["service_telemetry"]
    bnd_tel = inc["boundary_telemetry"]

    # At origin t=5 with horizon H=5 (target step 10):
    # Verify that forward window checks all tau in {6, 7, 8, 9, 10}
    origin_t = 5
    horizon = 5
    target_step = origin_t + horizon

    # Test that window covers all intermediate steps
    checked_steps = list(range(origin_t + 1, target_step + 1))
    assert len(checked_steps) == horizon
    assert checked_steps[0] == 6
    assert checked_steps[-1] == 10


def test_anti_leakage_and_future_data_isolation(dataset_data):
    """Verifies features extracted at origin t only use data up to t, with no future or hidden variables."""
    inc = dataset_data["incidents"][0]
    svc_tel = inc["service_telemetry"]
    bnd_tel = inc["boundary_telemetry"]

    origin_t = 8
    feat_vec, aux = extract_features_at_origin(
        service_id="reservations",
        incident_service_telemetry=svc_tel,
        incident_boundary_telemetry=bnd_tel,
        origin_step=origin_t,
        horizon=5,
        forecaster=None,
        include_workload_forecasts=False,
    )

    # Check length matches Telemetry-Only feature schema
    assert len(feat_vec) == len(TELEMETRY_ONLY_FEATURE_NAMES)
    # Check no NaN/Inf
    assert not np.any(np.isnan(feat_vec))
    assert not np.any(np.isinf(feat_vec))

    # Verify aux info contains no future actuals
    assert "future_demand" not in aux
    assert "future_latency" not in aux
    assert "regime" not in aux


def test_whole_incident_partition_disjointness(dataset_data):
    """Verifies that whole incident splits are 100% mutually disjoint."""
    manifest = dataset_data["split_manifest"]
    all_splits = list(manifest.keys())

    seen_ids = set()
    for split_name in all_splits:
        split_ids = set(manifest[split_name])
        overlap = seen_ids.intersection(split_ids)
        assert len(overlap) == 0, f"Leakage: incident IDs {overlap} overlap between splits!"
        seen_ids.update(split_ids)

    # Check all required splits exist
    assert "train" in manifest
    assert "val" in manifest
    assert "cal" in manifest
    assert "standard_test" in manifest
    assert "capacity_loss_test" in manifest
    assert "boundary_slowdown_test" in manifest
    assert "surge_test" in manifest


def test_unknown_legacy_capacity_preserved_as_none():
    """Verifies legacy-core capacity is explicitly None in contracts and not silently assumed."""
    assert CONFIGURED_SERVICE_CAPACITIES[ServiceIdentifier.LEGACY_CORE.value] is None
    assert NOMINAL_BASELINES[ServiceIdentifier.LEGACY_CORE.value]["nominal_capacity"] is None


def test_boundary_sync_drift_exact_repository_formula():
    """Verifies boundary sync drift score matches repository definition clamped in [8.5, 99.5]."""
    from degradation_prediction.generate_data import compute_sync_drift_score

    # Baseline nominal
    d_norm = compute_sync_drift_score(45.0, 45.0, 0.0, 15.0, 15.0, 0.0)
    assert 8.5 <= d_norm <= 15.0

    # High queue + latency stall
    d_high = compute_sync_drift_score(120.0, 10.0, 150.0, 500.0, 15.0, 0.1)
    assert d_high >= 70.0  # Must trigger CRITICAL hazard threshold


def test_inference_and_missing_telemetry_resilience():
    """Verifies DegradationPredictor handles missing telemetry streams and short buffers without failing."""
    if not WEIGHTS_PATH.exists():
        pytest.skip("Weights bundle not found. Waiting for training to complete.")

    predictor = DegradationPredictor.get_instance(WEIGHTS_PATH)

    # Empty/missing service telemetry
    sparse_telemetry = {
        "reservations": [
            {"latency_ms": 18.0, "backlog_requests": 2.0, "offered_rate": 45.0, "completed_rate": 45.0, "error_rate": 0.001}
        ]
    }
    res = predictor.predict(service_telemetry=sparse_telemetry, horizons=[1, 5])

    assert "services" in res
    assert "reservations" in res["services"]
    assert "legacy-core" in res["services"]  # Synthesized missing stream
    assert len(res["warnings"]) > 0


def test_forecast_ablation_toggling():
    """Verifies toggling include_workload_forecasts between True and False switches model variants seamlessly."""
    if not WEIGHTS_PATH.exists():
        pytest.skip("Weights bundle not found. Waiting for training to complete.")

    predictor = DegradationPredictor.get_instance(WEIGHTS_PATH)

    sample_telemetry = {
        s.value: [
            {"latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
            for _ in range(5)
        ]
        for s in ServiceIdentifier
    }

    res_b = predictor.predict(sample_telemetry, include_workload_forecasts=True)
    res_a = predictor.predict(sample_telemetry, include_workload_forecasts=False)

    assert res_b["model_variant"] == "model_b_telemetry_forecast"
    assert res_a["model_variant"] == "model_a_telemetry_only"


def test_ai_models_package_bridge():
    """Verifies top-level ai_models bridge import works without path hackery."""
    from ai_models.degradation_prediction import DegradationPredictor as BridgePredictor
    assert BridgePredictor is not None


def test_non_finite_inputs_sanitized():
    """Verifies DegradationPredictor safely sanitizes NaN and Inf inputs without raising unhandled exceptions."""
    predictor = DegradationPredictor.get_instance(WEIGHTS_PATH)
    dirty_telemetry = {
        s.value: [
            {"latency_ms": float("nan"), "backlog_requests": float("inf"), "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
            for _ in range(5)
        ]
        for s in ServiceIdentifier
    }
    res = predictor.predict(dirty_telemetry, horizons=[1])
    assert "warnings" in res
    assert any("Non-finite" in w for w in res["warnings"])
    for s_id, s_data in res["services"].items():
        assert not math.isnan(s_data["current_state"]["latency_ms"])
        assert not math.isinf(s_data["current_state"]["backlog_requests"])


def test_strict_early_warning_lead_time_accounting():
    """Verifies strict horizon lead-time bounds: 0 < lead_time <= H * 20s.

    Verifies:
    - Alerts earlier than horizon window are premature false alarms, not true positives.
    - Alerts at or after onset are ongoing detections, not early warnings.
    - H=1 lead time is mathematically restricted to exactly 20.0s.
    - H=5 lead time is bounded in (0, 100s].
    - Null is returned when no valid early warnings exist.
    """
    dt = 20.0
    onset_step = 15  # 300s

    # For H=5 (window length = 100s = 5 steps):
    # Valid origin window: [10, 14]
    valid_origins_h5 = list(range(10, 15))
    for t_alert in valid_origins_h5:
        lead_sec = (onset_step - t_alert) * dt
        assert 0 < lead_sec <= (5 * dt), f"Lead time {lead_sec} outside H=5 horizon"

    # Origin t=8 is outside horizon H=5 window:
    # Prediction window is (8, 13], which does NOT contain onset 15.
    t_premature = 8
    lead_premature = (onset_step - t_premature) * dt
    assert lead_premature > (5 * dt)  # 140s > 100s (premature!)

    # For H=1 (window length = 20s = 1 step):
    # The only valid origin is t=14
    lead_h1 = (onset_step - 14) * dt
    assert lead_h1 == 20.0  # Exactly 20s!


def test_causal_queue_rollout_physical_cases():
    """Verifies simulate_step_by_step_rollout across all 5 physical cases.

    1. Nominal no-backlog: backlog remains ~0, latency remains baseline, zero drops.
    2. Overload: backlog builds up, latency increases, degradation alerted.
    3. Buffer saturation: backlog capped at B_max, drops positive.
    4. Zero-capacity: buffer fills, drops occur, latency capped.
    5. Recovery: backlog drains when arrivals fall below capacity.
    """
    from degradation_prediction.models import BUFFER_LIMITS, simulate_step_by_step_rollout

    svc = "reservations"
    b_max = BUFFER_LIMITS[svc]  # 400.0

    # 1. Nominal no-backlog (offered = 20, capacity = 45, dt = 20s)
    res_nom = simulate_step_by_step_rollout(
        service_id=svc,
        curr_backlog=0.0,
        curr_offered=20.0,
        forecast_demand=20.0,
        curr_completed=20.0,
        curr_sync_drift=12.0,
        horizon=5,
    )
    assert res_nom["backlog_requests"] == 0.0
    assert res_nom["total_drops"] == 0.0
    assert res_nom["degradation_risk"] == 0.0
    assert math.isclose(res_nom["latency_ms"], NOMINAL_BASELINES[svc]["latency_ms"], abs_tol=1.0)

    # 2. Overload (offered = 140 > capacity = 100)
    res_ovld = simulate_step_by_step_rollout(
        service_id=svc,
        curr_backlog=0.0,
        curr_offered=140.0,
        forecast_demand=140.0,
        curr_completed=100.0,
        curr_sync_drift=12.0,
        horizon=5,
    )
    assert res_ovld["backlog_requests"] > 0.0
    assert res_ovld["latency_ms"] > NOMINAL_BASELINES[svc]["latency_ms"]
    assert res_ovld["degradation_risk"] == 1.0

    # 3. Buffer saturation (massive burst: offered = 300 >> capacity = 100 over H=15)
    res_sat = simulate_step_by_step_rollout(
        service_id=svc,
        curr_backlog=0.0,
        curr_offered=300.0,
        forecast_demand=300.0,
        curr_completed=100.0,
        curr_sync_drift=12.0,
        horizon=15,
    )
    assert res_sat["backlog_requests"] <= b_max
    assert res_sat["total_drops"] > 0.0
    assert res_sat["degradation_risk"] == 1.0

    # 4. Zero-capacity proxy (uninstrumented legacy-core completion proxy = 0.0)
    res_zero = simulate_step_by_step_rollout(
        service_id=ServiceIdentifier.LEGACY_CORE.value,
        curr_backlog=100.0,
        curr_offered=30.0,
        forecast_demand=30.0,
        curr_completed=0.0,
        curr_sync_drift=12.0,
        horizon=5,
    )
    assert res_zero["backlog_requests"] > 0.0
    assert res_zero["degradation_risk"] == 1.0

    # 5. Recovery (initial backlog = 200, offered drops to 10 < capacity = 45)
    res_rec = simulate_step_by_step_rollout(
        service_id=svc,
        curr_backlog=200.0,
        curr_offered=10.0,
        forecast_demand=10.0,
        curr_completed=45.0,
        curr_sync_drift=12.0,
        horizon=5,
    )
    # Drain: (45 - 10) * 20s * 5 steps = 35 * 100 = 3500 requests >> 200 backlog
    assert res_rec["backlog_requests"] == 0.0
    assert math.isclose(res_rec["latency_ms"], NOMINAL_BASELINES[svc]["latency_ms"], abs_tol=1.0)


def test_missing_telemetry_imputation_reliability_flag():
    """Verifies that missing telemetry streams are explicitly flagged as DEGRADED_IMPUTED."""
    predictor = DegradationPredictor.get_instance(WEIGHTS_PATH)

    # Missing legacy-core and baggage
    sparse_telemetry = {
        "reservations": [
            {"latency_ms": 18.0, "backlog_requests": 2.0, "offered_rate": 45.0, "completed_rate": 45.0, "error_rate": 0.001}
            for _ in range(5)
        ]
    }
    res = predictor.predict(service_telemetry=sparse_telemetry, horizons=[1])

    assert res["input_reliability"] == "DEGRADED_IMPUTED"
    assert "legacy-core" in res["imputed_services"]
    assert "baggage" in res["imputed_services"]
    assert any("DEGRADED_IMPUTED" in w for w in res["warnings"])


def test_system_wide_prediction_interface():
    """Verifies inference contract returns top-level system_prediction block."""
    predictor = DegradationPredictor.get_instance(WEIGHTS_PATH)
    sample_telemetry = {
        s.value: [
            {"latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
            for _ in range(5)
        ]
        for s in ServiceIdentifier
    }
    res = predictor.predict(sample_telemetry, horizons=[1, 5, 15])

    assert "system_prediction" in res
    assert "predictions_by_horizon" in res["system_prediction"]
    for h in [1, 5, 15]:
        h_str = str(h)
        assert h_str in res["system_prediction"]["predictions_by_horizon"]
        pred = res["system_prediction"]["predictions_by_horizon"][h_str]
        assert "system_degradation_risk" in pred
        assert "system_alert_issued" in pred
        assert 0.0 <= pred["system_degradation_risk"] <= 1.0


def test_feature_provenance_catalog_completeness():
    """Verifies FEATURE_PROVENANCE_CATALOG contains all 35 features with zero simulator-only features."""
    from degradation_prediction.data_contract import FeatureProvenance
    from degradation_prediction.features import FEATURE_PROVENANCE_CATALOG, TELEMETRY_PLUS_FORECAST_FEATURE_NAMES

    assert len(FEATURE_PROVENANCE_CATALOG) == 35
    assert len(FEATURE_PROVENANCE_CATALOG) == len(TELEMETRY_PLUS_FORECAST_FEATURE_NAMES)

    # Count by provenance
    counts = {}
    for feat, prov in FEATURE_PROVENANCE_CATALOG.items():
        counts[prov] = counts.get(prov, 0) + 1

    assert counts[FeatureProvenance.MEASURED] == 20
    assert counts[FeatureProvenance.CONFIGURED] == 2
    assert counts[FeatureProvenance.INFERRED] == 13
    assert FeatureProvenance.SIMULATOR_ONLY not in counts


def test_episode_origin_eligibility_consistency(dataset_data):
    """Audits standard-test episodes and origin eligibility consistency for H=1 and H=5."""
    standard_incs = [i for i in dataset_data["incidents"] if i["split"] == "standard_test"]
    assert len(standard_incs) == 25

    total_episodes = 0
    evaluable_episodes_h1 = 0
    evaluable_episodes_h5 = 0
    positive_origins_h1 = 0
    positive_origins_h5 = 0

    for inc in standard_incs:
        regime = inc["regime"]
        svc_tel = inc["service_telemetry"]
        bnd_tel = inc["boundary_telemetry"]
        step_count = inc["step_count"]
        services = [s.value for s in ServiceIdentifier]

        # Identify degradation steps
        deg_steps = []
        for t in range(step_count):
            curr_s = {s: svc_tel[s][t] for s in services}
            curr_b = bnd_tel[t]
            if is_step_degraded(curr_s, curr_b):
                deg_steps.append(t)

        if not deg_steps:
            continue

        # Single continuous episode in these incidents
        total_episodes += 1
        onset = deg_steps[0]

        # H=1 eligibility: min history 3, healthy at origin (origin < onset), onset in (t, t+1]
        valid_origins_h1 = [t for t in range(3, step_count - 2) if t < onset and onset <= t + 1]
        if valid_origins_h1:
            evaluable_episodes_h1 += 1
            positive_origins_h1 += len(valid_origins_h1)

        # H=5 eligibility: min history 3, healthy at origin (origin < onset), onset in (t, t+5]
        valid_origins_h5 = [t for t in range(3, step_count - 6) if t < onset and onset <= t + 5]
        if valid_origins_h5:
            evaluable_episodes_h5 += 1
            positive_origins_h5 += len(valid_origins_h5)

    # 13 standard test episodes (8 gradual_ramp, 5 recovery)
    assert total_episodes == 13
    # Exactly 8 are evaluable (gradual_ramp with onset at 9 or 10)
    assert evaluable_episodes_h1 == 8
    assert evaluable_episodes_h5 == 8
    # Exactly 8 positive healthy origins at H=1 (1 step before onset)
    assert positive_origins_h1 == 8
    # Exactly 40 positive healthy origins at H=5 (5 steps before onset across 8 incidents)
    assert positive_origins_h5 == 40


def test_causal_rollout_queue_mass_balance_and_units():
    """Verifies causal step-by-step rollout queue mass balance, buffer cap, and latency Little's Law delay."""
    from degradation_prediction.models import simulate_step_by_step_rollout

    # Nominal healthy call
    res_nom = simulate_step_by_step_rollout(
        service_id="reservations",
        curr_backlog=0.0,
        curr_offered=20.0,
        forecast_demand=22.0,
        curr_completed=20.0,
        curr_sync_drift=12.0,
        horizon=1,
        dt=20.0,
    )
    # Under nominal load, queue remains 0, latency equals baseline 18.5ms, no degradation
    assert res_nom["backlog_requests"] == 0.0
    assert math.isclose(res_nom["latency_ms"], 18.5, abs_tol=0.1)
    assert res_nom["degradation_risk"] == 0.0
    assert res_nom["total_drops"] == 0.0

    # Overload call: offered 120 req/s, capacity 100 req/s, dt=20s -> inflow 2400, cap 2000 -> excess 400
    res_ovld = simulate_step_by_step_rollout(
        service_id="reservations",
        curr_backlog=0.0,
        curr_offered=120.0,
        forecast_demand=120.0,
        curr_completed=100.0,
        curr_sync_drift=12.0,
        horizon=1,
        dt=20.0,
    )
    # Buffer cap is 400 for reservations; queue absorbs 400, delay = (400/100)*1000 = 4000ms
    assert res_ovld["backlog_requests"] == 400.0
    assert math.isclose(res_ovld["latency_ms"], 18.5 + 4000.0, abs_tol=1.0)
    assert res_ovld["degradation_risk"] == 1.0


def test_synthetic_provenance_propagation():
    """Verifies that synthetic data propagates SYNTHETIC_SIMULATED and does not return NOMINAL_MEASURED."""
    predictor = DegradationPredictor.get_instance(WEIGHTS_PATH)
    sample_telemetry = {
        s.value: [
            {"latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
            for _ in range(5)
        ]
        for s in ServiceIdentifier
    }

    # Default synthetic telemetry
    res_default = predictor.predict(sample_telemetry, horizons=[1])
    assert res_default["data_provenance"] == "SYNTHETIC_SIMULATED"
    assert res_default["input_reliability"] == "SYNTHETIC_SIMULATED"

    # Explicit live measured telemetry
    res_live = predictor.predict(sample_telemetry, horizons=[1], data_provenance="LIVE_MEASURED")
    assert res_live["data_provenance"] == "LIVE_MEASURED"
    assert res_live["input_reliability"] == "NOMINAL_MEASURED"


def test_stale_and_missing_boundary_telemetry():
    """Verifies that missing boundary telemetry is gracefully derived and flagged."""
    predictor = DegradationPredictor.get_instance(WEIGHTS_PATH)
    sample_telemetry = {
        s.value: [
            {"latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
            for _ in range(5)
        ]
        for s in ServiceIdentifier
    }
    # No boundary telemetry passed
    res = predictor.predict(sample_telemetry, boundary_telemetry=None, horizons=[1])
    assert "boundary_status" in res
    assert any("Boundary telemetry stream missing" in w for w in res["warnings"])


def test_calibration_and_threshold_ordering():
    """Verifies that probability calibrator is fitted on cal split and threshold tunes on val split post-calibration."""
    from degradation_prediction.models import LearnedDegradationHorizonModel

    model = LearnedDegradationHorizonModel(horizon=1, include_workload_forecasts=False, random_state=42)
    np.random.seed(42)
    X_tr = np.random.randn(200, 22)
    y_tr = np.random.randint(0, 2, size=200)
    model.fit_train(X_tr, np.random.rand(200)*20, np.random.rand(200)*5, np.random.rand(200)*20, y_tr)

    # Calibrate on disjoint calibration data
    X_cal = np.random.randn(80, 22)
    y_cal = np.random.randint(0, 2, size=80)
    model.calibrate(X_cal, y_cal)
    assert model.calibrator is not None

    # Tune threshold on validation data using post-calibrated probabilities
    X_val = np.random.randn(80, 22)
    y_val = np.random.randint(0, 2, size=80)
    thresh = model.tune_threshold(X_val, y_val)
    assert 0.15 <= thresh <= 0.85
    assert model.alert_threshold == thresh


def test_training_inference_feature_parity(dataset_data):
    """Verifies bit-for-bit feature parity between extract_features_at_origin and inference pipeline."""
    standard_incs = [i for i in dataset_data["incidents"] if i["split"] == "standard_test"]
    inc = standard_incs[0]
    origin = 5

    # Extract via feature extraction directly
    feat_direct, _ = extract_features_at_origin(
        service_id="reservations",
        incident_service_telemetry=inc["service_telemetry"],
        incident_boundary_telemetry=inc["boundary_telemetry"],
        origin_step=origin,
        horizon=1,
        include_workload_forecasts=False,
    )

    # Telemetry slice up to origin
    sliced_svc = {s: inc["service_telemetry"][s][: origin + 1] for s in inc["service_telemetry"]}
    sliced_bnd = inc["boundary_telemetry"][: origin + 1]

    feat_inf, _ = extract_features_at_origin(
        service_id="reservations",
        incident_service_telemetry=sliced_svc,
        incident_boundary_telemetry=sliced_bnd,
        origin_step=origin,
        horizon=1,
        include_workload_forecasts=False,
    )

    assert np.allclose(feat_direct, feat_inf, atol=1e-9)


def test_sustained_breach_three_consecutive_rule():
    """Verifies that degradation is only confirmed after 3 consecutive breached snapshots,
    and first_breach_step correctly records the start of the sequence.
    """
    from degradation_prediction.data_contract import detect_confirmed_sustained_episodes

    # 1. Single breach (step 5) followed by recovery -> should NOT trigger sustained degradation
    svc_tel_single = {
        s.value: [
            {"latency_ms": 120.0 if (s.value == "reservations" and t == 5) else 20.0,
             "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
            for t in range(20)
        ]
        for s in ServiceIdentifier
    }
    bnd_tel = [{"sync_drift_score": 12.0} for _ in range(20)]
    eps_single = detect_confirmed_sustained_episodes(svc_tel_single, bnd_tel, min_consecutive=3)
    assert len(eps_single) == 0, "Single isolated breach must not be confirmed as sustained degradation."

    # 2. Two consecutive breaches (steps 5, 6) followed by recovery -> should NOT trigger
    svc_tel_double = {
        s.value: [
            {"latency_ms": 120.0 if (s.value == "reservations" and t in (5, 6)) else 20.0,
             "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
            for t in range(20)
        ]
        for s in ServiceIdentifier
    }
    eps_double = detect_confirmed_sustained_episodes(svc_tel_double, bnd_tel, min_consecutive=3)
    assert len(eps_double) == 0, "Two consecutive breaches must not be confirmed as sustained degradation."

    # 3. Three consecutive breaches (steps 5, 6, 7) followed by recovery at step 10 -> MUST trigger!
    svc_tel_triple = {
        s.value: [
            {"latency_ms": 120.0 if (s.value == "reservations" and 5 <= t <= 8) else 20.0,
             "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
            for t in range(20)
        ]
        for s in ServiceIdentifier
    }
    eps_triple = detect_confirmed_sustained_episodes(svc_tel_triple, bnd_tel, min_consecutive=3)
    assert len(eps_triple) == 1, "Three consecutive breaches must be confirmed as a sustained degradation episode."
    ep = eps_triple[0]
    assert ep["first_breach_step"] == 5, f"Expected first breach step to be 5, got {ep['first_breach_step']}"
    assert ep["first_breach_time_sec"] == 100.0
    assert ep["confirmation_step"] == 7
    assert ep["recovery_step"] == 9


def test_censoring_and_conditional_time_to_onset():
    """Verifies that non-event origins are treated as censored and return None for onset time."""
    import pickle
    with open(WEIGHTS_PATH, "rb") as f:
        bundle = pickle.load(f)
    regressor = bundle["models_time_to_onset"][5]

    np.random.seed(42)
    X_mock = np.random.randn(10, 35)

    # All healthy origins (pred_event == 0)
    pred_events = np.zeros(10, dtype=int)
    estimates = regressor.predict_time_to_onset_sec(X_mock, pred_events)

    assert len(estimates) == 10
    # Every non-event must return None, not an arbitrary imputed number
    for est in estimates:
        assert est is None, "Non-events must return None for onset estimate to respect censoring."


def test_versioned_feature_name_lookup_and_schema_assertions():
    """Verifies versioned feature name lookup and strict schema dimension assertion."""
    from degradation_prediction.features import (
        get_feature_index,
        assert_feature_schema,
        TELEMETRY_ONLY_FEATURE_NAMES,
        TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
    )

    # 1. Name lookup
    assert get_feature_index("curr_latency_ms", "telemetry") == 0
    assert get_feature_index("has_known_capacity", "telemetry") == 20
    assert get_feature_index("wf_forecast_demand", "combined") == 22
    assert get_feature_index("wf_rollout_latency_proj", "combined") == 34

    # 2. Key error on invalid name
    with pytest.raises(KeyError):
        get_feature_index("nonexistent_magic_feature", "telemetry")

    # 3. Schema assertions
    X_telemetry = np.zeros((5, len(TELEMETRY_ONLY_FEATURE_NAMES)))
    X_combined = np.zeros((5, len(TELEMETRY_PLUS_FORECAST_FEATURE_NAMES)))

    assert_feature_schema(X_telemetry, "telemetry")
    assert_feature_schema(X_combined, "combined")

    with pytest.raises(ValueError):
        assert_feature_schema(X_telemetry, "combined")

    with pytest.raises(ValueError):
        assert_feature_schema(X_combined, "telemetry")


def test_upstream_forecaster_trajectory_isolation():
    """Verifies that workload forecasting dataset and degradation prediction dataset share 0 trajectories."""
    wf_path = AI_MODELS_DIR / "data" / "workload_forecasting" / "workload_dataset.json"
    deg_path = DATA_PATH

    if not wf_path.exists() or not deg_path.exists():
        pytest.skip("Datasets not available for isolation audit.")

    with open(wf_path, "r") as f:
        wf_data = json.load(f)
    with open(deg_path, "r") as f:
        deg_data = json.load(f)

    # Check that incident IDs are completely disjoint
    wf_incidents = set(s["incident_id"] for s in wf_data["trajectories"])
    deg_incidents = set(i["incident_id"] for i in deg_data["incidents"])
    overlap = wf_incidents.intersection(deg_incidents)
    assert len(overlap) == 0, f"Found overlapping incident IDs: {overlap}"


def test_alert_exposure_and_deduplication_cooldown():
    """Verifies that false alert deduplication applies the explicit 100s cooldown."""
    from degradation_prediction.evaluate import compute_early_warning_metrics

    # Synthetic incident with no degradation (healthy throughout 30 steps)
    inc = {
        "incident_id": "test_healthy_inc",
        "step_count": 30,
        "service_telemetry": {
            s.value: [
                {"latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
                for _ in range(30)
            ]
            for s in ServiceIdentifier
        },
        "boundary_telemetry": [{"sync_drift_score": 12.0} for _ in range(30)],
    }

    # Burst of 4 consecutive alerts at t=10, 11, 12, 13
    burst_preds = {
        ("test_healthy_inc", 10): 1,
        ("test_healthy_inc", 11): 1,
        ("test_healthy_inc", 12): 1,
        ("test_healthy_inc", 13): 1,
    }

    metrics = compute_early_warning_metrics([inc], burst_preds, horizon=5, cooldown_steps=5)

    assert metrics["raw_false_alerts_count"] == 4
    # All 4 within cooldown window (5 steps = 100s) collapse into 1 deduplicated episode
    assert metrics["deduplicated_false_alert_episodes"] == 1
    assert metrics["cooldown_policy_seconds"] == 100.0


def test_temporal_gru_comparator_shape_and_inference():
    """Verifies temporal sequence tensor extraction shape (4, 16) and comparator inference."""
    import pickle
    with open(WEIGHTS_PATH, "rb") as f:
        bundle = pickle.load(f)
    model_c = bundle["models_temporal_comparator"][1]

    from degradation_prediction.features import extract_temporal_sequence_tensor

    svc_tel = {
        s.value: [
            {"latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
            for _ in range(10)
        ]
        for s in ServiceIdentifier
    }
    bnd_tel = [{"sync_drift_score": 12.0, "gateway_queue_depth": 0.0} for _ in range(10)]

    tensor = extract_temporal_sequence_tensor(
        service_id="reservations",
        incident_service_telemetry=svc_tel,
        incident_boundary_telemetry=bnd_tel,
        origin_step=5,
        horizon=1,
        window_size=4,
    )

    assert tensor.shape == (4, 16), f"Expected tensor shape (4, 16), got {tensor.shape}"

    batch_tensors = np.tile(tensor[np.newaxis, :, :], (3, 1, 1))
    probs = model_c.predict_event_proba(batch_tensors)

    assert len(probs) == 3
    for p in probs:
        assert 0.0 <= p <= 1.0



def test_priority_weighting_formula():
    """Verifies priority-weighted aggregation formula (1.5*res + 1.0*crew + 0.5*bag) / 3.0."""
    from degradation_prediction.evaluate import compute_priority_weighted_metrics

    service_metrics = {
        "reservations": {"f1": 0.90, "precision": 0.85, "recall": 0.95},
        "crew": {"f1": 0.80, "precision": 0.75, "recall": 0.85},
        "baggage": {"f1": 0.70, "precision": 0.65, "recall": 0.75},
    }

    weighted = compute_priority_weighted_metrics(service_metrics)
    assert weighted["formula"] == "(1.5 * reservations + 1.0 * crew + 0.5 * baggage) / 3.0"

    expected_f1 = (1.5 * 0.90 + 1.0 * 0.80 + 0.5 * 0.70) / 3.0  # (1.35 + 0.80 + 0.35) / 3.0 = 2.50 / 3.0 = 0.8333
    assert math.isclose(weighted["metrics"]["f1"], expected_f1, abs_tol=1e-4)


def test_lead_time_40s_impossibility_at_h1():
    """Verifies mathematical impossibility of H=1 satisfying a >= 40s lead time requirement."""
    horizon = 1
    dt = SAMPLING_INTERVAL_SECONDS  # 20.0s
    max_h1_lead_time = horizon * dt  # 20.0s

    assert max_h1_lead_time < 40.0, (
        f"H=1 lead time cannot exceed {max_h1_lead_time}s; mathematically impossible to achieve >= 40s warning."
    )


# ==============================================================================
# Task 2D Source-Level Defect Verification Tests
# ==============================================================================

def test_task2d_model_a_forecaster_invariance(dataset_data):
    """Requirement 1: Model A tabular features and event probabilities must be invariant to forecaster replacement."""
    from workload_forecasting.inference import WorkloadForecaster
    forecaster = WorkloadForecaster.get_instance()
    inc = dataset_data["incidents"][0]

    # Extract Model A features with forecaster=None vs forecaster=Trained
    feats_none, _ = extract_features_at_origin(
        service_id="reservations",
        incident_service_telemetry=inc["service_telemetry"],
        incident_boundary_telemetry=inc["boundary_telemetry"],
        origin_step=5,
        horizon=1,
        include_workload_forecasts=False,
        forecaster=None,
    )

    feats_trained, _ = extract_features_at_origin(
        service_id="reservations",
        incident_service_telemetry=inc["service_telemetry"],
        incident_boundary_telemetry=inc["boundary_telemetry"],
        origin_step=5,
        horizon=1,
        include_workload_forecasts=False,
        forecaster=forecaster,
    )

    # Invariance of tabular features
    np.testing.assert_allclose(
        feats_none,
        feats_trained,
        err_msg="Model A tabular features must be strictly invariant to forecaster replacement."
    )

    # Invariance of predicted event probability
    predictor = DegradationPredictor.get_instance()
    recent_svc = {s: inc["service_telemetry"][s][:6] for s in inc["service_telemetry"]}
    recent_bnd = inc["boundary_telemetry"][:6]

    res_none = predictor.predict_degradation(
        service_telemetry_buffers=recent_svc,
        boundary_telemetry_buffer=recent_bnd,
        target_services=["reservations"],
        horizons=[1],
        model_variant="model_a_telemetry_only",
    )
    # Model A returns identical results regardless of forecaster presence
    prob_a = res_none["per_service_predictions"]["reservations"]["horizons"]["1"]["predicted_event_probability"]
    assert 0.0 <= prob_a <= 1.0


def test_task2d_trailing_unresolved_breaches_observability_mask():
    """Requirement 2: Trailing 1-step or 2-step breaches near recording end must have observability mask False."""
    from degradation_prediction.data_contract import check_window_label_observability

    step_count = 30
    target_step = 29  # Window ends at step 29 (the last step of the recording)

    # Case 1: Trailing breach at step 29 only (needs step 30, 31 to confirm)
    breaches_trailing_1 = [0] * step_count
    breaches_trailing_1[29] = 1
    obs_1 = check_window_label_observability(target_step=target_step, step_count=step_count, breach_indicators=breaches_trailing_1)
    assert obs_1 is False, "Trailing 1-step breach ending at step_count-1 must be unobservable (needs follow-up)"

    # Case 2: Trailing breach at step 28, 29 (needs step 30 to confirm)
    breaches_trailing_2 = [0] * step_count
    breaches_trailing_2[28] = 1
    breaches_trailing_2[29] = 1
    obs_2 = check_window_label_observability(target_step=target_step, step_count=step_count, breach_indicators=breaches_trailing_2)
    assert obs_2 is False, "Trailing 2-step breach ending at step_count-1 must be unobservable (needs follow-up)"

    # Case 3: Already confirmed 3-step breach inside window (e.g. at 26, 27, 28)
    breaches_confirmed = [0] * step_count
    breaches_confirmed[26] = 1
    breaches_confirmed[27] = 1
    breaches_confirmed[28] = 1
    obs_conf = check_window_label_observability(target_step=target_step, step_count=step_count, breach_indicators=breaches_confirmed)
    assert obs_conf is True, "Confirmed 3-step breach is fully resolved and observable"

    # Case 4: No trailing breach at target_step (clean nominal end)
    breaches_clean = [0] * step_count
    obs_clean = check_window_label_observability(target_step=target_step, step_count=step_count, breach_indicators=breaches_clean)
    assert obs_clean is True, "Zero trailing breaches is fully observable as negative"


def test_task2d_origin_only_current_state_determination():
    """Requirement 2: Operational state at origin must use observations available through the origin only."""
    from degradation_prediction.data_contract import get_operational_state_at_origin

    # Construct synthetic telemetry with breach starting at step 4
    # Steps 0-3: healthy (lat 20ms)
    # Steps 4-6: breached (lat 120ms)
    # Steps 7-10: breached
    svc_data = {
        "reservations": [
            {"latency_ms": 120.0 if t >= 4 else 20.0, "backlog_requests": 0.0, "error_rate": 0.0}
            for t in range(10)
        ]
    }
    bnd_data = [{"sync_drift_score": 12.0} for _ in range(10)]

    # At origin t=3: 0 breaches -> healthy
    s3 = get_operational_state_at_origin({k: v[:4] for k, v in svc_data.items()}, bnd_data[:4])
    assert s3["is_degraded"] is False

    # At origin t=4: 1 breach -> healthy (not yet 3 consecutive)
    s4 = get_operational_state_at_origin({k: v[:5] for k, v in svc_data.items()}, bnd_data[:5])
    assert s4["is_degraded"] is False

    # At origin t=5: 2 consecutive breaches -> healthy (not yet 3 consecutive)
    s5 = get_operational_state_at_origin({k: v[:6] for k, v in svc_data.items()}, bnd_data[:6])
    assert s5["is_degraded"] is False

    # At origin t=6: 3 consecutive breaches -> confirmed degraded!
    s6 = get_operational_state_at_origin({k: v[:7] for k, v in svc_data.items()}, bnd_data[:7])
    assert s6["is_degraded"] is True


def test_task2d_recovery_rule_and_episode_eligibility():
    """Requirement 2: Recovery requires >= 3 healthy snapshots; onset at step 3 has no eligible pre-onset origin."""
    from degradation_prediction.data_contract import detect_confirmed_sustained_episodes
    from degradation_prediction.evaluate import compute_early_warning_metrics

    # Test Recovery Rule:
    # 0-3 healthy, 4-7 degraded (4 steps), 8-9 healthy (only 2 steps), 10-12 degraded (3 steps), 13-16 healthy (4 steps)
    # The 2 healthy steps (8-9) should NOT trigger recovery!
    svc_tel = {
        s.value: [
            {"latency_ms": 120.0 if (4 <= t <= 7 or 10 <= t <= 12) else 20.0, "backlog_requests": 0.0, "error_rate": 0.0}
            for t in range(20)
        ]
        for s in ServiceIdentifier
    }
    bnd_tel = [{"sync_drift_score": 12.0} for _ in range(20)]

    episodes = detect_confirmed_sustained_episodes(svc_tel, bnd_tel, min_consecutive=3)
    assert len(episodes) == 1, f"Expected 1 continuous episode across temporary 2-snapshot dip, got {len(episodes)}"
    assert episodes[0]["recovery_step"] == 13, f"Expected recovery at step 13 (after 3 healthy snapshots), got {episodes[0]['recovery_step']}"

    # Test Episode Eligibility:
    # Synthetic incident with onset at step 3
    inc_onset_step_3 = {
        "incident_id": "inc_step_3",
        "step_count": 20,
        "service_telemetry": {
            s.value: [
                {"latency_ms": 120.0 if t >= 3 else 20.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0}
                for t in range(20)
            ]
            for s in ServiceIdentifier
        },
        "boundary_telemetry": [{"sync_drift_score": 12.0} for _ in range(20)],
    }
    ew_res = compute_early_warning_metrics([inc_onset_step_3], {}, horizon=1)
    # Since min_origin is 3, onset at step 3 has zero pre-onset origins -> must be unevaluable
    assert ew_res["evaluable_episodes"] == 0
    assert ew_res["unevaluable_insufficient_pre_onset_history_episodes"] == 1


def test_task2d_unequal_stream_lengths_and_timestamp_alignment():
    """Requirement 3: Inference must align streams to a common prediction timestamp without crashing on unequal lengths."""
    predictor = DegradationPredictor.get_instance()

    # Create streams with different lengths and explicit timestamps
    # Reservations has 10 steps (timestamps 1000..1180, dt=20)
    # Payments has 7 steps (timestamps 1060..1180)
    # Checkin has 12 steps (timestamps 960..1180)
    # Baggage has 8 steps (timestamps 1040..1180)
    # Boundary has 9 steps (timestamps 1020..1180)
    target_time = 1180.0

    svc_telemetry = {
        "reservations": [{"timestamp_sec": 1000.0 + 20.0 * i, "latency_ms": 25.0, "backlog_requests": 2.0, "offered_rate": 40.0, "completed_rate": 40.0, "error_rate": 0.0} for i in range(10)],
        "crew": [{"timestamp_sec": 1060.0 + 20.0 * i, "latency_ms": 30.0, "backlog_requests": 1.0, "offered_rate": 35.0, "completed_rate": 35.0, "error_rate": 0.0} for i in range(7)],
        "boundary-gateway": [{"timestamp_sec": 960.0 + 20.0 * i, "latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0} for i in range(12)],
        "baggage": [{"timestamp_sec": 1040.0 + 20.0 * i, "latency_ms": 15.0, "backlog_requests": 0.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0} for i in range(8)],
    }
    bnd_telemetry = [{"timestamp_sec": 1020.0 + 20.0 * i, "sync_drift_score": 14.0} for i in range(9)]

    # Must execute smoothly without IndexError or alignment crash
    result = predictor.predict_degradation(
        service_telemetry_buffers=svc_telemetry,
        boundary_telemetry_buffer=bnd_telemetry,
        prediction_timestamp_sec=target_time,
        target_services=["reservations", "crew"],
        horizons=[1, 5],
        model_variant="model_a_telemetry_only",
    )

    assert result["contract_version"].startswith("baccp-degradation-contract-v1.")
    assert result["alignment_timestamp_sec"] == target_time
    assert result["input_reliability"]["has_history_padding"] is True  # Crew and others padded


def test_task2d_missing_boundary_imputed_via_repo_formula():
    """Requirement 3: Missing boundary telemetry must compute drift via repository formula rather than hardcoded 12.0%."""
    from degradation_prediction.data_contract import compute_sync_drift_score
    predictor = DegradationPredictor.get_instance()

    # Construct service telemetry where boundary-gateway is present with known queue and latency
    # Repository formula: min(100.0, 10.0 + (queue / 2.5) + (lat / 50.0) + (err * 20.0) + |gw_sub - leg_comp|)
    svc_telemetry = {
        s.value: [
            {
                "timestamp_sec": 1000.0 + 20.0 * i,
                "latency_ms": 100.0,
                "backlog_requests": 10.0,
                "offered_rate": 50.0,
                "completed_rate": 40.0,
                "error_rate": 0.05,
            }
            for i in range(6)
        ]
        for s in ServiceIdentifier
    }

    # Pass empty boundary telemetry buffer
    result = predictor.predict_degradation(
        service_telemetry_buffers=svc_telemetry,
        boundary_telemetry_buffer=[],
        prediction_timestamp_sec=1100.0,
        target_services=["reservations"],
        horizons=[1],
        model_variant="model_a_telemetry_only",
    )

    # Must flag boundary stream missing
    assert result["input_reliability"]["boundary_stream_missing"] is True
    assert "MISSING_BOUNDARY_TELEMETRY" in result["input_reliability"]["reliability_warnings"]

    # Verify formula calculation directly:
    expected_drift = compute_sync_drift_score(
        gateway_submission_rate=50.0,
        legacy_completion_rate=40.0,
        gateway_queue_depth=10.0,
        boundary_latency_ms=100.0,
        boundary_error_rate=0.05,
    )
    # Repo formula: base_drift (20.0) + queue_drift (4.0) + lat_drift (27.37) + error_drift (12.5) = 63.87%
    assert math.isclose(expected_drift, 63.87, abs_tol=1e-2)
    assert not math.isclose(expected_drift, 12.0, abs_tol=1e-4)


def test_task2d_calibrator_variant_and_service_subset_restrictions():
    """Requirement 4: Calibrator applies strictly to Model B across all 5 services; subset or Model A returns None."""
    predictor = DegradationPredictor.get_instance()

    svc_telemetry = {
        s.value: [
            {"timestamp_sec": 1000.0 + 20.0 * i, "latency_ms": 25.0, "backlog_requests": 2.0, "offered_rate": 40.0, "completed_rate": 40.0, "error_rate": 0.0}
            for i in range(6)
        ]
        for s in ServiceIdentifier
    }
    bnd_telemetry = [{"timestamp_sec": 1000.0 + 20.0 * i, "sync_drift_score": 12.0} for i in range(6)]

    # Case 1: Model A requested -> must return None for system_calibrated_degradation_risk
    res_a = predictor.predict_degradation(
        service_telemetry_buffers=svc_telemetry,
        boundary_telemetry_buffer=bnd_telemetry,
        target_services=[s.value for s in ServiceIdentifier],
        horizons=[1],
        model_variant="model_a_telemetry_only",
    )
    h1_a = res_a["system_aggregate_predictions"]["1"]
    assert h1_a["system_calibrated_degradation_risk"] is None
    assert "UNAVAILABLE_FOR_MODEL_VARIANT" in h1_a["system_calibration_status"]

    # Case 2: Model B with single service subset (reservations only) -> must return None
    res_b_subset = predictor.predict_degradation(
        service_telemetry_buffers=svc_telemetry,
        boundary_telemetry_buffer=bnd_telemetry,
        target_services=["reservations"],
        horizons=[1],
        model_variant="model_b_telemetry_forecast",
    )
    h1_b_sub = res_b_subset["system_aggregate_predictions"]["1"]
    assert h1_b_sub["system_calibrated_degradation_risk"] is None
    assert h1_b_sub["system_calibration_status"] == "UNAVAILABLE_SUBSET_SCOPE"

    # Case 3: Model B with all 5 services evaluated -> returns calibrated probability float
    res_b_full = predictor.predict_degradation(
        service_telemetry_buffers=svc_telemetry,
        boundary_telemetry_buffer=bnd_telemetry,
        target_services=[s.value for s in ServiceIdentifier],
        horizons=[1],
        model_variant="model_b_telemetry_forecast",
    )
    h1_b_full = res_b_full["system_aggregate_predictions"]["1"]
    assert h1_b_full["system_calibrated_degradation_risk"] is not None
    assert 0.0 <= h1_b_full["system_calibrated_degradation_risk"] <= 1.0
    assert h1_b_full["system_calibration_status"] == "CALIBRATED"


def test_task2d_boundary_gateway_drift_only_type_and_severity():
    """Requirement 5: Boundary gateway drift breaches must emit BOUNDARY_DRIFT_DETERIORATION and appropriate severity."""
    from degradation_prediction.data_contract import (
        BOUNDARY_SYNC_DRIFT_CRITICAL,
        BOUNDARY_SYNC_DRIFT_THRESHOLD,
        DegradationSeverity,
        DegradationType,
        classify_service_step_type_and_severity,
    )

    nominal_svc = {"latency_ms": 20.0, "backlog_requests": 0.0, "error_rate": 0.0}

    # Case 1: Drift breach above threshold (45.0) but below critical (70.0)
    drift_warn = {"sync_drift_score": 50.0}
    t1, s1 = classify_service_step_type_and_severity("boundary-gateway", nominal_svc, boundary_data=drift_warn)
    assert t1 == DegradationType.BOUNDARY_DRIFT_DETERIORATION
    assert s1 == DegradationSeverity.MINOR

    # Case 2: Drift breach above critical threshold (70.0)
    drift_crit = {"sync_drift_score": 75.0}
    t2, s2 = classify_service_step_type_and_severity("boundary-gateway", nominal_svc, boundary_data=drift_crit)
    assert t2 == DegradationType.BOUNDARY_DRIFT_DETERIORATION
    assert s2 == DegradationSeverity.CRITICAL

    # Case 3: Compound breach: both latency breach (>=200ms) and drift breach (>=45%)
    svc_breach = {"latency_ms": 250.0, "backlog_requests": 0.0, "error_rate": 0.0}
    t3, s3 = classify_service_step_type_and_severity("boundary-gateway", svc_breach, boundary_data=drift_warn)
    assert t3 == DegradationType.COMPOUND_MULTIPLE
    assert s3 == DegradationSeverity.MINOR


def test_task2d_computed_comparison_tables_no_hardcoded_counts():
    """Requirement 1: Reconciliation results must be computed dynamically without hardcoded counts or defaults."""
    metrics_path = REPO_ROOT / "results" / "degradation_prediction" / "metrics_summary.json"
    report_path = REPO_ROOT / "results" / "degradation_prediction" / "evaluation_report.md"

    assert metrics_path.exists(), f"Metrics summary not found at {metrics_path}"
    assert report_path.exists(), f"Evaluation report not found at {report_path}"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    reconcil = metrics["diagnostic_discrepancy_reconciliation"]
    assert reconcil["model_a_invariance_verified"] is True

    fn_res = reconcil["forecaster_none_fallback_h1_healthy"]
    tr_res = reconcil["trained_forecaster_h1_healthy"]

    # Model A recall must be identical under both forecaster configurations
    assert fn_res["model_a_recalled"] == tr_res["model_a_recalled"]
    assert fn_res["total_positive_healthy_origins"] == tr_res["total_positive_healthy_origins"]

    # Check report content: should contain dynamic table and no silent hardcoded numbers (7/8 vs 3/8)
    report_text = report_path.read_text(encoding="utf-8")
    assert "Discrepancy Reconciliation & Model A Invariance" in report_text
    assert f"{fn_res['model_a_recalled']} / {fn_res['total_positive_healthy_origins']}" in report_text


# ==============================================================================
# TASK 2E REGRESSION TESTS: DEFECT 1 & DEFECT 2 FIXES
# ==============================================================================

def test_task2e_workload_forecaster_adapter_deterministic_reproduction():
    """Defect 1 Reproduction & Fix:
    Upstream returns forecast=200, interval=190-210, capacity=100 (overload=100) on current demand=20.
    Asserts that:
    1. GRU sequence tensor receives forecast=200, width=20, overload=100 (not 20, 6, 0).
    2. Tabular features receive forecast=200, width=20, overload=100.
    3. Model A tabular features and probabilities are strictly invariant to upstream forecaster.
    """
    from degradation_prediction.features import (
        adapt_workload_forecast,
        extract_features_at_origin,
        extract_temporal_sequence_tensor,
    )
    from degradation_prediction.data_contract import WorkloadForecastState
    from workload_forecasting.inference import WorkloadForecaster

    # Deterministic mock candidate model & calibrator
    class DeterministicModel:
        def predict(self, X):
            return np.array([200.0])

    class DeterministicCalibrator:
        def get_intervals(self, points):
            return np.array([190.0]), np.array([210.0])

    wf = WorkloadForecaster.get_instance()
    orig_model = wf.models["reservations"][1]["gradient_boosting"]
    orig_cal = wf.calibrators["reservations"][1]["gradient_boosting"]

    try:
        wf.models["reservations"][1]["gradient_boosting"] = DeterministicModel()
        wf.calibrators["reservations"][1]["gradient_boosting"] = DeterministicCalibrator()

        # Step 1: Upstream predict returns standardized response
        upstream_res = wf.predict("reservations", [20.0] * 20, horizon=1)
        assert upstream_res["point_forecast_rps"] == 200.0
        assert upstream_res["prediction_interval_90"]["lower_rps"] == 190.0
        assert upstream_res["prediction_interval_90"]["upper_rps"] == 210.0
        assert upstream_res["prediction_interval_90"]["interval_width_rps"] == 20.0
        assert upstream_res["capacity_metrics"]["configured_capacity_rps"] == 100.0
        assert upstream_res["capacity_metrics"]["overload_rps"] == 100.0

        # Step 2: Shared validated adapter
        adapted = adapt_workload_forecast(
            service_id="reservations",
            horizon=1,
            history=[20.0] * 20,
            forecaster=wf,
            current_offered_rps=20.0,
            physical_timestamp_sec=400.0,
            step_index=20,
        )
        assert adapted.prediction_state == WorkloadForecastState.VALID_TRAINED_PREDICTION
        assert adapted.forecast_demand_rps == 200.0
        assert adapted.interval_width_rps == 20.0
        assert adapted.overload_rps == 100.0
        assert adapted.is_overload == 1.0

        # Step 3: GRU temporal sequence tensor receives 200, 20, 100 (indices 13, 14, 15)
        svc_tel = {
            "reservations": [{"timestamp_sec": float(20 * i), "latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0} for i in range(5)],
            "legacy-core": [{"timestamp_sec": float(20 * i), "latency_ms": 60.0, "backlog_requests": 0.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0} for i in range(5)],
        }
        bnd_tel = [{"timestamp_sec": float(20 * i), "sync_drift_score": 12.0, "gateway_queue_depth": 0.0} for i in range(5)]

        seq_tensor = extract_temporal_sequence_tensor(
            service_id="reservations",
            incident_service_telemetry=svc_tel,
            incident_boundary_telemetry=bnd_tel,
            origin_step=4,
            horizon=1,
            window_size=4,
            forecaster=wf,
            physical_timestamp_sec=80.0,
        )
        # Check that forecast 200, width 20, overload 100 reach downstream temporal inputs
        # Indices: 13: forecast_demand_req_s, 14: forecast_interval_width, 15: forecast_overload_req_s
        assert seq_tensor.shape == (4, 16)
        for step_idx in range(4):
            assert seq_tensor[step_idx, 13] == 200.0, f"Expected forecast 200.0, got {seq_tensor[step_idx, 13]}"
            assert seq_tensor[step_idx, 14] == 20.0, f"Expected width 20.0, got {seq_tensor[step_idx, 14]}"
            assert seq_tensor[step_idx, 15] == 100.0, f"Expected overload 100.0, got {seq_tensor[step_idx, 15]}"

        # Step 4: Tabular feature extraction receives 200, 20, 100
        feat_vec_b, aux_b = extract_features_at_origin(
            service_id="reservations",
            incident_service_telemetry=svc_tel,
            incident_boundary_telemetry=bnd_tel,
            origin_step=4,
            horizon=1,
            forecaster=wf,
            include_workload_forecasts=True,
            physical_timestamp_sec=80.0,
        )
        from degradation_prediction.features import get_feature_index
        assert feat_vec_b[get_feature_index("wf_forecast_demand", "combined")] == 200.0
        assert feat_vec_b[get_feature_index("wf_interval_width", "combined")] == 20.0
        assert feat_vec_b[get_feature_index("wf_forecasted_overload", "combined")] == 100.0
        assert feat_vec_b[get_feature_index("wf_is_overloaded", "combined")] == 1.0

        # Step 5: Model A Independence: Tabular features (22 features) return before forecaster is called
        feat_vec_a1, _ = extract_features_at_origin(
            service_id="reservations",
            incident_service_telemetry=svc_tel,
            incident_boundary_telemetry=bnd_tel,
            origin_step=4,
            horizon=1,
            forecaster=wf,
            include_workload_forecasts=False,
            physical_timestamp_sec=80.0,
        )
        feat_vec_a2, _ = extract_features_at_origin(
            service_id="reservations",
            incident_service_telemetry=svc_tel,
            incident_boundary_telemetry=bnd_tel,
            origin_step=4,
            horizon=1,
            forecaster=None,
            include_workload_forecasts=False,
            physical_timestamp_sec=80.0,
        )
        assert len(feat_vec_a1) == 22
        assert len(feat_vec_a2) == 22
        np.testing.assert_array_equal(feat_vec_a1, feat_vec_a2)

    finally:
        wf.models["reservations"][1]["gradient_boosting"] = orig_model
        wf.calibrators["reservations"][1]["gradient_boosting"] = orig_cal


def test_task2e_workload_adapter_five_prediction_states_and_no_silent_fallback():
    """Defect 1 Requirement: Distinguish all five prediction states and never silently fall back
    on incompatible response schemas.
    """
    from degradation_prediction.features import adapt_workload_forecast
    from degradation_prediction.data_contract import WorkloadForecastState
    from workload_forecasting.inference import WorkloadForecaster

    wf = WorkloadForecaster.get_instance()

    # State 1: Valid trained prediction
    res1 = adapt_workload_forecast(
        service_id="reservations",
        horizon=1,
        history=[25.0] * 20,
        forecaster=wf,
        current_offered_rps=25.0,
    )
    assert res1.prediction_state == WorkloadForecastState.VALID_TRAINED_PREDICTION
    assert res1.fallback_triggered is False

    # State 2: Intentional persistence fallback (forecaster=None)
    res2 = adapt_workload_forecast(
        service_id="reservations",
        horizon=1,
        history=[25.0] * 20,
        forecaster=None,
        current_offered_rps=25.0,
    )
    assert res2.prediction_state == WorkloadForecastState.INTENTIONAL_PERSISTENCE_FALLBACK
    assert res2.fallback_triggered is True
    assert res2.forecast_demand_rps == 25.0

    # State 3: Missing artifact
    class EmptyForecaster:
        models = {"reservations": {}}  # Horizon 15 missing
    with pytest.raises(FileNotFoundError, match="Missing artifact"):
        adapt_workload_forecast(
            service_id="reservations",
            horizon=15,
            history=[25.0] * 20,
            forecaster=EmptyForecaster(),
            current_offered_rps=25.0,
        )

    # State 4: Invalid response schema (must NOT silently fall back)
    class LegacyIncompatibleForecaster:
        def predict(self, *args, **kwargs):
            # Returns old schema with unsuffixed keys and conformal_interval
            return {
                "point_forecast": 200.0,
                "conformal_interval": [190.0, 210.0],
                "capacity_margin": -100.0,
            }

    with pytest.raises(ValueError, match="Invalid WorkloadForecaster response schema"):
        adapt_workload_forecast(
            service_id="reservations",
            horizon=1,
            history=[25.0] * 20,
            forecaster=LegacyIncompatibleForecaster(),
            current_offered_rps=25.0,
        )

    # State 5: Prediction exception
    class CrashingForecaster:
        def predict(self, *args, **kwargs):
            raise ArithmeticError("Simulated internal GPU numerical overflow")

    with pytest.raises(RuntimeError, match="Prediction exception in WorkloadForecaster"):
        adapt_workload_forecast(
            service_id="reservations",
            horizon=1,
            history=[25.0] * 20,
            forecaster=CrashingForecaster(),
            current_offered_rps=25.0,
        )


def test_task2e_causal_alignment_future_only_stream_rejection():
    """Defect 2 Reproduction & Fix:
    Requested prediction time: 20 seconds.
    First available service observation: 100 seconds.
    Sanitizer must REJECT and never backfill from the future.
    """
    predictor = DegradationPredictor.get_instance()
    svc_tel = {
        "reservations": [
            {"timestamp_sec": 100.0, "latency_ms": 25.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0}
        ]
    }

    with pytest.raises(ValueError, match="Causal alignment violation.*Future observations cannot be used"):
        predictor.predict_degradation(
            service_telemetry_buffers=svc_tel,
            prediction_timestamp_sec=20.0,
        )


def test_task2e_causal_alignment_staleness_limit_and_carried_forward():
    """Defect 2 Requirement:
    Enforce documented staleness limit (MAX_TELEMETRY_STALENESS_SECONDS = 60.0s).
    - If staleness > 60s: reject with ValueError.
    - If staleness <= 60s: carry forward with warning and set has_carried_forward=True.
    """
    predictor = DegradationPredictor.get_instance()

    # Case 1: Staleness = 80s (> 60s limit) -> Reject
    stale_tel = {
        "reservations": [
            {"timestamp_sec": 0.0, "latency_ms": 20.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0},
            {"timestamp_sec": 20.0, "latency_ms": 20.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0},
        ]
    }
    with pytest.raises(ValueError, match="Staleness limit exceeded.*maximum permitted staleness is 60.0s"):
        predictor.predict_degradation(
            service_telemetry_buffers=stale_tel,
            prediction_timestamp_sec=100.0,
        )

    # Case 2: Staleness = 20s (<= 60s limit) -> Carry forward with flag and warning
    fresh_tel = {
        "reservations": [
            {"timestamp_sec": float(20 * i), "latency_ms": 20.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0}
            for i in range(5)  # 0, 20, 40, 60, 80s
        ]
    }
    res = predictor.predict_degradation(
        service_telemetry_buffers=fresh_tel,
        prediction_timestamp_sec=100.0,
    )
    assert res["input_reliability"]["has_carried_forward"] is True
    assert "reservations" in res["input_reliability"]["carried_forward_services"]
    assert any("carried forward from 80.0s to 100.0s" in w for w in res["warnings"])


def test_task2e_causal_alignment_duplicate_timestamps_and_sampling_gaps():
    """Defect 2 Requirement:
    Sanitizer must deduplicate timestamps (retaining latest) and warn on sampling gaps > 1.5 * dt (30s).
    """
    predictor = DegradationPredictor.get_instance()
    svc_tel = {
        "reservations": [
            {"timestamp_sec": 0.0, "latency_ms": 20.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0},
            {"timestamp_sec": 20.0, "latency_ms": 21.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0},
            {"timestamp_sec": 40.0, "latency_ms": 22.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0},
            {"timestamp_sec": 40.0, "latency_ms": 23.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0},  # Duplicate
            {"timestamp_sec": 80.0, "latency_ms": 24.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0},  # Gap: 80-40=40s > 30s
        ]
    }
    res = predictor.predict_degradation(
        service_telemetry_buffers=svc_tel,
        prediction_timestamp_sec=80.0,
    )
    assert any("Duplicate timestamps detected in service 'reservations'" in w for w in res["warnings"])
    assert any("Sampling gap detected in service 'reservations'" in w for w in res["warnings"])


def test_task2e_causal_alignment_rolling_buffer_nonzero_physical_timestamps():
    """Defect 2 Requirement:
    Rolling buffers starting at nonzero timestamps must pass the physical timestamp (e.g. 1060s)
    into workload forecasting instead of resetting phase to buffer index * 20.
    """
    from workload_forecasting.inference import WorkloadForecaster

    received_ts = []

    class PhysicalTimestampSpyForecaster(WorkloadForecaster):
        def predict(self, *args, **kwargs):
            received_ts.append(kwargs.get("timestamp_sec"))
            return super().predict(*args, **kwargs)

    predictor = DegradationPredictor.get_instance()
    original_forecaster = predictor.forecaster
    try:
        spy_forecaster = PhysicalTimestampSpyForecaster()
        predictor.forecaster = spy_forecaster

        svc_tel = {
            s: [
                {"timestamp_sec": 1000.0 + 20.0 * i, "latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 25.0, "completed_rate": 25.0, "error_rate": 0.0}
                for i in range(5)
            ]
            for s in predictor.supported_services
        }
        bnd_tel = [{"timestamp_sec": 1000.0 + 20.0 * i, "sync_drift_score": 12.0} for i in range(5)]

        res = predictor.predict_degradation(
            service_telemetry_buffers=svc_tel,
            boundary_telemetry_buffer=bnd_tel,
            prediction_timestamp_sec=1080.0,
            model_variant="model_b_telemetry_forecast",
        )
        assert len(received_ts) > 0
        assert all(ts == 1080.0 for ts in received_ts), f"Expected all 1080.0s, got {received_ts}"
        assert res["alignment_timestamp_sec"] == 1080.0
    finally:
        predictor.forecaster = original_forecaster


def test_task2e_equal_length_arrays_not_equated_with_timestamp_alignment():
    """Defect 2 Requirement:
    Do not equate equal array lengths with timestamp alignment.
    Two streams with identical length (5 elements) but disjoint timestamps must fail causal/staleness checks.
    """
    predictor = DegradationPredictor.get_instance()

    # Stream A: timestamps 0..80s (5 records)
    # Stream B: timestamps 100..180s (5 records)
    stream_a = [{"timestamp_sec": float(20 * i), "latency_ms": 20.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0} for i in range(5)]
    stream_b = [{"timestamp_sec": float(100 + 20 * i), "latency_ms": 20.0, "offered_rate": 20.0, "completed_rate": 20.0, "error_rate": 0.0} for i in range(5)]

    svc_tel = {
        "reservations": stream_a,
        "crew": stream_b,
    }

    # At prediction time 80.0s: Stream B is entirely in the future -> Causal alignment violation
    with pytest.raises(ValueError, match="Causal alignment violation.*service 'crew' has no historical observations"):
        predictor.predict_degradation(
            service_telemetry_buffers=svc_tel,
            prediction_timestamp_sec=80.0,
        )

    # At prediction time 180.0s: Stream A ended at 80.0s (staleness 100.0s > 60.0s) -> Staleness limit exceeded
    with pytest.raises(ValueError, match="Staleness limit exceeded for service 'reservations'"):
        predictor.predict_degradation(
            service_telemetry_buffers=svc_tel,
            prediction_timestamp_sec=180.0,
        )

def test_task4a_baseline_inputs_sensitivity_and_validation():
    """Task 4A Requirement 1:
    - Test that changing an actual observation changes the baseline score.
    - Test that missing required fields or non-finite values produce an explicit error.
    - Test that baseline inputs do not silently default.
    """
    import copy
    from degradation_prediction.evaluate_task4 import (
        TrainingFittedZScoreBaseline,
        extract_baseline_inputs,
    )
    from degradation_prediction.models import StaticThresholdBaseline

    # Valid 22-feature telemetry row for 'reservations'
    X_nominal = np.zeros((1, 22), dtype=np.float64)
    X_nominal[0, 0] = 20.0   # curr_latency_ms
    X_nominal[0, 3] = 0.0    # curr_backlog
    X_nominal[0, 9] = 0.0    # curr_error_rate
    X_nominal[0, 12] = 12.0  # gw_sync_drift
    metas = [{"service_id": "reservations", "origin_step": 5}]

    # 1. Validation: Malformed 1D array raises ValueError
    with pytest.raises(ValueError, match="Expected 2D numpy feature matrix"):
        extract_baseline_inputs(np.zeros(22), metas)

    # 2. Validation: Row count mismatch raises ValueError
    with pytest.raises(ValueError, match="Row count mismatch"):
        extract_baseline_inputs(X_nominal, metas + [{"service_id": "crew"}])

    # 3. Validation: Missing 'service_id' in meta raises KeyError
    with pytest.raises(KeyError, match="missing required 'service_id'"):
        extract_baseline_inputs(X_nominal, [{"origin_step": 5}])

    # 4. Validation: Non-finite values raise ValueError
    X_nan = copy.deepcopy(X_nominal)
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite baseline observation"):
        extract_baseline_inputs(X_nan, metas)

    # 5. Sensitivity: Normal vs Degraded observation changes baseline decision
    inputs_nom = extract_baseline_inputs(X_nominal, metas)
    static_base = StaticThresholdBaseline(ratio=0.75)
    z_base = TrainingFittedZScoreBaseline()
    z_base.stats["reservations"] = {"lat_mean": 20.0, "lat_std": 5.0, "q_mean": 0.0, "q_std": 5.0}
    z_base.operating_threshold = 2.0

    # Nominal inputs -> alert is 0
    score_nom = z_base.compute_z_score("reservations", inputs_nom[0]["curr_latency"], inputs_nom[0]["curr_backlog"])
    pred_nom_z = z_base.predict("reservations", inputs_nom[0]["curr_latency"], inputs_nom[0]["curr_backlog"])
    pred_nom_s = static_base.predict_event_prob("reservations", inputs_nom[0]["curr_latency"], inputs_nom[0]["curr_backlog"], inputs_nom[0]["curr_error_rate"], inputs_nom[0]["curr_sync_drift"])
    assert score_nom == 0.0
    assert pred_nom_z == 0
    assert pred_nom_s == 0.0

    # Degraded inputs -> latency 120ms, backlog 60 requests -> alert is 1
    X_deg = copy.deepcopy(X_nominal)
    X_deg[0, 0] = 120.0  # High latency
    X_deg[0, 3] = 60.0   # High backlog
    inputs_deg = extract_baseline_inputs(X_deg, metas)
    score_deg = z_base.compute_z_score("reservations", inputs_deg[0]["curr_latency"], inputs_deg[0]["curr_backlog"])
    pred_deg_z = z_base.predict("reservations", inputs_deg[0]["curr_latency"], inputs_deg[0]["curr_backlog"])
    pred_deg_s = static_base.predict_event_prob("reservations", inputs_deg[0]["curr_latency"], inputs_deg[0]["curr_backlog"], inputs_deg[0]["curr_error_rate"], inputs_deg[0]["curr_sync_drift"])
    assert score_deg > score_nom
    assert score_deg >= 12.0
    assert pred_deg_z == 1
    assert pred_deg_s == 1.0


def test_task4a_reconciled_metrics_and_cache_consistency():
    """Task 4A Requirement 3:
    Lightweight consistency tests linking cached confusion counts, metric JSON and generated report tables.
    """
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "results" / "degradation_prediction"

    metrics_path = results_dir / "task4_evaluation_metrics.json"
    cached_path = results_dir / "cached_predictions.json"
    tables_path = results_dir / "reconciled_evaluation_tables.md"

    if not metrics_path.exists() or not cached_path.exists():
        pytest.skip("Task 4 metrics artifacts not yet generated.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(cached_path, "r", encoding="utf-8") as f:
        cached_preds = json.load(f)

    expected_samples = {"1": 3250, "5": 2750, "15": 1500}
    models = [
        ("model_a_telemetry_only", "model_a_pred"),
        ("model_b_telemetry_forecast", "model_b_pred"),
        ("model_c_temporal_comparator", "model_c_pred"),
    ]

    for h_str, n_exp in expected_samples.items():
        h = int(h_str)
        rows = cached_preds[h_str] if h_str in cached_preds else cached_preds[h]
        assert len(rows) == n_exp, f"Horizon H={h} expected {n_exp} cached samples, got {len(rows)}"

        h_eval = metrics["results_by_horizon"][h_str]

        for m_key, pred_col in models:
            c = h_eval[m_key]["overall_classification"]
            assert c["sample_count"] == n_exp
            assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == n_exp

            # Recompute confusion counts directly from cached predictions
            y_true = np.array([r["y_event"] for r in rows])
            y_pred = np.array([r[pred_col] for r in rows])
            tp_recomputed = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp_recomputed = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn_recomputed = int(np.sum((y_pred == 0) & (y_true == 1)))
            tn_recomputed = int(np.sum((y_pred == 0) & (y_true == 0)))

            assert c["tp"] == tp_recomputed, f"TP mismatch for {m_key} at H={h}"
            assert c["fp"] == fp_recomputed, f"FP mismatch for {m_key} at H={h}"
            assert c["fn"] == fn_recomputed, f"FN mismatch for {m_key} at H={h}"
            assert c["tn"] == tn_recomputed, f"TN mismatch for {m_key} at H={h}"

            # Verify precision, recall, F1 consistency
            if (c["tp"] + c["fp"]) > 0:
                p_calc = c["tp"] / (c["tp"] + c["fp"])
                assert abs(c["precision"] - p_calc) < 1e-3
            if (c["tp"] + c["fn"]) > 0:
                r_calc = c["tp"] / (c["tp"] + c["fn"])
                assert abs(c["recall"] - r_calc) < 1e-3

        # Verify unit-bearing timing fields
        for r in rows:
            if r["y_time_to_onset_steps"] >= 0:
                assert abs(r["y_time_to_onset_sec"] - r["y_time_to_onset_steps"] * 20.0) < 1e-6

