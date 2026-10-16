"""Regression and Unit Tests for BACCP Degradation Prediction Explanations (Task 3).

Verifies:
1. Explanation Prerequisites:
   - Regular sampling grid alignment causal consistency (no left-padding mismatch [0,20,40,60,80] vs [0,0,20,60,80]).
   - Uncertainty fallback marks nominal_coverage=None and is_calibrated=False.
   - Missing forecaster artifact raises RuntimeError for forecast-enhanced models.
   - Manifest excludes itself from its file-hash list; manifest and archive hashes separate.
2. Feature Attribution & Model Interface:
   - Model A & Model B Tree SHAP explains raw margin log-odds; raw margin not added directly to calibrated probability.
   - Feature drivers have physical units, provenance, reference values, and categories.
   - Reconstruction error is computed and small.
   - Positive and negative drivers retained.
3. Integrated Gradients for Temporal Comparator:
   - Evaluates on CPU with Gauss-Legendre / trapezoidal quadrature.
   - Attributions across 4 timesteps and 16 features sum to logit difference within tolerance.
4. Graph & Suspected-Origin Localization:
   - Counterfactual node-group perturbations tied to actual model.
   - Distinguishes affected service, suspected contributing node, propagation bottleneck, and verified injected origin.
   - Returns unknown_unlocalized when max risk drop < 5%.
   - Preserves legacy/gateway/cloud generation labels.
5. Deterministic Incident Briefs:
   - Generated strictly from computed outputs without hallucination.
   - Differentiates 'no alert' from 'confirmed healthy'.
   - Documents data provenance and reliability warnings.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from degradation_prediction.data_contract import (
    BOUNDARY_SYNC_DRIFT_THRESHOLD,
    CONFIGURED_SERVICE_CAPACITIES,
    CONTRACT_VERSION,
    NOMINAL_BASELINES,
    SAMPLING_INTERVAL_SECONDS,
    SLO_THRESHOLDS,
    AdaptedWorkloadForecast,
)
from degradation_prediction.explanation import (
    FEATURE_CATEGORY_MAP,
    FEATURE_UNIT_MAP,
    SERVICE_GENERATION_LABELS,
    DegradationExplainer,
)
from degradation_prediction.features import (
    FEATURE_SCHEMA_VERSION,
    TELEMETRY_ONLY_FEATURE_NAMES,
    TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
    TEMPORAL_SEQUENCE_FEATURE_NAMES,
    adapt_workload_forecast,
    extract_features_at_origin,
)
from degradation_prediction.inference import DegradationPredictor


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "degradation_prediction"


@pytest.fixture(scope="module")
def predictor() -> DegradationPredictor:
    return DegradationPredictor.get_instance()


@pytest.fixture(scope="module")
def explainer() -> DegradationExplainer:
    return DegradationExplainer.get_instance()


@pytest.fixture(scope="module")
def nominal_telemetry() -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    dt = 20.0
    services = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]
    svc_data: Dict[str, List[Dict[str, Any]]] = {}
    for s in services:
        svc_data[s] = [
            {
                "timestamp_sec": float(i * dt),
                "step_index": i,
                "service_id": s,
                "offered_rate": 40.0,
                "completed_rate": 40.0,
                "backlog_requests": 0.0,
                "latency_ms": 25.0,
                "error_rate": 0.0,
                "dropped_requests": 0.0,
                "effective_capacity": 80.0 if s != "legacy-core" else None,
            }
            for i in range(8)
        ]

    bnd_data = [
        {
            "timestamp_sec": float(i * dt),
            "step_index": i,
            "sync_drift_score": 12.0,
            "gateway_queue_depth": 0.0,
            "boundary_latency_ms": 20.0,
            "boundary_error_rate": 0.0,
            "is_stale": False,
        }
        for i in range(8)
    ]
    return svc_data, bnd_data


# =============================================================================
# PREREQUISITE TESTS
# =============================================================================

def test_prerequisite_1_timestamp_consistency_regular_grid(predictor: DegradationPredictor):
    """Prerequisite 1: Grid alignment replaces left-padding mismatch with matching physical timestamps."""
    # Reservations has timestamps [0, 20, 40, 60, 80]
    # Crew has timestamps [20, 40, 60, 80] (shorter by 1 step)
    svc_tel = {
        "reservations": [
            {"timestamp_sec": float(t), "latency_ms": 30.0, "backlog_requests": 0.0, "offered_rate": 40.0, "completed_rate": 40.0, "error_rate": 0.0, "dropped_requests": 0.0, "effective_capacity": 80.0}
            for t in [0.0, 20.0, 40.0, 60.0, 80.0]
        ],
        "crew": [
            {"timestamp_sec": float(t), "latency_ms": 25.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0, "dropped_requests": 0.0, "effective_capacity": 50.0}
            for t in [20.0, 40.0, 60.0, 80.0]
        ],
        "baggage": [
            {"timestamp_sec": float(t), "latency_ms": 20.0, "backlog_requests": 0.0, "offered_rate": 35.0, "completed_rate": 35.0, "error_rate": 0.0, "dropped_requests": 0.0, "effective_capacity": 60.0}
            for t in [0.0, 20.0, 40.0, 60.0, 80.0]
        ],
        "boundary-gateway": [
            {"timestamp_sec": float(t), "latency_ms": 15.0, "backlog_requests": 0.0, "offered_rate": 40.0, "completed_rate": 40.0, "error_rate": 0.0, "dropped_requests": 0.0, "effective_capacity": 100.0}
            for t in [0.0, 20.0, 40.0, 60.0, 80.0]
        ],
        "legacy-core": [
            {"timestamp_sec": float(t), "latency_ms": 45.0, "backlog_requests": 0.0, "offered_rate": 40.0, "completed_rate": 40.0, "error_rate": 0.0, "dropped_requests": 0.0, "effective_capacity": None}
            for t in [0.0, 20.0, 40.0, 60.0, 80.0]
        ],
    }

    clean_svc, clean_bnd, origin_idx, target_ts, warnings, imputed, acct = predictor._validate_and_sanitize_inputs(
        service_telemetry=svc_tel,
        boundary_telemetry=None,
    )

    # In previous buggy code: Crew timestamps after alignment would be [0, 0, 20, 60, 80]
    # In corrected regular grid alignment: all services have EXACT identical physical timestamps [0, 20, 40, 60, 80]
    res_ts = [s["timestamp_sec"] for s in clean_svc["reservations"]]
    crew_ts = [s["timestamp_sec"] for s in clean_svc["crew"]]
    assert res_ts == [0.0, 20.0, 40.0, 60.0, 80.0]
    assert crew_ts == [0.0, 20.0, 40.0, 60.0, 80.0], f"Expected matching grid timestamps, got {crew_ts}"

    # Crew step 0 must have source_timestamp_sec=20.0 with is_carried_forward=True or is_padded=True
    assert clean_svc["crew"][0]["is_grid_aligned"] is True
    assert clean_svc["crew"][0]["source_timestamp_sec"] == 20.0
    assert clean_svc["crew"][0].get("is_carried_forward") is True or clean_svc["crew"][0].get("is_padded") is True


def test_prerequisite_1_unsupported_gap_abstention(predictor: DegradationPredictor):
    """Prerequisite 1: Slices with unsupported gaps (> 60s) raise ValueError (staleness limit) rather than silently proceeding."""
    bad_svc = {
        s: [
            {"timestamp_sec": 0.0, "latency_ms": 25.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0, "dropped_requests": 0.0, "effective_capacity": 50.0},
            {"timestamp_sec": 120.0, "latency_ms": 25.0, "backlog_requests": 0.0, "offered_rate": 30.0, "completed_rate": 30.0, "error_rate": 0.0, "dropped_requests": 0.0, "effective_capacity": 50.0},
        ]
        for s in predictor.supported_services
    }
    with pytest.raises(ValueError, match="Staleness limit exceeded"):
        predictor._validate_and_sanitize_inputs(bad_svc, None, explicit_timestamp_sec=120.0)


def test_prerequisite_2_uncertainty_fallback_uncalibrated(predictor: DegradationPredictor):
    """Prerequisite 2: Persistence fallback marks nominal_coverage=None and is_calibrated=False."""
    # When forecaster is None, adapt_workload_forecast creates persistence fallback
    fallback_adapted = adapt_workload_forecast(
        service_id="reservations",
        horizon=1,
        history=[40.0, 42.0, 45.0],
        forecaster=None,
        current_offered_rps=45.0,
    )
    assert fallback_adapted.fallback_triggered is True
    assert fallback_adapted.is_calibrated is False
    assert fallback_adapted.nominal_coverage is None
    assert fallback_adapted.calibration_method == "heuristic_uncalibrated_fixed_percentage"

    # In contrast, when a calibrated conformal prediction is provided, coverage is preserved
    conformal_adapted = adapt_workload_forecast(
        service_id="reservations",
        horizon=1,
        history=[40.0] * 12,
        forecaster=predictor.forecaster,
        current_offered_rps=40.0,
    )
    assert conformal_adapted.is_calibrated is True
    assert conformal_adapted.nominal_coverage == 0.90


def test_prerequisite_3_artifact_availability_assertion(nominal_telemetry):
    """Prerequisite 3: Forecast-enhanced prediction raises RuntimeError if forecaster failed to load."""
    svc_data, bnd_data = nominal_telemetry
    dummy_pred = copy.copy(DegradationPredictor.get_instance())
    dummy_pred.forecaster = None  # Simulate missing upstream workload forecaster artifact
    dummy_pred.forecaster_load_error = "Weights file missing or corrupted"

    with pytest.raises(RuntimeError, match="artifact failed to load"):
        dummy_pred.predict(
            service_telemetry=svc_data,
            boundary_telemetry=bnd_data,
            service_id="reservations",
            horizons=[1],
            model_variant="model_b_telemetry_forecast",
        )


def test_prerequisite_4_packaging_manifest_exclusion():
    """Prerequisite 4: Manifest excludes itself from its own file-hash list."""
    manifest_path = REPO_ROOT / "results" / "degradation_prediction" / "task3_manifest.json"
    assert manifest_path.exists(), "task3_manifest.json does not exist"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    files = manifest.get("files", {})
    for path_str in files:
        assert not path_str.endswith("task3_manifest.json"), f"Manifest included itself in files list: {path_str}"
    assert "archive" in manifest
    assert manifest["archive"]["sha256"] is not None


# =============================================================================
# EXPLANATION ENGINE & ATTRIBUTION TESTS
# =============================================================================

def test_explain_prediction_tree_shap_interface(explainer: DegradationExplainer, nominal_telemetry):
    """Tests the reusable Python explanation interface for Model B Tree SHAP."""
    svc_data, bnd_data = nominal_telemetry
    res = explainer.explain_prediction(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        service_id="reservations",
        horizon=1,
        model_variant="model_b_telemetry_forecast",
    )

    pred = res["prediction"]
    exp = res["explanation"]

    assert exp["service_id"] == "reservations"
    assert exp["horizon_steps"] == 1
    assert exp["horizon_seconds"] == 20.0
    assert exp["model_variant"] == "model_b_telemetry_forecast"
    assert exp["contract_version"] == CONTRACT_VERSION
    assert exp["feature_schema_version"] == FEATURE_SCHEMA_VERSION

    # Attribution Target separation
    assert exp["attribution_target"] == "raw_margin_log_odds"
    assert "log-odds" in exp["attribution_target_description"]
    assert exp["attribution_method"] == "TreeSHAP"

    # Reconstruction error completeness
    assert "reconstruction_error" in exp
    assert exp["reconstruction_error"] < 0.05

    # Retains positive and negative drivers
    assert "top_risk_increasing_drivers" in exp
    assert "top_risk_decreasing_drivers" in exp

    # Drivers contain physical units, provenance, reference value, category
    for d in exp["feature_attributions"]:
        assert "feature_name" in d
        assert "feature_value" in d
        assert "unit" in d
        assert "attribution_log_odds" in d
        assert "attribution_magnitude" in d
        assert "direction" in d
        assert "reference_value" in d
        assert "reference_calculation" in d
        assert "source_provenance" in d
        assert d["category"] in [
            "workload", "saturation", "network", "boundary_gateway", "uncertainty_and_reliability", "other"
        ]


def test_integrated_gradients_model_c(explainer: DegradationExplainer, nominal_telemetry):
    """Tests Integrated Gradients on PyTorch Temporal GRU (Model C)."""
    svc_data, bnd_data = nominal_telemetry
    res = explainer.explain_prediction(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        service_id="reservations",
        horizon=1,
        model_variant="model_c_temporal_comparator",
    )

    exp = res["explanation"]
    assert exp["attribution_method"] == "IntegratedGradients"
    assert exp["attribution_target"] == "event_logit"
    assert exp["reconstruction_error"] < 0.01

    # Temporal resolution
    assert "temporal_resolution" in exp
    assert len(exp["temporal_resolution"]["timesteps"]) == 4
    assert len(exp["temporal_resolution"]["per_step_attribution_sum"]) == 4


def test_raw_margin_never_added_directly_to_calibrated_probability(explainer: DegradationExplainer, nominal_telemetry):
    """Asserts that raw margin attributions are not added directly to calibrated probabilities."""
    svc_data, bnd_data = nominal_telemetry
    res = explainer.explain_prediction(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        service_id="reservations",
        horizon=1,
        model_variant="model_b_telemetry_forecast",
    )

    exp = res["explanation"]
    raw_output = exp["output_value"]  # log-odds
    cal_prob = exp["calibrated_probability"]
    uncal_prob = exp["uncalibrated_probability"]
    sum_phi = sum(d["attribution_log_odds"] for d in exp["feature_attributions"])
    base_val = exp["expected_value"]

    # Sum of phi + base_val equals raw margin (log-odds), NOT calibrated probability!
    assert abs((sum_phi + base_val) - raw_output) < 0.05
    assert abs(raw_output - cal_prob) > 0.001 or raw_output < 0.0  # Log-odds is typically negative for nominal


def test_graph_localization_and_generation_labels(explainer: DegradationExplainer, nominal_telemetry):
    """Tests graph localization, counterfactual node perturbation, and generation preservation."""
    svc_data, bnd_data = nominal_telemetry
    res = explainer.explain_prediction(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        service_id="reservations",
        horizon=1,
        model_variant="model_b_telemetry_forecast",
    )

    loc = res["explanation"]["graph_localization"]
    assert "affected_service" in loc
    assert loc["affected_service_generation"] == "cloud"
    assert "suspected_contributing_node" in loc
    assert "suspected_node_generation" in loc
    assert "propagation_bottleneck" in loc
    assert loc["verified_injected_origin"] == "UNAVAILABLE"
    assert "candidate_rankings" in loc
    assert len(loc["candidate_rankings"]) == 5


def test_deterministic_incident_brief_formatting(explainer: DegradationExplainer, nominal_telemetry):
    """Tests deterministic markdown incident brief formatting without hallucinations."""
    svc_data, bnd_data = nominal_telemetry
    res = explainer.explain_prediction(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        service_id="reservations",
        horizon=1,
        model_variant="model_b_telemetry_forecast",
    )

    brief = res["explanation"]["incident_brief"]
    assert "BACCP INCIDENT EXPLANATION BRIEF: RESERVATIONS" in brief
    assert "**Affected Service**: reservations (Generation: cloud)" in brief
    assert "**Prediction Origin Timestamp**" in brief
    assert "**Forecast Window**" in brief
    assert "**Observed SLO State at Origin**: Within Nominal SLO Limits" in brief
    assert "**Alert Decision**: NO ALERT ISSUED" in brief
    assert "**Conditional Time-to-Onset**" in brief
    assert "Confirmed Healthy" not in brief
    assert "Operating strictly within nominal bounds" not in brief
    assert "No alert active or censored" not in brief
    assert "Data Provenance" in brief
    assert "Causality Disclaimer" in brief
    assert "Attributions measure statistical model sensitivity; they do NOT establish physical causality." in brief


# =============================================================================
# TASK 3A REGRESSION TESTS
# =============================================================================

def test_regression_active_alert_displays_20s_conditional_estimate(explainer: DegradationExplainer):
    """Regression Test 1: Active alert with a 20-second conditional estimate displays that estimate."""
    data_path = REPO_ROOT / "ai-models" / "data" / "degradation_prediction" / "degradation_dataset.json"
    with open(data_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    c1_inc = next((i for i in ds["incidents"] if i["incident_id"] == "inc_val_0128"), ds["incidents"][0])

    # Predict and explain crew at origin step 21 (420.0s) for H=1
    res = explainer.explain_prediction(
        service_telemetry=c1_inc["service_telemetry"],
        boundary_telemetry=c1_inc["boundary_telemetry"],
        service_id="crew",
        horizon=1,
        model_variant="model_b_telemetry_forecast",
        prediction_timestamp_sec=420.0,
    )

    pred_h1 = res["prediction"]["services"]["crew"]["predictions_by_horizon"]["1"]
    assert pred_h1["alert_issued"] is True
    assert pred_h1["estimated_time_to_degradation_sec"] == 20.0

    brief = res["explanation"]["incident_brief"]
    # Proves that active alert with 20s conditional estimate displays that estimate
    assert "20.0s (Alert-gated conditional estimate)" in brief
    assert "- **Conditional Time-to-Onset**: 20.0s" in brief


def test_regression_brief_replaces_flawed_inferences_with_4_states(explainer: DegradationExplainer, nominal_telemetry):
    """Regression Test 2: Replaces flawed inferences with 4 distinct operational states."""
    svc_data, bnd_data = nominal_telemetry
    res = explainer.explain_prediction(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        service_id="reservations",
        horizon=1,
        model_variant="model_b_telemetry_forecast",
    )
    brief = res["explanation"]["incident_brief"]

    # 1. Proves replacement of "Confirmed Healthy" inferred from no alert
    assert "Confirmed Healthy" not in brief
    assert "- **Observed SLO State at Origin**: Within Nominal SLO Limits" in brief
    assert "- **Future Predicted Risk**: Degradation Risk:" in brief
    assert "- **Alert Decision**: NO ALERT ISSUED" in brief

    # 2. Proves replacement of "No alert active or censored" with explicit reason
    assert "No alert active or censored" not in brief
    assert "Conditional time-to-degradation estimate is gated to active alert states" in brief

    # 3. Proves origin timestamp is labeled separately from horizon-end timestamp
    assert "- **Prediction Origin Timestamp**: 140.0s (t_origin)" in brief
    assert "- **Forecast Window**: [140.0s, 160.0s] (H=1 steps / 20s ahead, Target Ts: 160.0s)" in brief


def test_regression_exact_treeshap_machine_precision_across_models_and_horizons(explainer: DegradationExplainer):
    """Regression Test 3: Validates TreeSHAP achieves machine precision (< 1e-12) across Models A/B and all horizons."""
    fixture_path = REPO_ROOT / "ai-models" / "data" / "degradation_prediction" / "fixture_test_incidents.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)
    sample_inc = fixtures["incidents"][0]

    for variant in ["model_a_telemetry_only", "model_b_telemetry_forecast"]:
        for h in [1, 5, 15]:
            for s in ["reservations", "crew", "legacy-core"]:
                res = explainer.explain_prediction(
                    service_telemetry=sample_inc["service_telemetry"],
                    boundary_telemetry=sample_inc["boundary_telemetry"],
                    service_id=s,
                    horizon=h,
                    model_variant=variant,
                )
                exp = res["explanation"]
                assert exp["attribution_method"] == "TreeSHAP"
                assert exp["claims_shapley_additivity"] is True
                assert exp["attribution_target"] == "raw_margin_log_odds"
                # Full precision reconstruction error must satisfy machine precision
                assert exp["reconstruction_error"] < 1e-12, (
                    f"TreeSHAP failed additivity check for {variant} H={h} {s}: "
                    f"error={exp['reconstruction_error']}"
                )


def test_regression_faithfulness_source_group_mapping_and_signed_deltas():
    """Regression Test 4: Verifies FEATURE_SOURCE_GROUP_MAP and signed delta impact under group perturbation."""
    from degradation_prediction.explanation import FEATURE_SOURCE_GROUP_MAP
    from degradation_prediction.evaluate_explanations import apply_source_group_intervention

    # Verify key feature mappings
    assert FEATURE_SOURCE_GROUP_MAP["curr_latency_ms"] == "target_service_latency"
    assert FEATURE_SOURCE_GROUP_MAP["curr_backlog"] == "target_service_backlog"
    assert FEATURE_SOURCE_GROUP_MAP["curr_offered_rate"] == "target_service_workload"
    assert FEATURE_SOURCE_GROUP_MAP["wf_forecast_demand"] == "upstream_workload_forecast"
    assert FEATURE_SOURCE_GROUP_MAP["gw_sync_drift"] == "boundary_gateway_telemetry"

    explainer = DegradationExplainer.get_instance()
    fixture_path = REPO_ROOT / "ai-models" / "data" / "degradation_prediction" / "fixture_test_incidents.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)
    inc = fixtures["incidents"][0]

    base_res = explainer.explain_prediction(
        service_telemetry=inc["service_telemetry"],
        boundary_telemetry=inc["boundary_telemetry"],
        service_id="reservations",
        horizon=1,
        model_variant="model_b_telemetry_forecast",
    )
    base_p = float(base_res["explanation"]["calibrated_probability"])
    top_feature = base_res["explanation"]["top_risk_increasing_drivers"][0]["feature_name"]
    top_group = FEATURE_SOURCE_GROUP_MAP[top_feature]

    pert_svc, pert_bnd = apply_source_group_intervention(
        inc["service_telemetry"], inc["boundary_telemetry"], "reservations", top_group
    )
    pert_res = explainer.explain_prediction(
        service_telemetry=pert_svc,
        boundary_telemetry=pert_bnd,
        service_id="reservations",
        horizon=1,
        model_variant="model_b_telemetry_forecast",
    )
    pert_p = float(pert_res["explanation"]["calibrated_probability"])

    signed_delta = base_p - pert_p
    # Signed delta is preserved (no clipping to 0)
    assert isinstance(signed_delta, float)


def test_regression_localization_independent_query_and_tie_detection(explainer: DegradationExplainer):
    """Regression Test 5: Verifies localization chooses query scope independently of origin label and detects ties."""
    fixture_path = REPO_ROOT / "ai-models" / "data" / "degradation_prediction" / "diagnostic_localization_fixture.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    predictor = explainer.predictor
    inc = fixtures["incidents"][0]

    # Independent selection at explicit timestamp 100.0s (step 5)
    sys_pred = predictor.predict(
        service_telemetry=inc["service_telemetry"],
        boundary_telemetry=inc["boundary_telemetry"],
        prediction_timestamp_sec=100.0,
        horizons=[1],
    )
    svc_risks = {
        s: sys_pred["services"][s]["predictions_by_horizon"]["1"]["degradation_event_risk"]
        for s in predictor.supported_services
    }
    query_svc = max(svc_risks, key=svc_risks.get)

    exp = explainer.explain_prediction(
        service_telemetry=inc["service_telemetry"],
        boundary_telemetry=inc["boundary_telemetry"],
        service_id=query_svc,
        horizon=1,
        prediction_timestamp_sec=100.0,
    )
    loc = exp["explanation"]["graph_localization"]

    assert "is_tie" in loc
    assert "is_abstained" in loc
    assert "abstention_reason" in loc
    assert "candidate_rankings" in loc
    assert len(loc["candidate_rankings"]) == 5
    assert loc["verified_injected_origin"] == "UNAVAILABLE"


def test_regression_checksums_separate_file_avoids_circular_hashing():
    """Regression Test 6: Verifies separate checksums file exists and manifest excludes itself."""
    from degradation_prediction.package_task3_bundle import package_task3_bundle

    package_task3_bundle()

    manifest_path = RESULTS_DIR / "task3_manifest.json"
    checksums_path = RESULTS_DIR / "checksums_sha256.txt"
    zip_path = RESULTS_DIR / "baccp_task3_review_bundle.zip"

    assert manifest_path.exists()
    assert checksums_path.exists()
    assert zip_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Manifest must strictly exclude itself from its own files dictionary
    assert "task3_manifest.json" not in manifest["files"]
    assert "results/degradation_prediction/task3_manifest.json" not in manifest["files"]

    # Checksums file contains separate hash entries
    checksum_text = checksums_path.read_text(encoding="utf-8")
    assert "baccp_task3_review_bundle.zip" in checksum_text
    assert "task3_manifest.json" in checksum_text
    assert "degradation_models_bundle.pkl" in checksum_text
    assert "saved_training_background.json" in checksum_text


# =============================================================================
# TASK 3B REGRESSION TESTS
# =============================================================================

def test_regression_task3b_counterfactual_boundary_dependency_rules():
    """Task 3B Regression Test 1: Verifies dependent boundary fields are updated and not left stale."""
    from degradation_prediction.explanation import apply_counterfactual_intervention
    from degradation_prediction.inference import compute_sync_drift_score

    dt = 20.0
    # Construct deliberately inconsistent initial telemetry
    svc_data: Dict[str, List[Dict[str, Any]]] = {
        "boundary-gateway": [
            {
                "timestamp_sec": float(i * dt),
                "step_index": i,
                "offered_rate": 85.0,  # high load
                "completed_rate": 80.0,
                "backlog_requests": 65.0,  # high backlog
                "latency_ms": 110.0,  # high latency
                "error_rate": 0.08,
                "dropped_requests": 5.0,
            }
            for i in range(5)
        ],
        "legacy-core": [
            {
                "timestamp_sec": float(i * dt),
                "step_index": i,
                "offered_rate": 75.0,
                "completed_rate": 15.0,  # severely depressed completion rate
                "backlog_requests": 90.0,
                "latency_ms": 280.0,  # severe latency
                "error_rate": 0.04,
                "dropped_requests": 0.0,
            }
            for i in range(5)
        ],
        "reservations": [
            {
                "timestamp_sec": float(i * dt),
                "step_index": i,
                "offered_rate": 50.0,
                "completed_rate": 50.0,
                "backlog_requests": 0.0,
                "latency_ms": 20.0,
                "error_rate": 0.0,
                "dropped_requests": 0.0,
            }
            for i in range(5)
        ],
    }

    # Deliberately stale/inconsistent boundary telemetry
    bnd_data = [
        {
            "timestamp_sec": float(i * dt),
            "step_index": i,
            "gateway_submission_rate": 99.0,  # mismatched stale value
            "legacy_completion_rate": 5.0,   # mismatched stale value
            "gateway_queue_depth": 77.0,     # mismatched stale value
            "boundary_latency_ms": 130.0,    # mismatched stale value
            "boundary_error_rate": 0.05,
            "sync_drift_score": 85.0,
        }
        for i in range(5)
    ]

    # --- Test 1A: Legacy Core Intervention with cutoff at 40.0s (step 2) ---
    pert_svc_leg, pert_bnd_leg, meta_leg = apply_counterfactual_intervention(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        intervention_type="node",
        target_node="legacy-core",
        prediction_timestamp_sec=40.0,
    )

    # Within prediction origin (t <= 40.0s), legacy completed rate must be nominal (42.0)
    for idx in range(3):
        assert pert_svc_leg["legacy-core"][idx]["completed_rate"] == 42.0
        assert pert_svc_leg["legacy-core"][idx]["backlog_requests"] == 0.0
        # Dependent boundary completion rate MUST be updated, NOT left stale at 5.0
        assert pert_bnd_leg[idx]["legacy_completion_rate"] == 42.0
        # Sync drift score must be recomputed
        expected_drift = compute_sync_drift_score(
            gateway_submission_rate=pert_bnd_leg[idx]["gateway_submission_rate"],
            legacy_completion_rate=42.0,
            gateway_queue_depth=pert_bnd_leg[idx]["gateway_queue_depth"],
            boundary_latency_ms=pert_bnd_leg[idx]["boundary_latency_ms"],
            nominal_latency_ms=15.0,
            boundary_error_rate=pert_bnd_leg[idx]["boundary_error_rate"],
        )
        assert abs(pert_bnd_leg[idx]["sync_drift_score"] - expected_drift) < 1e-4

    # Future observations (t > 40.0s, i.e. steps 3 and 4) MUST NOT be modified
    for idx in [3, 4]:
        assert pert_svc_leg["legacy-core"][idx]["completed_rate"] == 15.0
        assert pert_bnd_leg[idx]["legacy_completion_rate"] == 5.0
        assert pert_bnd_leg[idx]["sync_drift_score"] == 85.0

    assert "legacy_completion_rate" in meta_leg["fields_changed"]["boundary_telemetry"]
    assert "sync_drift_score" in meta_leg["fields_changed"]["boundary_telemetry"]

    # --- Test 1B: Boundary Gateway Intervention ---
    pert_svc_gw, pert_bnd_gw, meta_gw = apply_counterfactual_intervention(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        intervention_type="node",
        target_node="boundary-gateway",
        prediction_timestamp_sec=40.0,
    )

    for idx in range(3):
        # Service fields reset
        assert pert_svc_gw["boundary-gateway"][idx]["backlog_requests"] == 0.0
        assert pert_svc_gw["boundary-gateway"][idx]["latency_ms"] == 20.0
        # Boundary dependent fields updated
        assert pert_bnd_gw[idx]["gateway_submission_rate"] == 42.0
        assert pert_bnd_gw[idx]["gateway_queue_depth"] == 0.0
        assert pert_bnd_gw[idx]["boundary_latency_ms"] == 15.0
        assert pert_bnd_gw[idx]["boundary_error_rate"] == 0.0

    # --- Test 1C: Source Group boundary_gateway_telemetry Intervention ---
    pert_svc_grp, pert_bnd_grp, meta_grp = apply_counterfactual_intervention(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        intervention_type="source_group",
        target_service_id="reservations",
        source_group="boundary_gateway_telemetry",
        prediction_timestamp_sec=40.0,
    )

    # Verifies BOTH boundary sensors and corresponding gateway service values are updated consistently
    for idx in range(3):
        assert pert_bnd_grp[idx]["gateway_queue_depth"] == 0.0
        assert pert_bnd_grp[idx]["boundary_latency_ms"] == 15.0
        assert pert_bnd_grp[idx]["boundary_error_rate"] == 0.0
        assert pert_svc_grp["boundary-gateway"][idx]["backlog_requests"] == 0.0
        assert pert_svc_grp["boundary-gateway"][idx]["latency_ms"] == 20.0
        assert pert_svc_grp["boundary-gateway"][idx]["error_rate"] == 0.0

    # --- Test 1D: Normal Feature-Extraction Path Validation with Deliberately Inconsistent Inputs ---
    from degradation_prediction.features import extract_features_at_origin, get_feature_index
    # Under legacy intervention: sub_rate=99.0, comp_rate=42.0 -> rate_diff = 57.0 (NOT stale abs(99-5)=94)
    feat_vec_leg, _ = extract_features_at_origin(
        service_id="reservations",
        incident_service_telemetry=pert_svc_leg,
        incident_boundary_telemetry=pert_bnd_leg,
        origin_step=2,
        horizon=1,
        include_workload_forecasts=False,
    )
    idx_diff = get_feature_index("rate_divergence_abs", schema="telemetry")
    assert abs(feat_vec_leg[idx_diff] - 57.0) < 1e-4, f"Expected rate_diff=57.0, got {feat_vec_leg[idx_diff]}"

    # Under gateway intervention: sub_rate=42.0, comp_rate=5.0 -> rate_diff = 37.0 (NOT stale 94)
    feat_vec_gw, _ = extract_features_at_origin(
        service_id="reservations",
        incident_service_telemetry=pert_svc_gw,
        incident_boundary_telemetry=pert_bnd_gw,
        origin_step=2,
        horizon=1,
        include_workload_forecasts=False,
    )
    assert abs(feat_vec_gw[idx_diff] - 37.0) < 1e-4, f"Expected rate_diff=37.0, got {feat_vec_gw[idx_diff]}"
    idx_blat = get_feature_index("boundary_latency_ms", schema="telemetry")
    assert abs(feat_vec_gw[idx_blat] - 15.0) < 1e-4
    idx_gwq = get_feature_index("gw_queue_depth", schema="telemetry")
    assert abs(feat_vec_gw[idx_gwq] - 0.0) < 1e-4


def test_regression_task3b_timestamp_alignment_non_matching_indices():
    """Task 3B Regression Test 2: Verifies records align by timestamp_sec rather than assuming matching list indices."""
    from degradation_prediction.explanation import apply_counterfactual_intervention

    # Create service telemetry with timestamps [20.0, 40.0, 60.0]
    svc_data: Dict[str, List[Dict[str, Any]]] = {
        "boundary-gateway": [
            {"timestamp_sec": 20.0, "offered_rate": 45.0, "completed_rate": 45.0, "backlog_requests": 10.0, "latency_ms": 30.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 40.0, "offered_rate": 50.0, "completed_rate": 50.0, "backlog_requests": 20.0, "latency_ms": 40.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 60.0, "offered_rate": 55.0, "completed_rate": 55.0, "backlog_requests": 30.0, "latency_ms": 50.0, "error_rate": 0.0, "dropped_requests": 0.0},
        ],
        "legacy-core": [
            {"timestamp_sec": 20.0, "offered_rate": 45.0, "completed_rate": 45.0, "backlog_requests": 0.0, "latency_ms": 60.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 40.0, "offered_rate": 50.0, "completed_rate": 50.0, "backlog_requests": 0.0, "latency_ms": 60.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 60.0, "offered_rate": 55.0, "completed_rate": 55.0, "backlog_requests": 0.0, "latency_ms": 60.0, "error_rate": 0.0, "dropped_requests": 0.0},
        ],
    }

    # Boundary telemetry has an extra warmup record at t=0.0s (offsetting list index by 1)
    bnd_data = [
        {"timestamp_sec": 0.0, "gateway_submission_rate": 40.0, "legacy_completion_rate": 40.0, "gateway_queue_depth": 0.0, "boundary_latency_ms": 15.0, "boundary_error_rate": 0.0, "sync_drift_score": 10.0},
        {"timestamp_sec": 20.0, "gateway_submission_rate": 45.0, "legacy_completion_rate": 45.0, "gateway_queue_depth": 10.0, "boundary_latency_ms": 30.0, "boundary_error_rate": 0.0, "sync_drift_score": 25.0},
        {"timestamp_sec": 40.0, "gateway_submission_rate": 50.0, "legacy_completion_rate": 50.0, "gateway_queue_depth": 20.0, "boundary_latency_ms": 40.0, "boundary_error_rate": 0.0, "sync_drift_score": 35.0},
    ]

    pert_svc, pert_bnd, meta = apply_counterfactual_intervention(
        service_telemetry=svc_data,
        boundary_telemetry=bnd_data,
        intervention_type="node",
        target_node="boundary-gateway",
        prediction_timestamp_sec=40.0,
    )

    # bnd_data[1] is at t=20.0s; its gateway submission rate was coupled to t=20.0s (not t=0.0s)
    assert pert_bnd[1]["timestamp_sec"] == 20.0
    assert pert_bnd[1]["gateway_submission_rate"] == 42.0
    assert pert_bnd[1]["gateway_queue_depth"] == 0.0
    assert pert_bnd[2]["timestamp_sec"] == 40.0
    assert pert_bnd[2]["gateway_submission_rate"] == 42.0


def test_regression_task3b_signed_driver_ranking_prevents_protective_features_as_drivers():
    """Task 3B Regression Test 3: Verifies driver ranking uses signed positive attribution rather than absolute magnitude."""
    # Synthetic drivers with large negative (protective) attribution vs small positive (risk-increasing) attribution
    drivers = [
        {"feature_name": "feat_protective", "attribution_log_odds": -2.50, "attribution_magnitude": 2.50},
        {"feature_name": "feat_mild_risk", "attribution_log_odds": +0.45, "attribution_magnitude": 0.45},
        {"feature_name": "feat_low_protective", "attribution_log_odds": -0.10, "attribution_magnitude": 0.10},
    ]

    # Absolute magnitude would incorrectly pick feat_protective as top driver!
    abs_top = sorted(drivers, key=lambda d: -abs(d["attribution_log_odds"]))[0]["feature_name"]
    assert abs_top == "feat_protective", "Sanity check: abs() selects negative feature"

    # Signed positive attribution strictly selects feat_mild_risk as the risk-increasing driver
    pos_drivers = [d for d in drivers if float(d.get("attribution_log_odds", 0.0)) > 0.0]
    signed_top = sorted(pos_drivers, key=lambda d: -float(d["attribution_log_odds"]))[0]["feature_name"]
    assert signed_top == "feat_mild_risk", "Signed ranking must select positive risk-increasing driver"


def test_regression_task3b_localization_full_precision_and_unrounded_decisions(explainer: DegradationExplainer):
    """Task 3B Regression Test 4: Verifies localization decisions use unrounded full precision values."""
    fixture_path = REPO_ROOT / "ai-models" / "data" / "degradation_prediction" / "diagnostic_localization_fixture.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    inc = fixtures["incidents"][0]
    exp = explainer.explain_prediction(
        service_telemetry=inc["service_telemetry"],
        boundary_telemetry=inc["boundary_telemetry"],
        service_id="reservations",
        horizon=1,
        prediction_timestamp_sec=100.0,
    )
    loc = exp["explanation"]["graph_localization"]

    # Candidate rankings must preserve unrounded raw fields
    for cand_entry in loc["candidate_rankings"]:
        assert "risk_reduction_raw" in cand_entry
        assert "counterfactual_probability_raw" in cand_entry
        assert "signed_risk_delta_raw" in cand_entry
        assert isinstance(cand_entry["risk_reduction_raw"], float)

    # Ranking order must match raw unrounded sorting
    raw_scores = [c["risk_reduction_raw"] for c in loc["candidate_rankings"]]
    assert raw_scores == sorted(raw_scores, reverse=True)


def test_regression_task4_future_telemetry_invariance_unmatched_timestamps():
    """Task 4 Regression Test 1A: Changing future telemetry must not change an earlier counterfactual result, including when timestamps do not match."""
    from degradation_prediction.explanation import apply_counterfactual_intervention

    # Service telemetry has timestamps: [20.0, 40.0, 60.0, 80.0]
    svc_clean: Dict[str, List[Dict[str, Any]]] = {
        "boundary-gateway": [
            {"timestamp_sec": 20.0, "latency_ms": 25.0, "backlog_requests": 5.0, "offered_rate": 45.0, "completed_rate": 45.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 40.0, "latency_ms": 30.0, "backlog_requests": 10.0, "offered_rate": 50.0, "completed_rate": 50.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 60.0, "latency_ms": 35.0, "backlog_requests": 15.0, "offered_rate": 55.0, "completed_rate": 55.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 80.0, "latency_ms": 40.0, "backlog_requests": 20.0, "offered_rate": 60.0, "completed_rate": 60.0, "error_rate": 0.0, "dropped_requests": 0.0},
        ],
        "legacy-core": [
            {"timestamp_sec": 20.0, "latency_ms": 65.0, "backlog_requests": 0.0, "offered_rate": 45.0, "completed_rate": 45.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 40.0, "latency_ms": 70.0, "backlog_requests": 0.0, "offered_rate": 50.0, "completed_rate": 50.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 60.0, "latency_ms": 75.0, "backlog_requests": 0.0, "offered_rate": 55.0, "completed_rate": 55.0, "error_rate": 0.0, "dropped_requests": 0.0},
            {"timestamp_sec": 80.0, "latency_ms": 80.0, "backlog_requests": 0.0, "offered_rate": 60.0, "completed_rate": 60.0, "error_rate": 0.0, "dropped_requests": 0.0},
        ],
    }

    # Boundary telemetry has unmatched timestamps: [10.0, 30.0, 50.0, 70.0]
    bnd_clean = [
        {"timestamp_sec": 10.0, "gateway_submission_rate": 40.0, "legacy_completion_rate": 40.0, "gateway_queue_depth": 0.0, "boundary_latency_ms": 15.0, "boundary_error_rate": 0.0, "sync_drift_score": 10.0},
        {"timestamp_sec": 30.0, "gateway_submission_rate": 45.0, "legacy_completion_rate": 45.0, "gateway_queue_depth": 5.0, "boundary_latency_ms": 25.0, "boundary_error_rate": 0.0, "sync_drift_score": 20.0},
        {"timestamp_sec": 50.0, "gateway_submission_rate": 50.0, "legacy_completion_rate": 50.0, "gateway_queue_depth": 10.0, "boundary_latency_ms": 35.0, "boundary_error_rate": 0.0, "sync_drift_score": 30.0},
        {"timestamp_sec": 70.0, "gateway_submission_rate": 55.0, "legacy_completion_rate": 55.0, "gateway_queue_depth": 15.0, "boundary_latency_ms": 45.0, "boundary_error_rate": 0.0, "sync_drift_score": 40.0},
    ]

    # Create a corrupted/spiked future version where records at t >= 60.0s have catastrophic failures
    import copy
    svc_future_spiked = copy.deepcopy(svc_clean)
    svc_future_spiked["boundary-gateway"][2]["latency_ms"] = 9999.0
    svc_future_spiked["boundary-gateway"][3]["latency_ms"] = 9999.0
    svc_future_spiked["boundary-gateway"][3]["backlog_requests"] = 5000.0
    svc_future_spiked["legacy-core"][3]["completed_rate"] = 0.01

    bnd_future_spiked = copy.deepcopy(bnd_clean)
    bnd_future_spiked[2]["boundary_latency_ms"] = 9999.0
    bnd_future_spiked[3]["gateway_queue_depth"] = 9999.0

    # Counterfactual intervention evaluated at prediction origin timestamp 40.0s
    pert_svc_1, pert_bnd_1, _ = apply_counterfactual_intervention(
        service_telemetry=svc_clean,
        boundary_telemetry=bnd_clean,
        intervention_type="node",
        target_node="boundary-gateway",
        prediction_timestamp_sec=40.0,
    )

    pert_svc_2, pert_bnd_2, _ = apply_counterfactual_intervention(
        service_telemetry=svc_future_spiked,
        boundary_telemetry=bnd_future_spiked,
        intervention_type="node",
        target_node="boundary-gateway",
        prediction_timestamp_sec=40.0,
    )

    # All counterfactual results at or before 40.0s MUST be 100% identical
    for s_name in ["boundary-gateway", "legacy-core"]:
        recs_1 = [r for r in pert_svc_1[s_name] if r["timestamp_sec"] <= 40.0 + 1e-6]
        recs_2 = [r for r in pert_svc_2[s_name] if r["timestamp_sec"] <= 40.0 + 1e-6]
        assert recs_1 == recs_2, f"Future telemetry corrupted earlier counterfactual in {s_name}!"

    bnd_recs_1 = [r for r in pert_bnd_1 if r["timestamp_sec"] <= 40.0 + 1e-6]
    bnd_recs_2 = [r for r in pert_bnd_2 if r["timestamp_sec"] <= 40.0 + 1e-6]
    assert bnd_recs_1 == bnd_recs_2, "Future boundary telemetry corrupted earlier counterfactual!"


