"""Tests for the RCAEval external-validation step (ai-models/real_data/rcaeval.py).

Run offline on a tiny committed excerpt of three real RE1-TT cases
(ai-models/tests/fixtures/rcaeval_tt_tiny, CC-BY 4.0).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

AI_MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_MODELS))

from real_data import rcaeval  # noqa: E402

FIXTURE = rcaeval.FIXTURE


@pytest.fixture(scope="module")
def built():
    return rcaeval.build_samples(FIXTURE)


def test_service_mapping_follows_documented_roles():
    assert rcaeval.node_of("ts-order-service") == 0          # order -> reservations
    assert rcaeval.node_of("ts-preserve-service") == 0       # booking -> reservations
    assert rcaeval.node_of("ts-route-service") == 1          # scheduling -> crew
    assert rcaeval.node_of("ts-consign-service") == 2        # luggage consignment -> baggage
    assert rcaeval.node_of("ts-ui-dashboard") == 3           # edge gateway -> boundary-gateway
    assert rcaeval.node_of("ts-gateway-service") == 3
    assert rcaeval.node_of("ts-order-mongo") == 4            # databases -> legacy-core
    assert rcaeval.node_of("ts-voucher-mysql") == 4
    assert rcaeval.node_of("ts-news-service") is None


def test_column_parsing():
    assert rcaeval.split_column("ts-order-service_latency-90") == ("ts-order-service", "latency-90")
    assert rcaeval.split_column("ts-order-mongo_cpu") == ("ts-order-mongo", "cpu")
    assert rcaeval.split_column("time") is None


def test_cascade_labels_on_real_cases(built):
    _, meta = built
    labels = {c["case_id"]: c["cascade"] for c in meta["cases"]}
    assert labels == {"ts-order-service_disk/1": 0, "ts-route-service_delay/1": 1, "ts-train-service_loss/1": 1}
    for c in meta["cases"]:
        assert c["root_service"] not in c["affected_services"]  # "beyond the root cause"


def test_samples_match_cascade_predictor_schema(built):
    samples, _ = built
    assert len(samples) == 6  # one pre- and one post-injection window per case
    assert all(s["cascade_label"] == 0 for s in samples if s["phase"] == "pre")
    for s in samples:
        assert len(s["snapshots"]) == 5
        for snap in s["snapshots"]:
            x = np.array(snap["node_features"])
            assert x.shape == (5, 7)
            assert (x[:, 0] >= 0).all() and (x[:, 0] <= 10).all()   # _node_features clipping
            assert (x[:, 2:4] >= 0).all() and (x[:, 2:4] <= 1).all()
            assert x[:, 4:].tolist() == [[1, 0, 0]] * 3 + [[0, 1, 0], [0, 0, 1]]
            assert 0.0 <= snap["sync_drift_score"] <= 100.0


def test_delay_fault_raises_reservation_latency(built):
    samples, _ = built
    by_id = {s["sample_id"]: s for s in samples}
    pre = by_id["ts-route-service_delay/1#pre"]["snapshots"]
    post = by_id["ts-route-service_delay/1#post"]["snapshots"]
    assert np.mean([sn["node_features"][0][0] for sn in post]) > np.mean([sn["node_features"][0][0] for sn in pre])


def test_evaluation_runs_end_to_end(built, tmp_path):
    samples, meta = built
    result = rcaeval.evaluate(samples, meta, epochs=2, seed=0, train_cv=True, root=FIXTURE)
    r = result["results"]
    assert result["n_cases"] == 3 and result["label_noise_check"]["cases"] == 3
    cv = r["trained_on_rcaeval (leave-one-root-service-out CV)"]
    for model in ("baccp_hetero_rgcn", "logistic_regression_flat"):
        assert cv[model]["all_windows"]["n"] == 6
    for scores in r["zero_shot"].values():  # only present when trace weights have been trained locally
        assert scores["all_windows"]["roc_auc"] is not None
