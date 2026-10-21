"""Tests for the real-data pipeline (ai-models/real_data).

Run offline against the small committed artifacts in
ai-models/data/real_processed/ (derived from BTS, DB1B, Azure, Alibaba, GCP).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

AI_MODELS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_MODELS))

from real_data import cloud_health  # noqa: E402
from real_data.airline_ops import detect_disruption_windows  # noqa: E402
from real_data.catalog import CATALOG  # noqa: E402
from real_data.reservation_workload import (  # noqa: E402
    PROCESSED, WorkloadConfig, generate_workload, lead_convolve,
)
from real_data.trace_simulator import (  # noqa: E402
    SimConfig, cut_episodes, load_alibaba, load_incident_distributions, normal_day_mask,
    sample_faults, simulate_timeline, upsample_workload,
)

needs_processed = pytest.mark.skipif(not (PROCESSED / "manifest.json").exists(),
                                     reason="run ai-models/real_data/prepare.py first")


def test_catalog_entries_have_provenance():
    for spec in CATALOG.values():
        assert spec.homepage.startswith("https://")
        assert spec.licence and spec.citation and spec.role


def test_lead_convolve_shifts_events_earlier():
    x = np.zeros(20)
    x[10] = 1.0  # one departure at minute 10
    lags = np.arange(2, 5)  # activity 2..4 minutes before departure
    y = lead_convolve(x, lags, np.ones(3) / 3)
    assert np.allclose(y[6:9], 1 / 3) and y.sum() == pytest.approx(1.0)
    assert y[9:].sum() == 0  # nothing after the lead window


def test_lead_convolve_handles_negative_lags():
    x = np.zeros(10)
    x[5] = 1.0
    y = lead_convolve(x, np.arange(-1, 2), np.ones(3) / 3)  # 1 min after .. 1 min before
    assert np.allclose(y[4:7], 1 / 3)


@needs_processed
def test_real_disruptions_are_detected_from_bts():
    daily = pd.read_csv(PROCESSED / "bts_daily_carrier_summary.csv", parse_dates=["FlightDate"])
    wins = detect_disruption_windows(daily)
    wn = [w for w in wins if w.carrier == "WN" and w.start.year == 2022]
    dl = [w for w in wins if w.carrier == "DL" and w.start.year == 2024]
    # Southwest crew-scheduling meltdown, Dec 2022
    assert any(w.start <= pd.Timestamp("2022-12-26") <= w.end and w.cancelled_flights > 10000 for w in wn)
    # Delta / CrowdStrike, Jul 2024
    assert any(w.start <= pd.Timestamp("2024-07-21") <= w.end and w.cancelled_flights > 3000 for w in dl)


@needs_processed
def test_workload_reflects_real_disruption():
    wl = generate_workload(WorkloadConfig(carrier="WN", month="202212"))
    assert wl.attrs["pax_per_departure_source"].startswith("BTS DB1B")
    assert (wl[["reservations_rps", "crew_rps", "baggage_rps", "legacy_writes_rps"]] >= 0).all().all()
    normal = wl.loc["2022-12-05":"2022-12-15"]
    melt = wl.loc["2022-12-26":"2022-12-28"]
    assert melt["rebooking_rps"].mean() > 50 * max(normal["rebooking_rps"].mean(), 1e-3)
    assert melt["crew_rps"].mean() > 1.5 * normal["crew_rps"].mean()
    assert melt["baggage_rps"].mean() < normal["baggage_rps"].mean()  # cancelled flights carry no bags


@needs_processed
def test_incident_distributions_are_real():
    dist = load_incident_distributions()
    assert dist["source"] == "Google Cloud Service Health"
    assert len(dist["duration_min"]) > 100
    assert abs(sum(dist["fault_class_share"].values()) - 1.0) < 1e-6


def test_fault_classifier_keywords():
    assert cloud_health.classify_fault("Elevated network latency in us-east1") == "network-delay"
    assert cloud_health.classify_fault("Dataflow batch jobs stuck in queue") == "batch-job-stall"


@needs_processed
def test_simulation_labels_and_episodes():
    cfg = SimConfig()
    spm = cfg.steps_per_min
    wl = upsample_workload(generate_workload(WorkloadConfig(carrier="DL", month="202407")).iloc[: 3 * 1440],
                           cfg.step_sec)
    rng = np.random.RandomState(0)
    tu, fl = load_alibaba()
    faults = [{"start": 600, "end": 900, "fault_type": "batch-job-stall", "fault_level": "high",
               "duration_min": 300, "ramp_min": 10.0}]
    sim = simulate_timeline(wl, faults, cfg, tu, fl, normal_day_mask(wl, "DL"), rng)
    assert 0.4 <= tu <= 0.8
    assert sim.attrs["step_sec"] == cfg.step_sec
    assert sim["gw_latency_ms"].max() <= 2 * cfg.gateway_timeout_ms
    assert sim["cascade_state"].iloc[620 * spm:900 * spm].mean() > 0.5   # stall -> cascade
    assert sim["cascade_state"].iloc[:550 * spm].sum() == 0              # healthy before the fault
    # the fault ramps up: intensity grows over the 10-minute ramp, then stays at 1
    x = sim["fault_intensity"].to_numpy()
    assert 0 < x[600 * spm] < x[605 * spm] < 1 and x[615 * spm] == 1.0
    # a cascade always involves the boundary
    assert not (sim["cascade_raw"].astype(bool) & (sim["gw_error"] < 0.05) & (sim["gw_latency_ms"] < 100)).any()
    eps = cut_episodes(wl, sim, cfg, 6, rng, tag="DL:202407")
    pos = [e for e in eps if e["cascade_occurred"]]
    assert pos and all(e["manifest_time"] >= 600 for e in pos)    # warm-up respected
    assert all(len(e["snapshots"]) == cfg.episode_minutes // cfg.snapshot_every_min for e in eps)
    assert len(eps[0]["snapshots"][0]["node_seq"]) == 5


def test_upsample_preserves_counts_and_rates():
    idx = pd.date_range("2024-07-01", periods=3, freq="1min")
    wl = pd.DataFrame({"scheduled_departures_pm": [6.0, 0.0, 12.0], "legacy_writes_rps": [10.0, 20.0, 30.0]},
                      index=idx)
    up = upsample_workload(wl, 10)
    assert len(up) == 18
    assert up["scheduled_departures_pm"].sum() == pytest.approx(18.0)   # counts preserved
    assert up["legacy_writes_rps"].iloc[3] == pytest.approx(15.0)       # linear between minutes


def test_persistence_ignores_single_step_blips():
    cfg = SimConfig()
    wl = upsample_workload(generate_workload(WorkloadConfig(carrier="DL", month="202407")).iloc[:600],
                           cfg.step_sec)
    tu, fl = load_alibaba()
    sim = simulate_timeline(wl, [], cfg, tu, fl, np.ones(len(wl), bool), np.random.RandomState(3))
    raw, state = sim["cascade_raw"].to_numpy(), sim["cascade_state"].to_numpy()
    assert (state <= raw).all()                                  # state only where raw held
    k = cfg.cascade_persist_sec // cfg.step_sec
    for i in np.flatnonzero(np.diff(np.r_[0, state]) == 1):     # every cascade start lasts >= k steps
        assert raw[i:i + k].all()


@needs_processed
def test_fault_sampling_respects_cap():
    rng = np.random.RandomState(1)
    faults = sample_faults(30 * 1440, load_incident_distributions(), SimConfig(), rng)
    assert faults and all(f["end"] - f["start"] <= SimConfig().fault_max_duration_min for f in faults)


@needs_processed
def test_processed_manifest_declares_real_provenance():
    m = json.loads((PROCESSED / "manifest.json").read_text())
    assert "real public datasets" in m["provenance"]
    assert m["gcp_incidents"]["n_incidents"] > 100
