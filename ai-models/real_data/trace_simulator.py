"""Trace-driven simulation of the airline hybrid stack under *real* workload.

What is real and what is simulated
----------------------------------
REAL   : request demand per service, minute by minute (BTS flights + DB1B
         passengers + Azure traffic shape), real disruption surges (BTS
         cancellations/delays), fault durations and fault-type mix (Google
         Cloud incident history), cloud utilisation targets and noise
         (Alibaba production microservice nodes).
SIMULATED: the five airline services themselves (reservations, crew,
         baggage, boundary-gateway, legacy-core) - their queueing behaviour
         follows standard queueing relations (Little's law, M/M/1 delay
         factor, fixed connection pool to the mainframe, lagged autoscaling
         of the cloud tier).

How the cloud manages the load
------------------------------
* Cloud tier (reservations / crew / baggage) autoscales: every
  ``scale_interval_min`` it sets replicas so utilisation returns to the target
  utilisation observed in Alibaba's production cluster. Scale-out lags demand.
* Legacy core (mainframe) does **not** autoscale: fixed capacity sized for
  normal-day peaks, reached through a fixed connection pool at the gateway.
* So when demand surges, autoscaling pushes *more* traffic into the boundary -
  the elastic/inelastic mismatch that BACCP is designed to predict.

Time resolution and faults
--------------------------
The per-minute workload is linearly interpolated onto ``step_sec`` (10 s)
steps and the month is simulated continuously (queues carry state across
hours). Most faults *worsen gradually* (ramp of 2-20 min, as in incidents that
start with "elevated latency"); a share are abrupt (hardware / data-centre
failures such as Alaska Airlines 2025). A cascade must persist for
``cascade_persist_sec`` so one noisy step does not count.

60-minute episodes are cut in the exact format consumed by the team's
training code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

AI_MODELS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_MODELS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AI_MODELS_DIR))

from importlib import import_module  # noqa: E402

from real_data.reservation_workload import PROCESSED, WorkloadConfig, generate_workload  # noqa: E402,F401

FAULT_LEVEL_PARAMS = {  # mirrors testbed/chaos/profiles.json
    "network-delay": {"low": 100, "medium": 500, "high": 1500},      # added ms
    "connection-drop": {"low": 10, "medium": 2, "high": 1},          # drop 1-in-N
    "batch-job-stall": {"low": 1, "medium": 5, "high": 15},          # stall intensity
}

# Nominal testbed request rates the graph features are expressed in, so the
# trained models see the same feature scale as the live testbed.
TESTBED_NOMINAL_RPS = {"reservations": 20.0, "crew": 15.0, "baggage": 10.0, "boundary-gateway": 45.0}

# Columns of the workload that are counts per minute (not rates): summed, not interpolated.
COUNT_COLUMNS = ("scheduled_departures_pm", "cancellations_pm", "delayed_pm")


@dataclass
class SimConfig:
    step_sec: int = 10                     # simulation resolution
    legacy_headroom: float = 1.25          # legacy capacity = headroom x p99 normal-day demand
    legacy_base_latency_ms: float = 15.0
    pool_concurrency_factor: float = 10.0  # connections = capacity x base_latency x factor
    gateway_buffer_sec: float = 30.0       # queue length the gateway holds before rejecting
    gateway_timeout_ms: float = 3000.0
    gateway_retries: int = 2               # client retries on a dropped call (retry amplification)
    scale_interval_min: int = 3            # autoscaler reaction time
    target_utilisation: Optional[float] = None  # None -> from Alibaba trace
    replica_rps: Dict[str, float] = None   # per-replica capacity of cloud services
    fault_mean_interarrival_h: float = 8.0   # oversampled vs real GCP rate (documented)
    # Real incident durations are long-tailed (GCP mean ~15 h) because they measure the
    # provider's whole incident; an airline fails over / mitigates far sooner, so the
    # empirical distribution is truncated here (shape below the cap is kept).
    fault_max_duration_min: float = 240.0
    # ASSUMPTION: most incidents degrade gradually ("elevated latency" first); some are abrupt.
    abrupt_fault_share: float = 0.3
    fault_ramp_min: Tuple[float, float] = (2.0, 20.0)
    cascade_persist_sec: int = 30          # a cascade must last this long to count
    fine_every_sec: int = 30               # compact node-feature history kept per episode
    episode_minutes: int = 60
    snapshot_every_min: int = 2
    seed: int = 7

    def __post_init__(self):
        if self.replica_rps is None:
            self.replica_rps = {"reservations": 40.0, "crew": 10.0, "baggage": 25.0}

    @property
    def steps_per_min(self) -> int:
        return max(1, 60 // self.step_sec)


# ---------------------------------------------------------------------- inputs

def load_alibaba(processed: Path = PROCESSED) -> Tuple[float, np.ndarray]:
    p = processed / "alibaba_utilisation.json"
    fl_p = processed / "alibaba_cpu_fluctuations.npy"
    target = 0.6
    if p.exists():
        cpu = json.loads(p.read_text())["cpu"]
        # Production clusters are run well below saturation; we use their P90 as
        # the autoscaler target so that typical peaks are absorbed.
        target = float(np.clip(cpu["p90"] / 100.0, 0.4, 0.8))
    fl = np.load(fl_p) if fl_p.exists() else np.ones(1000)
    return target, np.clip(fl, 0.2, 3.0)


def load_incident_distributions(processed: Path = PROCESSED) -> Dict[str, Any]:
    p = processed / "gcp_incidents.csv.gz"
    if not p.exists():
        return {"duration_min": np.array([30.0, 60.0, 120.0]),
                "fault_class_share": {k: 1 / 3 for k in FAULT_LEVEL_PARAMS}, "source": "fallback"}
    inc = pd.read_csv(p, compression="gzip")
    dur = inc["duration_min"].dropna().to_numpy()
    dur = dur[(dur > 0) & (dur < 60 * 24 * 14)]
    share = inc["fault_class"].value_counts(normalize=True).to_dict()
    return {"duration_min": dur, "fault_class_share": share, "source": "Google Cloud Service Health",
            "severity_share": inc["severity"].value_counts(normalize=True).to_dict()}


def upsample_workload(wl: pd.DataFrame, step_sec: int) -> pd.DataFrame:
    """Per-minute workload -> ``step_sec`` grid. Rates are linearly interpolated;
    per-minute counts are spread evenly so window sums are preserved."""
    spm = max(1, 60 // step_sec)
    if spm == 1:
        return wl
    idx = pd.date_range(wl.index[0], wl.index[-1] + pd.Timedelta(seconds=60 - step_sec), freq=f"{step_sec}s")
    out = pd.DataFrame(index=idx)
    t_min = (wl.index - wl.index[0]).total_seconds().to_numpy()
    t_fine = (idx - wl.index[0]).total_seconds().to_numpy()
    for c in wl.columns:
        v = wl[c].to_numpy(dtype=float)
        if c in COUNT_COLUMNS:
            out[c] = np.repeat(v, spm)[: len(idx)] / spm
        else:
            out[c] = np.interp(t_fine, t_min, v)
    out.attrs.update(wl.attrs)
    out.attrs["step_sec"] = step_sec
    return out


def sample_faults(n_minutes: int, dist: Dict[str, Any], cfg: SimConfig,
                  rng: np.random.RandomState) -> List[Dict[str, Any]]:
    """Fault schedule (times in minutes): real durations / types, oversampled frequency."""
    faults, t = [], 0.0
    classes = list(dist["fault_class_share"].keys())
    probs = np.array([dist["fault_class_share"][c] for c in classes], float)
    probs /= probs.sum()
    sev = dist.get("severity_share") or {}
    level_p = np.array([sev.get("low", 1.0), sev.get("medium", 1.0), sev.get("high", 1.0)], float)
    level_p = level_p / level_p.sum() if level_p.sum() else np.ones(3) / 3
    while True:
        t += rng.exponential(cfg.fault_mean_interarrival_h * 60.0)
        if t >= n_minutes:
            break
        dur = float(max(5.0, min(rng.choice(dist["duration_min"]), cfg.fault_max_duration_min)))
        ramp = 0.0 if rng.rand() < cfg.abrupt_fault_share else float(rng.uniform(*cfg.fault_ramp_min))
        faults.append({
            "start": round(t, 2), "end": round(min(float(n_minutes), t + dur), 2),
            "fault_type": str(rng.choice(classes, p=probs)),
            "fault_level": str(rng.choice(["low", "medium", "high"], p=level_p)),
            "duration_min": round(dur, 1), "ramp_min": round(min(ramp, dur), 1),
        })
    return faults


# ---------------------------------------------------------------------- simulation

def simulate_timeline(wl: pd.DataFrame, faults: List[Dict[str, Any]], cfg: SimConfig,
                      target_util: float, cpu_fluct: np.ndarray, normal_mask: np.ndarray,
                      rng: np.random.RandomState) -> pd.DataFrame:
    """Simulate on ``wl``'s own grid (pass ``upsample_workload(wl, cfg.step_sec)``)."""
    n = len(wl)
    dt = float(wl.attrs.get("step_sec", 60))
    spm = max(1, int(round(60.0 / dt)))
    lam_legacy = wl["legacy_writes_rps"].to_numpy()
    cfgw = wl.attrs["config"]
    res_txn = (wl["booking_rps"] * cfgw["legacy_calls_per_booking"]
               + wl["checkin_rps"] + wl["rebooking_rps"] * cfgw["rebooking_calls_per_pax"]).to_numpy()
    demand = {"reservations": res_txn, "crew": wl["crew_rps"].to_numpy(), "baggage": wl["baggage_rps"].to_numpy()}
    # share of each cloud service's requests that must cross the boundary
    legacy_share = {
        "reservations": np.clip((wl["booking_rps"] * cfgw["legacy_calls_per_booking"]
                                 + wl["checkin_rps"] * cfgw["legacy_calls_per_checkin"]
                                 + wl["rebooking_rps"] * cfgw["rebooking_calls_per_pax"]).to_numpy()
                                / np.maximum(res_txn, 1e-6), 0, 1),
        "crew": np.clip((wl["crew_repair_rps"] + 0.25 * wl["crew_routine_rps"]).to_numpy()
                        / np.maximum(demand["crew"], 1e-6), 0, 1),
        "baggage": np.full(n, cfgw["baggage_legacy_share"]),
    }

    # capacity sized on per-minute demand (as an operator would plan it)
    lam_min = lam_legacy[normal_mask]
    if spm > 1:
        lam_min = lam_min[: len(lam_min) // spm * spm].reshape(-1, spm).mean(1)
    C0 = cfg.legacy_headroom * float(np.quantile(lam_min, 0.99))
    pool = C0 * (cfg.legacy_base_latency_ms / 1000.0) * cfg.pool_concurrency_factor
    q_max = C0 * cfg.gateway_buffer_sec

    # fault schedule on the step grid: type, level and intensity in [0, 1]
    fault_type = np.array([""] * n, dtype=object)
    fault_level = np.array([""] * n, dtype=object)
    intensity = np.zeros(n)
    for f in faults:
        a, b = int(f["start"] * spm), min(n, int(math.ceil(f["end"] * spm)))
        if a >= n:
            continue
        fault_type[a:b] = f["fault_type"]
        fault_level[a:b] = f["fault_level"]
        ramp_steps = max(1.0, f.get("ramp_min", 0.0) * spm)
        intensity[a:b] = np.minimum(1.0, (np.arange(b - a) + 1) / ramp_steps)

    # pre-drawn noise (vectorised for speed)
    lat_noise = rng.randn(n)
    exp20 = rng.exponential(20.0, n)
    exp300 = rng.exponential(300.0, n)
    cpu_noise = rng.choice(cpu_fluct, size=(n, 3))

    replicas = {s: max(1, math.ceil(demand[s][0] / (cfg.replica_rps[s] * target_util))) for s in demand}
    scale_every = cfg.scale_interval_min * spm
    Q = 0.0
    out = {k: np.zeros(n) for k in [
        "legacy_capacity", "legacy_latency_ms", "legacy_error", "legacy_completed",
        "gw_queue", "gw_latency_ms", "gw_error", "drift",
        "res_latency_ms", "res_error", "res_util", "crew_latency_ms", "crew_error", "crew_util",
        "bag_latency_ms", "bag_error", "bag_util", "res_replicas", "crew_replicas", "bag_replicas"]}
    features = import_module("ai-models.graph-builder.features")
    base_lat = {"reservations": 12.0, "crew": 10.0, "baggage": 14.0}
    svc_keys = [("reservations", "res", 0), ("crew", "crew", 1), ("baggage", "bag", 2)]
    r = cfg.gateway_retries
    b0 = cfg.legacy_base_latency_ms

    for t in range(n):
        # ---- legacy core under (possibly ramping) fault
        lat = b0 * (1.0 + 0.05 * lat_noise[t])
        drop = 0.0
        ft, x = fault_type[t], intensity[t]
        if ft:
            p = FAULT_LEVEL_PARAMS[ft][fault_level[t]]
            if ft == "network-delay":
                lat = b0 + x * p * (1.0 + 0.1 * lat_noise[t])
            elif ft == "connection-drop":
                drop = x * min(0.95, 1.0 / max(1.0, p))
                lat = b0 + x * (35.0 + exp20[t])
            elif ft == "batch-job-stall":
                lat = b0 + x * (1185.0 + exp300[t])
                drop = x * min(0.9, 0.05 * p)
        lat = max(5.0, lat)
        # fixed connection pool => throughput <= pool / latency (Little's law)
        cap = min(C0, pool / (lat / 1000.0))

        # ---- gateway queue. Dropped calls are retried: small drop rates are masked,
        # large ones multiply the load on the boundary (1 + d + d^2 + ...).
        amplification = sum(drop ** k for k in range(r + 1))
        unrecovered = drop ** (r + 1)
        arrivals = lam_legacy[t] * dt * amplification
        served = min(Q + arrivals, cap * dt)
        Q = Q + arrivals - served
        rejected = max(0.0, Q - q_max)
        Q = min(Q, q_max)
        gw_lat = lat + (Q / max(cap, 1e-6)) * 1000.0
        timeout_frac = float(np.clip((gw_lat - cfg.gateway_timeout_ms) / cfg.gateway_timeout_ms, 0, 1))
        gw_err = float(np.clip(rejected / max(arrivals, 1e-6) + unrecovered + timeout_frac, 0, 1))
        completed = served / dt * (1.0 - drop)
        gw_lat_obs = min(gw_lat, 2.0 * cfg.gateway_timeout_ms)  # beyond this requests are dead (in gw_err)
        drift = features.compute_sync_drift_score(
            gateway_submission_rate=lam_legacy[t], legacy_completion_rate=completed,
            gateway_queue_depth=Q / max(C0, 1e-6) * 45.0,  # express backlog in testbed units
            boundary_latency_ms=gw_lat, boundary_error_rate=gw_err)

        # ---- autoscaled cloud services
        if t % scale_every == 0:
            for s in demand:
                replicas[s] = max(1, math.ceil(demand[s][t] / (cfg.replica_rps[s] * target_util)))
        for s, key, j in svc_keys:
            rho = demand[s][t] / (replicas[s] * cfg.replica_rps[s])
            own_lat = base_lat[s] / max(0.05, 1.0 - min(rho, 0.95))
            own_err = max(0.0, (rho - 1.0) / max(rho, 1e-6))
            share = legacy_share[s][t]
            out[f"{key}_latency_ms"][t] = own_lat + share * gw_lat_obs
            out[f"{key}_error"][t] = min(1.0, own_err + share * gw_err)
            out[f"{key}_util"][t] = min(1.0, rho * target_util * cpu_noise[t, j])
            out[f"{key}_replicas"][t] = replicas[s]

        out["legacy_capacity"][t] = cap
        out["legacy_latency_ms"][t] = lat
        out["legacy_error"][t] = drop
        out["legacy_completed"][t] = completed
        out["gw_queue"][t] = Q
        out["gw_latency_ms"][t] = gw_lat_obs
        out["gw_error"][t] = gw_err
        out["drift"][t] = drift

    sim = pd.DataFrame(out, index=wl.index)
    sim["fault_type"] = fault_type
    sim["fault_level"] = fault_level
    sim["fault_intensity"] = intensity
    sim["legacy_demand"] = lam_legacy
    sim["overloaded"] = (lam_legacy > sim["legacy_capacity"]).astype(int)
    # A *cross-boundary* cascade: the customer-facing reservations path is degraded
    # (same thresholds as the synthetic dataset) AND the degradation comes through the
    # gateway. Pure cloud-side saturation while the autoscaler catches up is kept as a
    # (hard) negative - that is the cloud managing load, not a boundary cascade.
    res_degraded = (sim["res_error"] >= 0.10) | (sim["res_latency_ms"] >= 150.0)
    boundary_degraded = (sim["gw_error"] >= 0.05) | (sim["gw_latency_ms"] >= 100.0)
    raw = (res_degraded & boundary_degraded).to_numpy()
    k = max(1, int(round(cfg.cascade_persist_sec / dt)))
    persistent = raw.copy()
    for j in range(1, k):  # state counts only if it holds for k consecutive steps from here
        persistent[:-j] &= raw[j:]
        persistent[-j:] = False
    # once a persistent run starts, mark the whole run
    run = np.zeros(n, dtype=bool)
    on = False
    for i in range(n):
        on = (on and raw[i]) or persistent[i]
        run[i] = on
    sim["cascade_raw"] = raw.astype(int)
    sim["cloud_saturation"] = (res_degraded & ~boundary_degraded).astype(int)
    sim["cascade_state"] = run.astype(int)
    sim.attrs["legacy_capacity_nominal"] = C0
    sim.attrs["step_sec"] = dt
    return sim


# ---------------------------------------------------------------------- episodes

def _node_metrics(wl_row: pd.Series, sim_row: pd.Series, scale: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    gw_rate = wl_row["boundary_gateway_rps"] * scale["boundary-gateway"]
    q_units = min(150.0, sim_row["gw_queue"] / max(1e-6, sim_row.get("legacy_capacity_nominal", 1.0)) * 45.0)
    return {
        "legacy-core": {
            "request_rate": sim_row["legacy_completed"] * scale["boundary-gateway"],
            "error_rate": sim_row["legacy_error"], "latency_ms": sim_row["legacy_latency_ms"],
            "queue_depth": 0.0, "cpu_load": min(1.0, 0.2 + sim_row["legacy_latency_ms"] / 1500.0 * 0.7),
            "sync_drift_score": sim_row["drift"]},
        "boundary-gateway": {
            "request_rate": gw_rate, "error_rate": sim_row["gw_error"], "latency_ms": sim_row["gw_latency_ms"],
            "queue_depth": q_units, "cpu_load": min(1.0, 0.25 + min(q_units, 150.0) / 150.0 * 0.6),
            "sync_drift_score": sim_row["drift"]},
        "reservations": {
            "request_rate": wl_row["_res_txn"] * scale["reservations"], "error_rate": sim_row["res_error"],
            "latency_ms": sim_row["res_latency_ms"], "queue_depth": max(0.0, q_units * 0.2),
            "cpu_load": sim_row["res_util"], "sync_drift_score": sim_row["drift"]},
        "crew": {
            "request_rate": wl_row["crew_rps"] * scale["crew"], "error_rate": sim_row["crew_error"],
            "latency_ms": sim_row["crew_latency_ms"], "queue_depth": 0.0,
            "cpu_load": sim_row["crew_util"], "sync_drift_score": sim_row["drift"]},
        "baggage": {
            "request_rate": wl_row["baggage_rps"] * scale["baggage"], "error_rate": sim_row["bag_error"],
            "latency_ms": sim_row["bag_latency_ms"], "queue_depth": 0.0,
            "cpu_load": sim_row["bag_util"], "sync_drift_score": sim_row["drift"]},
    }


# Node order used by ai-models/cascade_predictor (dataset.json "nodes").
CANONICAL_NODES = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]

EDGES = [
    {"source": "boundary-gateway", "destination": "legacy-core", "protocol": "tcp", "observation_count": 120},
    {"source": "legacy-core", "destination": "boundary-gateway", "protocol": "tcp", "observation_count": 118},
    {"source": "boundary-gateway", "destination": "reservations", "protocol": "http", "observation_count": 85},
    {"source": "reservations", "destination": "boundary-gateway", "protocol": "http", "observation_count": 85},
    {"source": "reservations", "destination": "crew", "protocol": "http", "observation_count": 40},
    {"source": "reservations", "destination": "baggage", "protocol": "http", "observation_count": 35},
]


def cut_episodes(wl: pd.DataFrame, sim: pd.DataFrame, cfg: SimConfig, n_target: int,
                 rng: np.random.RandomState, tag: str, warmup_min: int = 10) -> List[Dict[str, Any]]:
    """Cut 60-minute episodes, balanced between cascade and non-cascade.

    ``wl`` and ``sim`` must share the same grid. Cascade episode: the
    reservations path enters (persistent) cascade state at >= ``warmup_min``
    into the window and was healthy at the window start. Lead time =
    manifestation - first precursor (fault onset or legacy demand above
    capacity) inside the window, in seconds.
    """
    schema = import_module("ai-models.graph-builder.schema")
    graph = schema.GenerationTypedGraph()
    dt = float(sim.attrs.get("step_sec", 60))
    spm = max(1, int(round(60.0 / dt)))
    E, S, W = cfg.episode_minutes * spm, cfg.snapshot_every_min * spm, warmup_min * spm
    cas = sim["cascade_state"].to_numpy()
    onset_precursor = ((sim["fault_type"].to_numpy() != "") | (sim["overloaded"].to_numpy() == 1)).astype(int)

    wl = wl.copy()
    c = wl.attrs["config"]
    wl["_res_txn"] = wl["booking_rps"] * c["legacy_calls_per_booking"] + wl["checkin_rps"] \
        + wl["rebooking_rps"] * c["rebooking_calls_per_pax"]
    scale = {
        "reservations": TESTBED_NOMINAL_RPS["reservations"] / max(1e-6, wl["_res_txn"].median()),
        "crew": TESTBED_NOMINAL_RPS["crew"] / max(1e-6, wl["crew_rps"].median()),
        "baggage": TESTBED_NOMINAL_RPS["baggage"] / max(1e-6, wl["baggage_rps"].median()),
        "boundary-gateway": TESTBED_NOMINAL_RPS["boundary-gateway"] / max(1e-6, wl["boundary_gateway_rps"].median()),
    }
    sim = sim.assign(legacy_capacity_nominal=sim.attrs["legacy_capacity_nominal"])

    # candidate window starts (every 5 min). Hard negatives = a fault, a capacity overload
    # or cloud-side saturation happened in the window but did NOT become a boundary cascade.
    sat = sim["cloud_saturation"].to_numpy()
    starts_pos, starts_hard, starts_easy = [], [], []
    for s0 in range(0, len(sim) - E, 5 * spm):
        w = cas[s0:s0 + E]
        if w[:W].any():
            continue
        if w.any():
            starts_pos.append(s0)
        elif onset_precursor[s0:s0 + E].any() or sat[s0:s0 + E].any():
            starts_hard.append(s0)
        else:
            starts_easy.append(s0)
    for lst in (starts_pos, starts_hard, starts_easy):
        rng.shuffle(lst)
    n_pos = min(n_target // 2, len(starts_pos))
    n_neg = n_target - n_pos
    n_hard = min(len(starts_hard), n_neg // 2)
    chosen = ([(s, 1) for s in starts_pos[:n_pos]] + [(s, 0) for s in starts_hard[:n_hard]]
              + [(s, 0) for s in starts_easy[:n_neg - n_hard]])

    episodes = []
    for s0, _ in chosen:
        w_cas = cas[s0:s0 + E]
        manifest = int(np.argmax(w_cas)) if w_cas.any() else None
        pre = onset_precursor[s0:s0 + E]
        onset = int(np.argmax(pre)) if pre.any() else None
        ft = next((x for x in sim["fault_type"].iloc[s0:s0 + E] if x), "none")
        lv = next((x for x in sim["fault_level"].iloc[s0:s0 + E] if x), "none")
        driver = "fault" if ft != "none" else ("demand-surge" if pre.any() else "none")
        snaps = []
        for k in range(0, E, S):
            i = s0 + k
            nm = _node_metrics(wl.iloc[i], sim.iloc[i], scale)
            t_sec = float(k * dt)
            snap = graph.build_snapshot(nm, EDGES, timestamp=t_sec)
            fx, fe, fa = graph.to_flat_tensors(snap)
            node_seq = [[nm[n]["latency_ms"], nm[n]["request_rate"], nm[n]["error_rate"], nm[n]["cpu_load"]]
                        for n in CANONICAL_NODES]
            snaps.append({"time": t_sec, "drift_score": float(sim["drift"].iloc[i]), "node_seq": node_seq,
                          "flat_x": fx, "flat_edge_index": fe, "flat_edge_attr": fa,
                          "x_dict": snap.x_dict, "edge_index_dict": snap.edge_index_dict,
                          "edge_attr_dict": snap.edge_attr_dict})
        # compact 30-second history (node features + drift) for early-warning models
        F = max(1, int(round(cfg.fine_every_sec / dt)))
        fine_t, fine_nodes, fine_drift = [], [], []
        for k in range(0, E, F):
            i = s0 + k
            nm = _node_metrics(wl.iloc[i], sim.iloc[i], scale)
            fine_t.append(float(k * dt))
            fine_nodes.append([[round(nm[n]["latency_ms"], 2), round(nm[n]["request_rate"], 3),
                                round(nm[n]["error_rate"], 4), round(nm[n]["cpu_load"], 4)]
                               for n in CANONICAL_NODES])
            fine_drift.append(round(float(sim["drift"].iloc[i]), 2))
        occurred = int(manifest is not None)
        lead = 300.0
        if occurred:
            lead = float(max(1, manifest - onset) * dt) if onset is not None and onset <= manifest else dt
        affected = []
        if occurred:
            affected = ["boundary-gateway", "reservations"]
            win = sim.iloc[s0 + manifest: s0 + E]
            if (win["crew_error"] >= 0.08).any():
                affected.append("crew")
            if (win["bag_error"] >= 0.08).any():
                affected.append("baggage")
        severity = "LOW"
        if occurred:
            severity = "CRITICAL" if lead < 300 or len(affected) >= 3 else ("HIGH" if lead < 900 else "MEDIUM")
        episodes.append({
            "fault_type": ft, "fault_level": lv, "driver": driver,
            "manifest_time": float(manifest * dt) if occurred else None,
            "onset_time": float(onset * dt) if onset is not None else None,
            "cascade_occurred": occurred, "lead_time_seconds": lead, "severity": severity,
            "root_cause_node": "boundary-gateway" if occurred else "none",
            "affected_nodes": affected, "snapshot_count": len(snaps), "snapshots": snaps,
            "fine": {"time": fine_t, "node_seq": fine_nodes, "drift": fine_drift},
            "real_window_start": str(sim.index[s0]), "source": tag,
            "real_flights_in_window": int(round(wl["scheduled_departures_pm"].iloc[s0:s0 + E].sum())),
            "real_cancellations_in_window": int(round(wl["cancellations_pm"].iloc[s0:s0 + E].sum())),
        })
    return episodes


def normal_day_mask(wl: pd.DataFrame, carrier: str, processed: Path = PROCESSED) -> np.ndarray:
    """True for timestamps outside the carrier's real disruption windows."""
    p = processed / "bts_disruption_windows.json"
    mask = np.ones(len(wl), dtype=bool)
    if p.exists():
        for w in json.loads(p.read_text()):
            if w["carrier"] == carrier:
                lo, hi = pd.Timestamp(w["start"]), pd.Timestamp(w["end"]) + pd.Timedelta(days=1)
                mask &= ~((wl.index >= lo) & (wl.index < hi))
    return mask
