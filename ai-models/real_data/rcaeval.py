"""External validation of the BACCP cascade predictor on RCAEval (RE1, Train-Ticket).

RCAEval (Pham et al. 2025, arXiv 2412.17015, Zenodo 14504481 / 14590730) is a public
benchmark of fault-injection experiments on real microservice deployments. RE1-TT
holds 125 metrics-only cases on Train-Ticket, a ~40-service ticket-booking system:
5 fault types (cpu, mem, disk, delay, loss) x 5 root-cause services x 5 repetitions.
Unlike our trace-driven dataset, the services and their telemetry here are real.

This module

1. maps Train-Ticket service metrics onto the 5-node BACCP schema used by
   ai-models/cascade_predictor (see SERVICE_TO_NODE below),
2. labels each injected fault as a *cascade* when it propagates beyond the
   root-cause service (another service's latency becomes anomalous),
3. builds 5-snapshot samples before and after the injection,
4. evaluates the trace-trained weights (weights/trace_full, weights/trace_early)
   zero-shot, and trains/evaluates directly on RCAEval with grouped CV,
5. writes results/real_data/rcaeval_metrics.json.

    python ai-models/real_data/download.py rcaeval      # 280 MB -> data/raw/rcaeval/RE1-TT
    python ai-models/real_data/rcaeval.py               # -> results/real_data/rcaeval_metrics.json

Mapping Train-Ticket -> BACCP nodes
-----------------------------------
Train-Ticket is a railway system, so the mapping is by *function*, not by name:

- reservations     : order, booking (``preserve``), seat, ticket search and payment
                     services - the passenger booking path.
- crew             : route / train / station / schedule-planning services - the
                     operational resource-scheduling side (the closest analogue of
                     crew and equipment scheduling).
- baggage          : consignment (``consign`` = luggage consignment), food and
                     assurance - the ancillary services attached to a ticket.
- boundary-gateway : the edge gateway (``ts-gateway-service`` or, in older
                     deployments, the ``ts-ui-dashboard`` nginx front door).
- legacy-core      : the database / stateful tier (MySQL, MongoDB, Redis, Nacos) -
                     the system of record behind the services, playing the part of
                     the mainframe. Auth/user/verification services also sit here:
                     they are stateful account stores every booking depends on.

Anything not matched is left out of the aggregation (listed in the output as
``unmapped_services``). RE1 ``simple_data.csv`` gives, per service and second:
``cpu`` (percent of one core), ``mem`` (bytes), and for the 29 Istio-meshed services
``workload`` (req/s), ``latency-50`` / ``latency-90`` (seconds) and, for 4 of them,
``error`` (failed req/s). Per node and second: latency = workload-weighted mean of
member p50 latencies (members with no traffic report histogram noise, so they get
no weight), request rate = sum, error rate = failed / total requests, CPU = mean.
Units become ms / req/s / fraction / fraction-of-a-core before the *unchanged*
``train_on_trace._node_features`` normalisation is applied. Database pods only
expose CPU and memory, so legacy-core latency and rate come from the account
services listed above.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

AI_MODELS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_MODELS_DIR.parent
sys.path.insert(0, str(AI_MODELS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from real_data.train_on_trace import _node_features  # noqa: E402

RAW = REPO_ROOT / "data" / "raw" / "rcaeval" / "RE1-TT"
RESULTS = REPO_ROOT / "results" / "real_data"
WEIGHTS = AI_MODELS_DIR / "weights"
FIXTURE = AI_MODELS_DIR / "tests" / "fixtures" / "rcaeval_tt_tiny"

NODES = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]
FAULT_TYPES = ["cpu", "mem", "disk", "delay", "loss"]

# Ordered (pattern, node): first match wins. Patterns are matched against the
# service name with the "ts-" prefix and "-service" suffix stripped.
SERVICE_TO_NODE: List[Tuple[str, int]] = [
    (r"(mysql|mongo|redis|nacos|rabbitmq|db$|database)", 4),
    (r"^(auth|user|verification-code|security|admin-user)$", 4),
    (r"^(gateway|ui-dashboard)$", 3),
    (r"^(order|order-other|preserve|preserve-other|seat|ticketinfo|travel|travel2|price|"
     r"basic|inside-payment|payment|cancel|rebook|execute|contacts|notification|voucher|"
     r"admin-order|admin-travel|wait-order|ticket-office)$", 0),
    (r"^(route|train|station|config|route-plan|travel-plan|admin-route|admin-basic-info|"
     r"train-food|station-food|delivery)$", 1),
    (r"^(consign|consign-price|food|food-map|food-delivery|assurance)$", 2),
]


def node_of(service: str) -> Optional[int]:
    short = re.sub(r"^ts-", "", service.lower())
    short = re.sub(r"-(service|mysql|mongo)$", lambda m: "" if m.group(1) == "service" else m.group(0), short)
    for pat, node in SERVICE_TO_NODE:
        if re.search(pat, short):
            return node
    return None


# ---------------------------------------------------------------- loading

METRIC_RE = re.compile(r"^(?P<svc>.+?)_(?P<metric>cpu|mem|diskio|disk|latency[-_]?50|latency[-_]?90|latency|"
                       r"lat50|lat90|workload|error|errors|tx|rx|socket)$")


def split_column(col: str) -> Optional[Tuple[str, str]]:
    m = METRIC_RE.match(col)
    if not m:
        return None
    metric = m.group("metric").replace("_", "-")
    metric = {"lat50": "latency-50", "lat90": "latency-90", "latency50": "latency-50",
              "latency90": "latency-90", "errors": "error"}.get(metric.replace("-", ""), metric)
    return m.group("svc"), metric


def find_cases(root: Path) -> List[Path]:
    """Case directories are the ones holding inject_time.txt."""
    return sorted(p.parent for p in root.rglob("inject_time.txt"))


def parse_case_name(case_dir: Path, root: Path) -> Dict[str, Any]:
    """RE1-TT/<service>_<fault>/<rep>/  ->  root service, fault type, repetition."""
    rel = case_dir.relative_to(root).parts
    head = rel[0]
    svc, fault = head.rsplit("_", 1)
    rep = rel[1] if len(rel) > 1 else "1"
    return {"case_id": "/".join(rel), "root_service": svc, "fault_type": fault, "rep": rep}


def load_case(case_dir: Path) -> Tuple[pd.DataFrame, float]:
    for name in ("simple_data.csv", "data.csv", "metrics.csv"):
        if (case_dir / name).exists():
            df = pd.read_csv(case_dir / name)
            break
    else:
        df = pd.read_json(case_dir / "metrics.json")
    df = df.loc[:, ~df.columns.duplicated()]
    inject = float((case_dir / "inject_time.txt").read_text().strip())
    return df.sort_values("time").reset_index(drop=True), inject


# ---------------------------------------------------------------- mapping


def service_frame(df: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
    out: Dict[str, Dict[str, pd.Series]] = {}
    for col in df.columns:
        sm = split_column(col)
        if sm:
            out.setdefault(sm[0], {})[sm[1]] = pd.to_numeric(df[col], errors="coerce")
    return out


def node_timeseries(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """[T, 5, 4] raw (latency_ms, req_rate, error_rate, cpu) per BACCP node."""
    svcs = service_frame(df)
    T = len(df)
    lat = [[] for _ in NODES]  # (latency_ms, workload) pairs
    rate = [[] for _ in NODES]
    err = [[] for _ in NODES]
    cpu = [[] for _ in NODES]

    def arr(x: pd.Series) -> np.ndarray:
        return np.nan_to_num(x.to_numpy(dtype=float), nan=0.0)
    mapping, unmapped = {n: [] for n in NODES}, []
    for svc, m in svcs.items():
        k = node_of(svc)
        if k is None:
            unmapped.append(svc)
            continue
        mapping[NODES[k]].append(svc)
        wl = arr(m["workload"]) if "workload" in m else np.zeros(T)
        if "latency-50" in m:
            lat[k].append((arr(m["latency-50"]) * 1000.0, wl))  # s -> ms
        if "workload" in m:
            rate[k].append(wl)
        if "error" in m:
            err[k].append(arr(m["error"]))
        if "cpu" in m:
            cpu[k].append(arr(m["cpu"]) / 100.0)  # percent of a core -> fraction

    raw = np.zeros((T, len(NODES), 4))
    for k in range(len(NODES)):
        if lat[k]:
            L = np.vstack([x for x, _ in lat[k]])
            W = np.vstack([w for _, w in lat[k]])
            tot = W.sum(0)
            raw[:, k, 0] = np.where(tot > 0, (L * W).sum(0) / np.maximum(tot, 1e-9), np.median(L, axis=0))
        total_rate = np.vstack(rate[k]).sum(0) if rate[k] else np.zeros(T)
        raw[:, k, 1] = total_rate
        if err[k]:
            raw[:, k, 2] = np.clip(np.vstack(err[k]).sum(0) / np.maximum(total_rate, 1.0), 0.0, 1.0)
        if cpu[k]:
            raw[:, k, 3] = np.vstack(cpu[k]).mean(0)
    return raw, {"mapping": {k: sorted(v) for k, v in mapping.items()}, "unmapped": sorted(unmapped)}


# ---------------------------------------------------------------- labelling


def robust_z(post: np.ndarray, pre: np.ndarray) -> float:
    med = np.nanmedian(pre)
    mad = 1.4826 * np.nanmedian(np.abs(pre - med)) + 1e-9
    floor = 0.05 * abs(med) + 1e-6  # avoid huge z on near-constant series
    return float((np.nanmedian(post) - med) / max(mad, floor))


def propagation(df: pd.DataFrame, inject: float, root_service: str, window_s: float = 300.0,
                z_thr: float = 8.0, ratio_thr: float = 2.0, min_rps: float = 0.5) -> Dict[str, Any]:
    """Which services other than the root show a sustained latency anomaly after injection.

    Thresholds were fixed with a placebo test (``placebo_rate``): the same rule applied
    to two fault-free windows before injection fires in 1 of 125 cases (z>=5, 1.5x
    fired in 11), so a positive label is very unlikely to be baseline noise.
    """
    t = df["time"].to_numpy(dtype=float)
    pre_m = (t >= inject - window_s) & (t < inject)
    post_m = (t >= inject) & (t < inject + window_s)
    affected = []
    for svc, m in service_frame(df).items():
        if svc == root_service:
            continue
        s = m.get("latency-90")
        if s is None or "workload" not in m:
            continue
        if np.nanmean(m["workload"].to_numpy(dtype=float)[pre_m | post_m]) < min_rps:
            continue  # idle services report histogram noise, not latency
        x = s.to_numpy(dtype=float)
        pre, post = x[pre_m], x[post_m]
        if np.isnan(pre).all() or np.isnan(post).all():
            continue
        z = robust_z(post, pre)
        ratio = (np.nanmedian(post) + 1e-9) / (np.nanmedian(pre) + 1e-9)
        if z >= z_thr and ratio >= ratio_thr:
            affected.append(svc)
    return {"cascade": int(bool(affected)), "affected_services": sorted(affected)}


def placebo_rate(root: Path, window_s: float = 150.0) -> Dict[str, Any]:
    """Label-noise check: run the detector on [inject-2w, inject-w) vs [inject-w, inject)."""
    real = placebo = 0
    cases = find_cases(root)
    for case_dir in cases:
        info = parse_case_name(case_dir, root)
        df, inject = load_case(case_dir)
        real += propagation(df, inject, info["root_service"], window_s)["cascade"]
        placebo += propagation(df, inject - window_s, info["root_service"], window_s)["cascade"]
    return {"window_s": window_s, "cases": len(cases), "fault_windows_flagged": real,
            "fault_free_windows_flagged": placebo}


# ---------------------------------------------------------------- samples


def drift_score(raw_step: np.ndarray, gw_nominal_ms: float) -> float:
    """BACCP sync-drift score from the boundary (gateway -> legacy) as observed in RCAEval.

    RE1 has no per-call completion counts and no gateway queue depth, so the rate and
    queue terms are zero (submission = completion); drift comes only from the gateway's
    latency against its own pre-fault median and its error rate, using the unchanged
    formula in ai-models/graph-builder/features.py.
    """
    from importlib import import_module

    features = import_module("ai-models.graph-builder.features")
    gw = raw_step[3]
    return float(features.compute_sync_drift_score(
        gateway_submission_rate=gw[1], legacy_completion_rate=gw[1], gateway_queue_depth=0.0,
        boundary_latency_ms=gw[0], nominal_latency_ms=max(1.0, gw_nominal_ms), boundary_error_rate=gw[2]))


def snapshots(raw: np.ndarray, t: np.ndarray, start: float, end: float, gw_nominal_ms: float,
              n: int = 5) -> List[Dict[str, Any]]:
    """n snapshots, each the mean over one of n equal buckets of [start, end)."""
    edges = np.linspace(start, end, n + 1)
    out = []
    for i in range(n):
        m = (t >= edges[i]) & (t < edges[i + 1])
        step = np.nanmean(raw[m], axis=0) if m.any() else raw[np.argmin(np.abs(t - edges[i]))]
        step = np.nan_to_num(step)
        out.append({"t": i, "sync_drift_score": drift_score(step, gw_nominal_ms),
                    "node_features": _node_features(step.tolist())})
    return out


def build_samples(root: Path, window_s: float = 300.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    samples, cases, mapping = [], [], None
    for case_dir in find_cases(root):
        info = parse_case_name(case_dir, root)
        df, inject = load_case(case_dir)
        raw, mp = node_timeseries(df)
        mapping = mapping or mp
        t = df["time"].to_numpy(dtype=float)
        pre_gw = raw[(t >= inject - window_s) & (t < inject), 3, 0]
        gw_nominal = float(np.nanmedian(pre_gw)) if pre_gw.size else 15.0
        prop = propagation(df, inject, info["root_service"], window_s)
        root_node = node_of(info["root_service"])
        cases.append({**info, **prop, "root_node": NODES[root_node] if root_node is not None else None})
        common = {"group": info["root_service"], "fault_type": info["fault_type"],
                  "root_cause_node": root_node if root_node is not None else 3,
                  "lead_time_seconds": 0.0}
        samples.append({**common, "sample_id": f"{info['case_id']}#pre", "phase": "pre", "cascade_label": 0,
                        "severity_level": 0,
                        "snapshots": snapshots(raw, t, inject - window_s, inject, gw_nominal)})
        samples.append({**common, "sample_id": f"{info['case_id']}#post", "phase": "post",
                        "cascade_label": prop["cascade"], "severity_level": 3 if prop["cascade"] else 0,
                        "snapshots": snapshots(raw, t, inject, inject + window_s, gw_nominal)})
    return samples, {"cases": cases, **(mapping or {})}


# ---------------------------------------------------------------- evaluation


def _scores(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    pred = (p >= thr).astype(int)
    two = len(np.unique(y)) > 1
    return {"n": int(len(y)), "positives": int(y.sum()),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4) if two else None,
            "pr_auc": round(float(average_precision_score(y, p)), 4) if two else None}


def _views(samples: List[Dict[str, Any]], p: np.ndarray) -> Dict[str, Any]:
    """all = pre (normal) + post windows; post_only = propagating vs contained faults."""
    y = np.array([s["cascade_label"] for s in samples])
    post = np.array([s["phase"] == "post" for s in samples])
    return {"all_windows": _scores(y, p), "post_injection_only": _scores(y[post], p[post])}


def predict_with_weights(samples: List[Dict[str, Any]], weight_path: Path) -> np.ndarray:
    import torch
    sys.path.insert(0, str(AI_MODELS_DIR / "cascade_predictor"))
    from model import HeteroCascadePredictor
    import train as cp_train

    model = HeteroCascadePredictor()
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    model.eval()
    ts = [cp_train.tensorize_sample(s) for s in samples]
    with torch.no_grad():
        out = model.forward_tensors(torch.stack([t.x_seq for t in ts]), torch.stack([t.drift_seq for t in ts]))
    return out["cascade_prob"].numpy()


def _flat(sample: Dict[str, Any]) -> np.ndarray:
    return np.array([v for s in sample["snapshots"] for row in s["node_features"] for v in row[:4]]
                    + [s["sync_drift_score"] / 100.0 for s in sample["snapshots"]])


def grouped_cv(samples: List[Dict[str, Any]], epochs: int, seed: int) -> Dict[str, np.ndarray]:
    """Leave-one-root-service-out CV: train on 4 fault locations, test on the 5th.

    Out-of-fold probabilities for (a) the BACCP model trained from scratch on RCAEval and
    (b) logistic regression on the same flattened features.
    """
    import tempfile
    import torch
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, str(AI_MODELS_DIR / "cascade_predictor"))
    import train as cp_train

    groups = sorted({s["group"] for s in samples})
    p_model, p_lr = np.full(len(samples), np.nan), np.full(len(samples), np.nan)
    orig_loader = cp_train.load_and_split_data
    for g in groups:
        te = [i for i, s in enumerate(samples) if s["group"] == g]
        rest = [i for i, s in enumerate(samples) if s["group"] != g]
        rest_groups = sorted({samples[i]["group"] for i in rest})
        if len(rest_groups) >= 3:  # hold one training location out for early stopping
            val_group = np.random.RandomState(seed).permutation(rest_groups)[0]
            va = [i for i in rest if samples[i]["group"] == val_group]
            tr = [i for i in rest if samples[i]["group"] != val_group]
        else:  # tiny fixture: no spare location, validate on the training cases
            tr = va = rest
        parts = [[cp_train.tensorize_sample(samples[i]) for i in idx] for idx in (tr, va, te)]
        cp_train.load_and_split_data = lambda *_a, _p=parts, **_k: (*_p, {
            "dataset_hash": "rcaeval-re1-tt", "total_samples": len(samples), "train_samples": len(_p[0]),
            "val_samples": len(_p[1]), "test_samples": len(_p[2]), "feature_metadata": {}})
        with tempfile.TemporaryDirectory() as tmp:
            model, _ = cp_train.train_model(str(Path(tmp) / "unused.json"), epochs=epochs, seed=seed, patience=15,
                                            output_weight_path=str(Path(tmp) / "w.pt"),
                                            output_meta_path=str(Path(tmp) / "m.json"))
        model.eval()
        with torch.no_grad():
            b = parts[2]
            out = model.forward_tensors(torch.stack([s.x_seq for s in b]), torch.stack([s.drift_seq for s in b]))
        p_model[te] = out["cascade_prob"].numpy()

        X_tr = np.stack([_flat(samples[i]) for i in rest])
        y_tr = np.array([samples[i]["cascade_label"] for i in rest])
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
        if len(np.unique(y_tr)) < 2:
            p_lr[te] = float(y_tr.mean())
        else:
            lr = LogisticRegression(max_iter=5000, C=0.5).fit((X_tr - mu) / sd, y_tr)
            p_lr[te] = lr.predict_proba((np.stack([_flat(samples[i]) for i in te]) - mu) / sd)[:, 1]
    cp_train.load_and_split_data = orig_loader
    return {"model": p_model, "lr": p_lr}


def evaluate(samples: List[Dict[str, Any]], meta: Dict[str, Any], epochs: int = 60, seed: int = 42,
             train_cv: bool = True, root: Optional[Path] = None) -> Dict[str, Any]:
    drift_max = np.array([max(sn["sync_drift_score"] for sn in s["snapshots"]) for s in samples])
    zero_shot = {}
    for mode in ("full", "early"):
        w = WEIGHTS / f"trace_{mode}" / "cascade_predictor_tier1a.pt"
        if w.exists():
            zero_shot[f"baccp_hetero_rgcn trace_{mode} (zero-shot)"] = _views(samples, predict_with_weights(samples, w))
    result: Dict[str, Any] = {
        "zero_shot": zero_shot,
        "drift_rule_eps_ge_45": _views(samples, (drift_max >= 45.0).astype(float)),
        "drift_score_max (ranking)": _views(samples, drift_max / 100.0),
    }
    if train_cv:
        cv = grouped_cv(samples, epochs, seed)
        result["trained_on_rcaeval (leave-one-root-service-out CV)"] = {
            "baccp_hetero_rgcn": _views(samples, cv["model"]),
            "logistic_regression_flat": _views(samples, cv["lr"]),
        }
    cases = meta["cases"]
    by_fault = {f: {"cases": sum(c["fault_type"] == f for c in cases),
                    "cascades": sum(c["cascade"] for c in cases if c["fault_type"] == f)} for f in FAULT_TYPES}
    return {
        "dataset": "RCAEval RE1-TT (Train-Ticket, metrics only)",
        "source": "https://zenodo.org/records/14590730 (concept record 14504481), arXiv 2412.17015, CC-BY 4.0",
        "n_cases": len(cases), "n_samples": len(samples),
        "cascade_definition": ("fault propagates beyond the root-cause service: at least one other service's "
                               "p90 latency median over the 5 min after injection is >= 2x its pre-injection "
                               "median and >= 8 robust z-scores (MAD) above it; services averaging < 0.5 req/s "
                               "are ignored"),
        "label_noise_check": placebo_rate(root) if root is not None else None,
        "sample_definition": "per case: 5 snapshots over the 5 min before injection (label 0) and 5 over the "
                             "5 min after (label = cascade); each snapshot is the mean of a 1-min bucket",
        "cascade_rate_by_fault_type": by_fault,
        "node_mapping": meta.get("mapping"), "unmapped_services": meta.get("unmapped"),
        "results": result,
        "cases": [{k: c[k] for k in ("case_id", "fault_type", "root_service", "root_node", "cascade",
                                     "affected_services")} for c in cases],
    }


FIXTURE_CASES = ["ts-route-service_delay/1", "ts-train-service_loss/1", "ts-order-service_disk/1"]
FIXTURE_SERVICES = ["ts-ui-dashboard", "ts-order-service", "ts-preserve-service", "ts-seat-service",
                    "ts-travel-service", "ts-travel2-service", "ts-admin-travel-service", "ts-ticketinfo-service",
                    "ts-basic-service", "ts-route-service", "ts-train-service", "ts-auth-service",
                    "ts-food-service", "ts-consign-service", "ts-order-mongo", "ts-route-mongo",
                    "ts-train-mongo", "ts-auth-mongo"]


def make_fixture(root: Path = RAW, out: Path = FIXTURE, step_s: int = 10, window_s: float = 300.0) -> None:
    """Cut a tiny real excerpt for the tests: 3 cases, 18 services, +-5 min, 10 s resolution."""
    for case in FIXTURE_CASES:
        df, inject = load_case(root / case)
        keep = ["time"] + [c for c in df.columns if (split_column(c) or ("", ""))[0] in FIXTURE_SERVICES]
        df = df.loc[(df["time"] >= inject - window_s) & (df["time"] < inject + window_s), keep]
        df = df[((df["time"] - inject) % step_s) == 0]
        dest = out / case
        dest.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest / "simple_data.csv", index=False, float_format="%.6g")
        (dest / "inject_time.txt").write_text(str(int(inject)))
    print(f"[+] fixture written to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=RAW, help="extracted RE1-TT directory")
    ap.add_argument("--out", type=Path, default=RESULTS / "rcaeval_metrics.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-train", action="store_true", help="zero-shot only")
    ap.add_argument("--make-fixture", action="store_true", help="(re)cut the tiny test fixture and exit")
    a = ap.parse_args()
    if a.make_fixture:
        make_fixture(a.root)
        return
    if not find_cases(a.root):
        raise SystemExit(f"no RCAEval cases under {a.root}; run: python ai-models/real_data/download.py rcaeval")
    samples, meta = build_samples(a.root)
    result = evaluate(samples, meta, epochs=a.epochs, seed=a.seed, train_cv=not a.no_train, root=a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("n_cases", "cascade_rate_by_fault_type", "results")}, indent=2))
    print(f"[+] wrote {a.out}")


if __name__ == "__main__":
    main()
