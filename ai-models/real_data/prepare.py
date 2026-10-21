"""Turn the raw downloads into small, committable, real-data artifacts.

Raw files (hundreds of MB) stay in ``data/raw/`` (git-ignored). This script
extracts exactly what the workload generator and simulator need and writes it
to ``ai-models/data/real_processed/`` (a few MB), so teammates and CI can run
the pipeline without re-downloading anything.

    python ai-models/real_data/prepare.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from real_data import airline_ops, cloud_health, cloud_traces  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
OUT = REPO_ROOT / "ai-models" / "data" / "real_processed"

DEFAULT_CARRIERS = ["WN", "DL", "AA", "UA", "AS"]


def _log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def prepare_bts(carriers: list[str]) -> dict:
    zips = sorted((RAW / "bts_ontime").glob("*.zip"))
    if not zips:
        _log("no BTS on-time files - skipped")
        return {}
    info = {"source_files": [z.name for z in zips], "series": []}
    dailies, windows = [], []
    for z in zips:
        ontime = airline_ops.load_ontime([z])
        ym = ontime["FlightDate"].dt.strftime("%Y%m").mode()[0]
        daily = airline_ops.daily_carrier_summary(ontime)
        daily["month"] = ym
        dailies.append(daily)
        for w in airline_ops.detect_disruption_windows(daily):
            windows.append({**w.__dict__, "start": str(w.start.date()), "end": str(w.end.date()), "month": ym})
        for c in carriers:
            s = airline_ops.schedule_series(ontime, "1min", carrier=c)
            if s["departures"].sum() == 0:
                continue
            path = OUT / f"bts_schedule_{c}_{ym}_1min.csv.gz"
            s.astype("float32").to_csv(path, compression="gzip", float_format="%.0f")
            info["series"].append({"carrier": c, "month": ym, "file": path.name,
                                   "flights": int(s["departures"].sum()),
                                   "cancellations": int(s["cancellations"].sum())})
            _log(f"{c} {ym}: {int(s['departures'].sum())} flights, {int(s['cancellations'].sum())} cancelled")
        info.setdefault("flights_per_month", {})[ym] = (
            ontime.groupby("Reporting_Airline").size().astype(int).to_dict())
    pd.concat(dailies).to_csv(OUT / "bts_daily_carrier_summary.csv", index=False)
    (OUT / "bts_disruption_windows.json").write_text(json.dumps(windows, indent=2, default=str))
    _log(f"{len(windows)} real disruption windows detected")
    return info


def prepare_db1b(bts_info: dict) -> dict:
    zips = sorted((RAW / "bts_db1b_market").glob("*.zip"))
    if not zips:
        _log("no DB1B files - skipped")
        return {}
    db1b = airline_ops.load_db1b_market(zips[0])
    year, quarter = int(db1b["Year"].iloc[0]), int(db1b["Quarter"].iloc[0])
    pax = airline_ops.passengers_per_carrier(db1b)
    out = {"source_file": zips[0].name, "year": year, "quarter": quarter,
           "sample_rate": airline_ops.DB1B_SAMPLE_RATE,
           "passenger_trips_per_quarter": {k: int(v) for k, v in pax.head(25).items()},
           "passengers_per_departure": {}, "top_markets": {}}
    # Match to an on-time month inside the same quarter, if we have one.
    months_in_q = {f"{year}{m:02d}" for m in range(3 * quarter - 2, 3 * quarter + 1)}
    for ym, per_carrier in (bts_info.get("flights_per_month") or {}).items():
        if ym in months_in_q:
            for c, flights in per_carrier.items():
                if c in pax.index:
                    out["passengers_per_departure"][c] = round(float(pax[c]) / (flights * 3.0), 1)
            out["matched_ontime_month"] = ym
    for c in DEFAULT_CARRIERS:
        out["top_markets"][c] = airline_ops.top_markets(db1b, c, 10).to_dict(orient="records")
    (OUT / "db1b_summary.json").write_text(json.dumps(out, indent=2))
    _log(f"DB1B {year}Q{quarter}: passengers/departure {out['passengers_per_departure']}")
    return out


def prepare_azure(days=(1, 2, 3, 4, 5, 6, 7), n_blocks: int = 60) -> dict:
    tars = sorted((RAW / "azure_functions_2019").glob("*.tar*"))
    if not tars:
        _log("no Azure Functions trace - skipped")
        return {}
    inv = cloud_traces.load_azure_invocations(tars[0], days=list(days), trigger="http", top_n=400)
    prof = cloud_traces.diurnal_profile(inv)
    blocks = cloud_traces.residual_blocks(inv)
    rng = np.random.RandomState(0)
    cols = rng.choice(blocks.shape[1], size=min(n_blocks, blocks.shape[1]), replace=False)
    np.savez_compressed(OUT / "azure_functions_shape.npz",
                        diurnal_profile=prof.astype(np.float32),
                        residual_blocks=blocks[:, cols].astype(np.float32))
    stats = {"days": list(days), "functions_used": int(inv.shape[1]), "minutes": int(inv.shape[0]),
             "human_driven_functions": int(len(cloud_traces.human_driven_functions(inv))),
             "peak_to_trough": round(float(prof.max() / prof.min()), 2),
             "residual_cv": round(float(np.std(blocks[:, cols]) / np.mean(blocks[:, cols])), 3)}
    _log(f"Azure: {stats}")
    return stats


def prepare_alibaba() -> dict:
    tars = sorted((RAW / "alibaba_ms_2022_node").glob("*.tar.gz"))
    if not tars:
        _log("no Alibaba NodeMetrics - skipped")
        return {}
    nm = cloud_traces.load_alibaba_node_metrics(tars[0], max_nodes=1000)
    cpu = cloud_traces.utilisation_stats(nm, "cpu_utilization")
    mem = cloud_traces.utilisation_stats(nm, "memory_utilization")
    fl = cloud_traces.utilisation_fluctuations(nm)
    rng = np.random.RandomState(0)
    np.save(OUT / "alibaba_cpu_fluctuations.npy", rng.choice(fl, size=min(20000, len(fl))).astype(np.float32))
    out = {"source_file": tars[0].name, "cpu": cpu, "memory": mem}
    (OUT / "alibaba_utilisation.json").write_text(json.dumps(out, indent=2))
    _log(f"Alibaba CPU: {cpu}")
    return out


def prepare_gcp() -> dict:
    srcs = sorted((RAW / "gcp_incidents").glob("*.json")) + sorted((RAW / "gcp_incidents_archive").glob("*.json"))
    if not srcs:
        _log("no GCP incidents - skipped")
        return {}
    inc = cloud_health.load_gcp_incidents(*srcs)
    inc.drop(columns=["products"]).assign(products=inc["products"].str.join("; "))\
        .to_csv(OUT / "gcp_incidents.csv.gz", index=False, compression="gzip")
    s = cloud_health.summary(inc)
    d = cloud_health.incident_distributions(inc)
    s["fault_class_share"] = {k: round(v, 3) for k, v in d["fault_class_share"].items()}
    s["severity_share"] = {k: round(v, 3) for k, v in d["severity_share"].items()}
    _log(f"GCP incidents: {s}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carriers", nargs="*", default=DEFAULT_CARRIERS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    bts = prepare_bts(args.carriers)
    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "provenance": "derived exclusively from real public datasets; see ai-models/real_data/catalog.py",
        "bts_ontime": bts,
        "bts_db1b": prepare_db1b(bts),
        "azure_functions_2019": prepare_azure(),
        "alibaba_ms_2022_node": prepare_alibaba(),
        "gcp_incidents": prepare_gcp(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    _log(f"wrote {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
