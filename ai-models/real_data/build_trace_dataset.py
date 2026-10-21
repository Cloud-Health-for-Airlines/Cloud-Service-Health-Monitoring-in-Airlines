"""Build the real-data-driven (trace-driven) BACCP training dataset.

    python ai-models/real_data/build_trace_dataset.py                 # WN Dec-2022 + DL Jul-2024
    python ai-models/real_data/build_trace_dataset.py --sources WN:202212 --episodes 300

Output (same format as ai-models/data/generate_dataset.py, so the existing
training / evaluation scripts work unchanged):

    ai-models/data/dataset_trace/baccp_dataset.pt
    ai-models/data/dataset_trace/manifest.json
    ai-models/data/dataset_trace/dataset_report.md
    ai-models/data/dataset_trace/timeline_<carrier>_<month>.csv.gz   (minute-level, for plots)

Split strategy ``event`` (default with >1 source): the *last* source is held out
entirely as the test set, i.e. the model is tested on a different airline and a
different real-world disruption than it was trained on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_data.reservation_workload import PROCESSED, WorkloadConfig, generate_workload  # noqa: E402
from real_data.trace_simulator import (  # noqa: E402
    SimConfig, cut_episodes, load_alibaba, load_incident_distributions, normal_day_mask,
    sample_faults, simulate_timeline, upsample_workload,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "dataset_trace"


def build(sources: List[str], episodes_per_source: int, split: str, seed: int,
          look_to_book: float, out_dir: Path = OUT_DIR, processed: Path = PROCESSED) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    cfg = SimConfig(seed=seed)
    target_util, cpu_fluct = load_alibaba(processed)
    inc = load_incident_distributions(processed)
    all_eps: List[Dict[str, Any]] = []
    source_meta = []

    for src in sources:
        carrier, month = src.split(":")
        wl = generate_workload(WorkloadConfig(carrier=carrier, month=month, look_to_book=look_to_book,
                                              seed=seed), processed)
        faults = sample_faults(len(wl), inc, cfg, rng)
        wl = upsample_workload(wl, cfg.step_sec)
        spm = cfg.steps_per_min
        mask = normal_day_mask(wl, carrier, processed)
        sim = simulate_timeline(wl, faults, cfg, target_util, cpu_fluct, mask, rng)
        tl = wl[["scheduled_departures_pm", "cancellations_pm", "legacy_writes_rps"]].join(
            sim[["legacy_capacity", "gw_queue", "gw_latency_ms", "gw_error", "drift", "res_error",
                 "res_latency_ms", "fault_type", "fault_level", "fault_intensity", "overloaded",
                 "cloud_saturation", "cascade_state"]])
        tl.to_csv(out_dir / f"timeline_{carrier}_{month}.csv.gz", compression="gzip", float_format="%.4g")
        eps = cut_episodes(wl, sim, cfg, episodes_per_source, rng, tag=src)
        all_eps.extend(eps)
        source_meta.append({
            "source": src, "minutes": len(wl) // spm, "step_sec": cfg.step_sec,
            "faults_injected": len(faults),
            "abrupt_faults": int(sum(f["ramp_min"] == 0 for f in faults)),
            "legacy_capacity_rps": round(sim.attrs["legacy_capacity_nominal"], 2),
            "pax_per_departure": wl.attrs["pax_per_departure"],
            "pax_per_departure_source": wl.attrs["pax_per_departure_source"],
            "cloud_shape_source": wl.attrs["cloud_shape_source"],
            "minutes_in_cascade_state": int(sim["cascade_state"].sum() // spm),
            "minutes_cloud_saturation_only": int(sim["cloud_saturation"].sum() // spm),
            "minutes_overloaded_without_fault": int(((sim["overloaded"] == 1) & (sim["fault_type"] == "")).sum() // spm),
            "episodes": len(eps), "cascade_episodes": int(sum(e["cascade_occurred"] for e in eps)),
        })
        print(f"[build] {src}: {len(eps)} episodes, {source_meta[-1]['cascade_episodes']} cascades, "
              f"{len(faults)} faults, legacy cap {source_meta[-1]['legacy_capacity_rps']} rps")

    for i, e in enumerate(all_eps):
        e["run_id"] = i

    ids_by_src: Dict[str, List[int]] = {}
    for e in all_eps:
        ids_by_src.setdefault(e["source"], []).append(e["run_id"])
    if split == "event" and len(sources) > 1:
        test_ids = ids_by_src[sources[-1]]
        pool = [i for s in sources[:-1] for i in ids_by_src[s]]
        rng.shuffle(pool)
        cut = int(0.8 * len(pool))
        train_ids, val_ids = sorted(pool[:cut]), sorted(pool[cut:])
    else:  # temporal: order by real time inside each source
        ordered = sorted(all_eps, key=lambda e: (e["source"], e["real_window_start"]))
        n = len(ordered)
        train_ids = sorted(e["run_id"] for e in ordered[: int(0.6 * n)])
        val_ids = sorted(e["run_id"] for e in ordered[int(0.6 * n): int(0.8 * n)])
        test_ids = sorted(e["run_id"] for e in ordered[int(0.8 * n):])

    casc = [e for e in all_eps if e["cascade_occurred"]]
    leads = [e["lead_time_seconds"] for e in casc]
    breakdown: Dict[str, Dict[str, int]] = {}
    for e in all_eps:
        b = breakdown.setdefault(e["fault_type"], {"total": 0, "cascades": 0})
        b["total"] += 1
        b["cascades"] += e["cascade_occurred"]
    drivers: Dict[str, int] = {}
    for e in casc:
        drivers[e["driver"]] = drivers.get(e["driver"], 0) + 1

    manifest = {
        "dataset_version": "baccp-trace-v2.0",
        "generated_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generation_mode": "trace-driven",
        "is_synthetic_fallback": False,
        "data_provenance": {
            "workload": "BTS On-Time + BTS DB1B + Azure Functions 2019 (real)",
            "fault_durations_and_types": inc["source"],
            "cloud_utilisation": "Alibaba microservices v2022 NodeMetrics" if (processed / "alibaba_utilisation.json").exists() else "default",
            "services": "simulated (queueing model) - see real_data/trace_simulator.py",
        },
        "regeneration_command": "python ai-models/real_data/build_trace_dataset.py --sources " + " ".join(sources),
        "sources": source_meta,
        "autoscaler_target_utilisation": round(target_util, 3),
        "simulation": {"step_sec": cfg.step_sec, "abrupt_fault_share": cfg.abrupt_fault_share,
                       "fault_ramp_min": list(cfg.fault_ramp_min),
                       "cascade_persist_sec": cfg.cascade_persist_sec},
        "total_runs": len(all_eps),
        "class_balance": {"cascade_events": len(casc), "nominal_events": len(all_eps) - len(casc),
                          "cascade_percentage": round(100 * len(casc) / max(1, len(all_eps)), 2)},
        "cascade_drivers": drivers,
        "lead_time_seconds": {
            "mean": round(float(np.mean(leads)), 2) if leads else 0.0,
            "median": round(float(np.median(leads)), 2) if leads else 0.0,
            "min": round(float(np.min(leads)), 2) if leads else 0.0,
            "max": round(float(np.max(leads)), 2) if leads else 0.0,
        },
        "fault_breakdown": breakdown,
        "split_strategy": split if len(sources) > 1 else "temporal",
        "splits": {"train_run_ids": train_ids, "val_run_ids": val_ids, "test_run_ids": test_ids},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    torch.save({"manifest": manifest, "runs": all_eps}, out_dir / "baccp_dataset.pt")
    write_report(manifest, out_dir / "dataset_report.md")
    print(f"[build] wrote {out_dir / 'baccp_dataset.pt'}")
    return manifest


def write_report(m: Dict[str, Any], path: Path) -> None:
    lines = [
        "# BACCP Trace-Driven Dataset (real workload)", "",
        f"**Generated:** {m['generated_timestamp']}  ", f"**Command:** `{m['regeneration_command']}`", "",
        "## Provenance", "",
        "| Component | Source |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in m["data_provenance"].items()], "",
        "## Sources", "",
        "| Source | Minutes | Faults | Legacy cap (rps) | Pax/flight (source) | Overloaded min w/o fault | Episodes | Cascades |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in m["sources"]:
        lines.append(f"| {s['source']} | {s['minutes']} | {s['faults_injected']} | {s['legacy_capacity_rps']} | "
                     f"{s['pax_per_departure']:.1f} ({s['pax_per_departure_source']}) | "
                     f"{s['minutes_overloaded_without_fault']} | {s['episodes']} | {s['cascade_episodes']} |")
    lines += ["", "## Labels", "",
              f"- Episodes: **{m['total_runs']}**, cascades **{m['class_balance']['cascade_events']}** "
              f"({m['class_balance']['cascade_percentage']}%)",
              f"- Cascade drivers: {m['cascade_drivers']}",
              f"- Lead time (s): {m['lead_time_seconds']}",
              f"- Split: `{m['split_strategy']}` -> train {len(m['splits']['train_run_ids'])}, "
              f"val {len(m['splits']['val_run_ids'])}, test {len(m['splits']['test_run_ids'])}", "",
              "| Fault type | Episodes | Cascades |", "|---|---|---|"]
    for k, v in m["fault_breakdown"].items():
        lines.append(f"| {k} | {v['total']} | {v['cascades']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["WN:202212", "DL:202407"])
    ap.add_argument("--episodes", type=int, default=500, help="episodes per source")
    ap.add_argument("--split", choices=["event", "temporal"], default="event")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--look-to-book", type=float, default=1000.0)
    a = ap.parse_args()
    build(a.sources, a.episodes, a.split, a.seed, a.look_to_book)


if __name__ == "__main__":
    main()
