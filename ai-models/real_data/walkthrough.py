"""Live, step-by-step walkthrough of the real-data pipeline (for reviews / demos).

Prints what happens at every stage with real numbers, so the audience sees the
inside of the project instead of a dashboard.

    python ai-models/real_data/walkthrough.py            # run straight through
    python ai-models/real_data/walkthrough.py --pause    # wait for Enter between steps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_data.airline_ops import detect_disruption_windows  # noqa: E402
from real_data.reservation_workload import PROCESSED, WorkloadConfig, generate_workload  # noqa: E402
from real_data.trace_simulator import SimConfig, load_alibaba, normal_day_mask  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PAUSE = False


def step(n: int, title: str, why: str) -> None:
    if PAUSE and n > 1:
        input("\n   [Enter] for next step ...")
    print("\n" + "=" * 78)
    print(f" STEP {n}: {title}")
    print("=" * 78)
    print(f" Why: {why}\n")


def main() -> None:
    m = json.loads((PROCESSED / "manifest.json").read_text())

    step(1, "Real flights - BTS On-Time Performance (U.S. DOT)",
         "tells us WHEN airline activity happens and when the operation actually broke.")
    for s in m["bts_ontime"]["series"]:
        if s["carrier"] in ("WN", "DL"):
            print(f"   {s['carrier']} {s['month']}: {s['flights']:>7,} real flights, {s['cancellations']:>6,} cancelled")
    daily = pd.read_csv(PROCESSED / "bts_daily_carrier_summary.csv", parse_dates=["FlightDate"])
    print("\n   Disruptions found automatically (cancel-rate anomaly, no hand labels):")
    for w in detect_disruption_windows(daily)[:4]:
        print(f"   - {w.carrier}: {w.start.date()} .. {w.end.date()}  peak {w.peak_cancel_rate:5.1%} cancelled, "
              f"{w.cancelled_flights:,} flights  (normal {w.baseline_cancel_rate:.2%})")
    print("   -> WN Dec-2022 = Southwest meltdown (SkySolver crew tool), DL Jul-2024 = CrowdStrike outage.")

    step(2, "Real bookings - BTS DB1B (10% sample of real tickets)",
         "tells us HOW MANY passengers each flight generates for the reservation system.")
    d = m["bts_db1b"]
    print(f"   Quarter {d['year']}Q{d['quarter']}; Southwest passenger journeys/quarter: "
          f"{d['passenger_trips_per_quarter']['WN']:,}")
    for c in ("WN", "DL", "AA", "UA"):
        print(f"   {c}: {d['passengers_per_departure'][c]:6.1f} booked journeys per departure")

    step(3, "Real cloud traffic shape - Azure Functions 2019 (Microsoft)",
         "real users do not arrive at a constant rate; this gives the day/night cycle and burstiness.")
    a = m["azure_functions_2019"]
    z = np.load(PROCESSED / "azure_functions_shape.npz")
    prof = z["diurnal_profile"].reshape(24, 60).mean(1)
    print(f"   {a['functions_used']} busiest HTTP functions, {a['human_driven_functions']} with a human day/night cycle")
    print(f"   peak/trough = {a['peak_to_trough']}x ; busiest hour {prof.argmax():02d}:00 UTC, quietest {prof.argmin():02d}:00 UTC")

    step(4, "Real cluster behaviour - Alibaba microservices trace 2022",
         "shows how hard a real production cloud runs its machines -> sets our autoscaler target.")
    c = m["alibaba_ms_2022_node"]["cpu"]
    tu, _ = load_alibaba()
    print(f"   {c['n_nodes']} nodes, {c['n_samples']:,} one-minute samples")
    print(f"   CPU median {c['p50']:.1f}%  p90 {c['p90']:.1f}%  p99 {c['p99']:.1f}%")
    print(f"   -> cloud has big headroom; autoscaler target utilisation = {tu:.2f}")
    print("   -> the mainframe behind it does NOT autoscale. That mismatch is our research gap.")

    step(5, "Real cloud failures - Google Cloud Service Health incident history",
         "gives realistic fault TYPES and DURATIONS instead of invented ones.")
    g = m["gcp_incidents"]
    print(f"   {g['n_incidents']} incidents over {g['span_days']:.0f} days ({g['incidents_per_30d']} / 30 days)")
    print(f"   duration median {g['duration_median_min']/60:.1f} h, p90 {g['duration_p90_min']/60:.1f} h")
    print(f"   fault mix {g['fault_class_share']}")

    step(6, "Turning real data into reservation workload (one real hour)",
         "requests/second per service = real flights x real passengers x per-passenger multipliers.")
    wl = generate_workload(WorkloadConfig(carrier="WN", month="202212", seed=42))  # same seed as the dataset build
    cfg = wl.attrs["config"]
    print(f"   Assumptions: look-to-book {cfg['look_to_book']:.0f}:1, rebooking {cfg['rebooking_calls_per_pax']:.0f} "
          f"mainframe calls/passenger, {cfg['crew_per_flight']:.0f} crew/flight")
    cols = ["scheduled_departures_pm", "cancellations_pm", "search_rps", "booking_rps", "checkin_rps",
            "rebooking_rps", "crew_rps", "legacy_writes_rps"]
    for label, day in [("normal day ", "2022-12-14"), ("meltdown day", "2022-12-26")]:
        h = wl.loc[f"{day} 14:00":f"{day} 14:59", cols]
        print(f"   {label} {day} 14:00-15:00: flights {h.scheduled_departures_pm.sum():.0f} "
              f"(cancelled {h.cancellations_pm.sum():.0f}) | search {h.search_rps.mean():6.0f} rps | "
              f"booking {h.booking_rps.mean():4.1f} | check-in {h.checkin_rps.mean():4.1f} | "
              f"rebooking {h.rebooking_rps.mean():4.1f} | crew {h.crew_rps.mean():4.1f} | "
              f"MAINFRAME {h.legacy_writes_rps.mean():5.1f} rps")
    mask = normal_day_mask(wl, "WN")
    cap = SimConfig().legacy_headroom * float(np.quantile(wl["legacy_writes_rps"].to_numpy()[mask], 0.99))
    man = REPO / "ai-models" / "data" / "dataset_trace" / "manifest.json"
    if man.exists():  # the simulator sizes it on the (smoother) 10-second grid
        cap = next((s["legacy_capacity_rps"] for s in json.loads(man.read_text())["sources"]
                    if s["source"] == "WN:202212"), cap)
    print(f"   Mainframe capacity = 1.25 x P99 of normal-day demand = {cap:.1f} rps")
    print(f"   Sustained mainframe load, meltdown vs normal days: "
          f"{wl['legacy_writes_rps'][~mask].mean() / wl['legacy_writes_rps'][mask].mean():.2f}x")

    step(7, "Simulating the hybrid stack - one gradual fault, every 30 seconds",
         "shows the mechanism and the warning window: the mainframe path slows, then chokes.")
    tl_path = REPO / "ai-models" / "data" / "dataset_trace" / "timeline_WN_202212.csv.gz"
    if not tl_path.exists():
        print("   (run build_trace_dataset.py first to produce the timeline)")
    else:
        t = pd.read_csv(tl_path, index_col=0, parse_dates=True)
        t["fault_type"] = t["fault_type"].fillna("")
        onsets = t.index[(t["fault_type"] != "").astype(int).diff() == 1]
        pick, k_min = None, None
        for o in onsets:
            after = t.loc[o: o + pd.Timedelta(minutes=20)]
            if t.at[o, "fault_type"] != "network-delay" or after["cascade_state"].iloc[0] or not after["cascade_state"].any():
                continue
            k = int(after["cascade_state"].to_numpy().argmax())  # 10-second steps
            if 18 <= k <= 72 and t.at[o, "fault_intensity"] < 0.2:
                pick, k_min = o, k / 6
                break
        o = pick if pick is not None else onsets[0]
        w = t.loc[o - pd.Timedelta(minutes=1): o + pd.Timedelta(minutes=(k_min or 5) + 3)].iloc[::3]
        print(f"   Fault: {t.at[o, 'fault_type']} ({t.at[o, 'fault_level']}) starting {o:%Y-%m-%d %H:%M:%S}, "
              f"worsening gradually; cascade after {k_min:.1f} min" if k_min else "")
        print("   time      fault%  demand  capacity  queue  gw_lat_ms  gw_err  drift%  res_lat_ms  CASCADE")
        for ts, r in w.iterrows():
            print(f"   {ts:%H:%M:%S}  {100 * r.fault_intensity:5.0f}  {r.legacy_writes_rps:6.1f}  "
                  f"{r.legacy_capacity:8.1f}  {r.gw_queue:5.0f}  {r.gw_latency_ms:9.0f}  {r.gw_error:6.2f}  "
                  f"{r.drift:6.1f}  {r.res_latency_ms:10.0f}  {'<<<' if r.cascade_state else ''}")
        print("   Read it: as the mainframe delay grows, gateway latency and drift creep up for minutes")
        print("   (the warning window). The cascade starts when a booking waits >150 ms end-to-end while the")
        print("   boundary is >100 ms. Capacity (= connection pool / latency, Little's law) keeps falling")
        print("   toward demand; once below it the 30 s gateway buffer fills and calls time out.")
        print("   The cloud side did nothing wrong.")
        print("   Abrupt faults (~30%, e.g. hardware failure) give no warning - only fast detection helps.")

    step(8, "Results - trained on Southwest Dec-2022, tested on Delta Jul-2024 (never seen)",
         "an honest test on a different airline and a different real event.")
    for mode, label in [("early", "EARLY WARNING (2 min before)"), ("full", "DETECTION (whole window)")]:
        p = REPO / "results" / "real_data" / f"cascade_metrics_{mode}.json"
        if p.exists():
            r = json.loads(p.read_text())
            print(f"   {label}")
            for name, v in r["models"].items():
                print(f"     {name:42s} AUC {v['roc_auc']:.3f}   F1 {v['f1']:.3f}")
    p = REPO / "results" / "real_data" / "rcaeval_metrics.json"
    if p.exists():
        r = json.loads(p.read_text())["results"]
        zs = r["zero_shot"]["baccp_hetero_rgcn trace_full (zero-shot)"]["all_windows"]
        print(f"   EXTERNAL CHECK on real Train-Ticket telemetry (RCAEval, 125 faults), detection model "
              f"zero-shot: AUC {zs['roc_auc']:.3f}")
    print("\n   Synthetic data gave ~0.96 AUC. Real data is harder - that gap is WHY real data matters.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", action="store_true", help="wait for Enter between steps")
    PAUSE = ap.parse_args().pause
    main()
