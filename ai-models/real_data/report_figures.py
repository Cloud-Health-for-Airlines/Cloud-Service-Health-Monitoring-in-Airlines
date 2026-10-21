"""Figures + sensitivity analysis for the real-data pipeline.

    python ai-models/real_data/report_figures.py

Writes to results/real_data/:
    real_inputs.png                 - what the real datasets look like
    southwest_dec2022_workload.png  - real cancellations vs generated boundary load
    sensitivity.json / sensitivity.md - how the assumed multipliers change the outcome
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_data.reservation_workload import PROCESSED, WorkloadConfig, generate_workload  # noqa: E402
from real_data.trace_simulator import SimConfig, normal_day_mask  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "real_data"


def fig_real_inputs() -> None:
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    daily = pd.read_csv(PROCESSED / "bts_daily_carrier_summary.csv", parse_dates=["FlightDate"])
    for c, col in [("WN", "#d62728"), ("DL", "#1f77b4"), ("AA", "#7f7f7f"), ("UA", "#2ca02c")]:
        d = daily[(daily["Reporting_Airline"] == c) & (daily["month"] == 202212)].sort_values("FlightDate")
        ax[0].plot(d["FlightDate"].dt.day, 100 * d["cancel_rate"], label=c, color=col)
    ax[0].set(title="BTS: cancellation rate, Dec 2022", xlabel="day of month", ylabel="% flights cancelled")
    ax[0].legend()

    z = np.load(PROCESSED / "azure_functions_shape.npz")
    ax[1].plot(np.arange(1440) / 60, z["diurnal_profile"], color="#9467bd")
    ax[1].set(title="Azure Functions 2019: diurnal shape", xlabel="hour (UTC)", ylabel="relative demand")

    ali = json.loads((PROCESSED / "alibaba_utilisation.json").read_text())["cpu"]
    fl = np.load(PROCESSED / "alibaba_cpu_fluctuations.npy")
    ax[2].hist(np.clip(fl * ali["mean"], 0, 100), bins=60, color="#ff7f0e")
    ax[2].axvline(ali["p50"], ls="--", color="k", label=f"median {ali['p50']:.0f}%")
    ax[2].set(title="Alibaba 2022: node CPU utilisation", xlabel="CPU %", ylabel="samples")
    ax[2].legend()

    inc = pd.read_csv(PROCESSED / "gcp_incidents.csv.gz")
    d = np.sort(inc["duration_min"].dropna().clip(lower=1).to_numpy())
    ax[3].plot(d / 60, np.arange(1, len(d) + 1) / len(d), color="#17becf")
    ax[3].set_xscale("log")
    ax[3].set(title=f"Google Cloud: {len(inc)} incidents, duration CDF", xlabel="hours", ylabel="fraction")
    fig.tight_layout()
    fig.savefig(OUT / "real_inputs.png", dpi=130)
    plt.close(fig)


def fig_southwest() -> None:
    wl = generate_workload(WorkloadConfig(carrier="WN", month="202212", seed=42))  # as in the dataset build
    mask = normal_day_mask(wl, "WN")
    cap = SimConfig().legacy_headroom * float(np.quantile(wl["legacy_writes_rps"].to_numpy()[mask], 0.99))
    h = wl.resample("1h").agg({"cancellations_pm": "sum", "legacy_writes_rps": "mean", "crew_rps": "mean"})
    fig, ax1 = plt.subplots(figsize=(14, 4.2))
    ax1.bar(h.index, h["cancellations_pm"], width=1 / 24, color="#d62728", alpha=0.35,
            label="real cancelled departures / hour (BTS)")
    ax1.set_ylabel("cancellations / hour")
    ax2 = ax1.twinx()
    ax2.plot(h.index, h["legacy_writes_rps"], color="#1f77b4", lw=1, label="generated mainframe demand (rps)")
    ax2.axhline(cap, color="k", ls="--", lw=1, label=f"mainframe capacity ({cap:.0f} rps)")
    ax2.set_ylabel("requests / second")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title("Southwest, December 2022: real disruption -> reservation/crew workload at the legacy boundary")
    fig.tight_layout()
    fig.savefig(OUT / "southwest_dec2022_workload.png", dpi=130)
    plt.close(fig)


def sensitivity() -> list:
    rows = []
    for carrier, month in [("WN", "202212"), ("DL", "202407")]:
        for rb in [2, 4, 6, 8, 10]:
            wl = generate_workload(WorkloadConfig(carrier=carrier, month=month, rebooking_calls_per_pax=rb))
            mask = normal_day_mask(wl, carrier)
            dem = wl["legacy_writes_rps"].to_numpy()
            p99 = float(np.quantile(dem[mask], 0.99))
            for head in [1.1, 1.25, 1.5]:
                cap = head * p99
                over = dem > cap
                rows.append({"carrier": carrier, "month": month, "rebooking_calls_per_pax": rb,
                             "legacy_headroom": head,
                             "overload_min_disruption_days": int(over[~mask].sum()),
                             "overload_min_normal_days": int(over[mask].sum()),
                             "mean_demand_disruption_vs_normal": round(float(dem[~mask].mean() / dem[mask].mean()), 2)
                             if (~mask).any() else None})
    (OUT / "sensitivity.json").write_text(json.dumps(rows, indent=2))
    df = pd.DataFrame(rows)
    heads = sorted(df["legacy_headroom"].unique())
    md = ["# Sensitivity of demand-only overload to assumed multipliers", "",
          "Minutes during each airline's *real* disruption window in which generated mainframe demand",
          "exceeds capacity with **no fault injected** (capacity = headroom x P99 of normal-day demand).", "",
          "| carrier | rebooking calls / pax | demand ratio (disruption / normal) | "
          + " | ".join(f"overload min @ headroom {h}" for h in heads) + " |",
          "|---|---|---|" + "---|" * len(heads)]
    for (c, rb), g in df.groupby(["carrier", "rebooking_calls_per_pax"]):
        vals = [str(int(g[g["legacy_headroom"] == h]["overload_min_disruption_days"].iloc[0])) for h in heads]
        md.append(f"| {c} | {rb} | {g['mean_demand_disruption_vs_normal'].iloc[0]} | " + " | ".join(vals) + " |")
    (OUT / "sensitivity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_real_inputs()
    fig_southwest()
    sensitivity()
    print((OUT / "sensitivity.md").read_text(encoding="utf-8"))
