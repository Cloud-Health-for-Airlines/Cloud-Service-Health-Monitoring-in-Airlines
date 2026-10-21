"""Generate airline reservation-system workload from real data.

This answers the reviewer's question *"how do you generate reservation workload
and how does the cloud manage it?"*. Every request stream is derived from a
real dataset; the only modelling choices are the per-passenger request
multipliers in :class:`WorkloadConfig`, each of which is documented with its
source or flagged as an assumption so it can be varied in sensitivity runs.

    real flights (BTS On-Time, per minute)  ─┐
    real passengers / flight (BTS DB1B)     ─┼─► passenger events ─► per-service requests/s
    real cloud traffic shape (Azure 2019)   ─┘        (booking, check-in, baggage,
                                                       crew, rebooking, search)
    real disruptions (BTS cancellations / delays) ─► rebooking + crew re-pairing surges

Request streams and where they are served
-----------------------------------------
search         reservations (cloud cache)      look-to-book x bookings
booking        reservations  -> legacy core    PNR create / price / ticket
check-in       reservations  -> legacy core    24h .. 40min before departure
rebooking      reservations  -> legacy core    passengers of cancelled flights
baggage        baggage       -> (10% legacy)   bag check / sort / load events
crew           crew          -> legacy core    roster checks + re-pairing on disruption
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "real_processed"


@dataclass
class WorkloadConfig:
    carrier: str = "WN"
    month: Optional[str] = None  # "YYYYMM"; first available if None
    # Real-data derived (filled from DB1B when available).
    pax_per_departure: Optional[float] = None
    fallback_pax_per_departure: float = 120.0  # ASSUMPTION used only if DB1B is missing
    # Distribution multipliers -------------------------------------------------
    # Look-to-book: Amadeus reported ~1000:1 (PhocusWire 2015); OAG now reports 10k+:1.
    look_to_book: float = 1000.0
    legacy_calls_per_booking: float = 3.0      # ASSUMPTION: create PNR, price, ticket
    online_checkin_share: float = 0.6          # ASSUMPTION
    legacy_calls_per_checkin: float = 2.0      # ASSUMPTION: seat + boarding pass
    bags_per_pax: float = 0.7                  # ASSUMPTION
    events_per_bag: float = 3.0                # check, sort, load
    baggage_legacy_share: float = 0.1          # ASSUMPTION: reconciliation messages
    crew_per_flight: float = 5.0               # 2 pilots + ~3 flight attendants (FAA minimums)
    crew_queries_per_member: float = 2.0       # ASSUMPTION: roster/legality checks per flight
    # Crew pairings span several legs, so one disrupted flight forces re-planning of the
    # crew's downstream legs too (the load that overwhelmed Southwest's SkySolver in 2022).
    crew_ops_per_disrupted_member: float = 12.0  # ASSUMPTION: 4 ops x ~3 downstream legs
    # Re-accommodation needs *live* inventory (cached availability is stale during a
    # disruption), PNR modify, ticket reissue, APIS + notification -> legacy-heavy.
    rebooking_calls_per_pax: float = 6.0       # ASSUMPTION (sensitivity-tested 2..10)
    rebooking_searches_per_pax: float = 20.0   # ASSUMPTION: cached availability lookups
    # Burstiness: average of k real Azure function residual series.
    burst_k: int = 16  # aggregate airline traffic = many users -> smoother than one function
    seed: int = 0
    extra: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------- helpers

def lead_convolve(x: np.ndarray, lags: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """y[t] = sum_i weights[i] * x[t + lags[i]]  for contiguous integer ``lags``.

    ``lags`` are minutes *before* the event (positive = earlier); e.g. check-in
    happening 120 minutes before departure uses lag +120.
    """
    a, b = int(lags[0]), int(lags[-1])
    n, N = b - a + 1, len(x)
    P, B = max(0, -a), max(0, b)
    u = np.concatenate([np.zeros(P), x, np.zeros(B)])
    valid = np.convolve(u, weights[::-1], mode="valid")
    start = a + P
    return valid[start:start + N]


def _window_kernel(lo: int, hi: int, shape: str = "uniform") -> Tuple[np.ndarray, np.ndarray]:
    lags = np.arange(lo, hi + 1)
    if shape == "uniform":
        w = np.ones_like(lags, dtype=float)
    elif shape == "late_peak":  # more activity closer to the event
        w = np.exp(-(lags - lo) / max(1.0, (hi - lo) / 3.0))
    else:
        raise ValueError(shape)
    return lags, w / w.sum()


def load_schedule(carrier: str, month: Optional[str] = None, processed: Path = PROCESSED) -> pd.DataFrame:
    pattern = f"bts_schedule_{carrier}_{month or '*'}_1min.csv.gz"
    files = sorted(processed.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No processed BTS schedule for {carrier} ({pattern}) in {processed}. "
            "Run: python ai-models/real_data/download.py && python ai-models/real_data/prepare.py")
    df = pd.read_csv(files[0], index_col=0, parse_dates=True, compression="gzip")
    df.attrs["source_file"] = files[0].name
    return df


def load_cloud_shape(processed: Path = PROCESSED) -> Optional[Dict[str, np.ndarray]]:
    p = processed / "azure_functions_shape.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {"diurnal_profile": z["diurnal_profile"], "residual_blocks": z["residual_blocks"]}


def pax_per_departure(cfg: WorkloadConfig, processed: Path = PROCESSED) -> Tuple[float, str]:
    if cfg.pax_per_departure:
        return float(cfg.pax_per_departure), "config"
    p = processed / "db1b_summary.json"
    if p.exists():
        d = json.loads(p.read_text())
        v = d.get("passengers_per_departure", {}).get(cfg.carrier)
        if v:
            return float(v), f"BTS DB1B {d['year']}Q{d['quarter']}"
    return cfg.fallback_pax_per_departure, "ASSUMPTION (DB1B unavailable)"


def _burst_series(shape: Optional[Dict[str, np.ndarray]], n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    if shape is None:
        return np.ones(n)
    blocks = shape["residual_blocks"]
    m, f = blocks.shape
    acc = np.zeros(n)
    for _ in range(k):
        col = blocks[:, rng.randint(f)]
        off = rng.randint(m)
        idx = (off + np.arange(n)) % m
        acc += col[idx]
    out = acc / k
    return out / out.mean()


# ---------------------------------------------------------------------- generator

def generate_workload(cfg: WorkloadConfig, processed: Path = PROCESSED) -> pd.DataFrame:
    """Per-minute request rates (requests/second) for every airline service.

    Returned frame is indexed by real timestamps of the chosen carrier-month and
    carries provenance in ``df.attrs``.
    """
    rng = np.random.RandomState(cfg.seed)
    sched = load_schedule(cfg.carrier, cfg.month, processed)
    shape = load_cloud_shape(processed)
    ppd, ppd_src = pax_per_departure(cfg, processed)

    n = len(sched)
    deps = sched["departures"].to_numpy(float)
    cancels = sched["cancellations"].to_numpy(float)
    delayed = sched["delayed"].to_numpy(float)
    flown = deps - cancels

    # --- bookings: real daily passenger volume x real diurnal shape of cloud traffic
    day = sched.index.normalize()
    daily_pax = pd.Series(deps * ppd, index=sched.index).groupby(day).transform("sum").to_numpy()
    minute_of_day = (sched.index.hour * 60 + sched.index.minute).to_numpy()
    if shape is not None:
        prof = shape["diurnal_profile"][minute_of_day]
    else:
        prof = np.ones(n)
    bookings_pm = daily_pax / 1440.0 * prof * _burst_series(shape, n, cfg.burst_k, rng)
    searches_pm = bookings_pm * cfg.look_to_book * _burst_series(shape, n, cfg.burst_k, rng)

    # --- check-in: online 24h..1h before departure, airport 3h..40min before
    lag_o, w_o = _window_kernel(60, 24 * 60)
    lag_a, w_a = _window_kernel(40, 180, "late_peak")
    pax_flown = flown * ppd
    checkin_pm = (cfg.online_checkin_share * lead_convolve(pax_flown, lag_o, w_o)
                  + (1 - cfg.online_checkin_share) * lead_convolve(pax_flown, lag_a, w_a))
    checkin_pm *= _burst_series(shape, n, cfg.burst_k, rng)

    # --- baggage: bag events in the 2h before departure of flights that actually flew
    lag_b, w_b = _window_kernel(0, 120, "late_peak")
    baggage_pm = lead_convolve(pax_flown * cfg.bags_per_pax * cfg.events_per_bag, lag_b, w_b)

    # --- crew: routine roster checks 3h..0 before departure + re-pairing on disruptions
    lag_c, w_c = _window_kernel(0, 180)
    crew_routine_pm = lead_convolve(deps * cfg.crew_per_flight * cfg.crew_queries_per_member, lag_c, w_c)
    disrupted = cancels + delayed
    # re-pairing happens from ~4h before until ~1h after the disrupted departure
    lag_r, w_r = _window_kernel(-60, 240)
    crew_repair_pm = lead_convolve(disrupted * cfg.crew_per_flight * cfg.crew_ops_per_disrupted_member, lag_r, w_r)

    # --- rebooking: passengers of cancelled flights re-accommodated 6h..1h-after
    lag_rb, w_rb = _window_kernel(-60, 360)
    rebook_pax_pm = lead_convolve(cancels * ppd, lag_rb, w_rb)

    sec = 60.0
    df = pd.DataFrame(index=sched.index)
    df["scheduled_departures_pm"] = deps
    df["cancellations_pm"] = cancels
    df["delayed_pm"] = delayed
    df["search_rps"] = searches_pm / sec
    df["booking_rps"] = bookings_pm / sec
    df["checkin_rps"] = checkin_pm / sec
    df["rebooking_rps"] = rebook_pax_pm / sec
    df["baggage_rps"] = baggage_pm / sec
    df["crew_routine_rps"] = crew_routine_pm / sec
    df["crew_repair_rps"] = crew_repair_pm / sec

    # --- per service demand ----------------------------------------------------
    df["reservations_rps"] = (df["search_rps"] + df["booking_rps"] * cfg.legacy_calls_per_booking
                              + df["checkin_rps"]
                              + df["rebooking_rps"] * (cfg.rebooking_calls_per_pax + cfg.rebooking_searches_per_pax))
    df["crew_rps"] = df["crew_routine_rps"] + df["crew_repair_rps"]
    df["legacy_writes_rps"] = (df["booking_rps"] * cfg.legacy_calls_per_booking
                               + df["checkin_rps"] * cfg.legacy_calls_per_checkin
                               + df["rebooking_rps"] * cfg.rebooking_calls_per_pax
                               + df["crew_repair_rps"] + df["crew_routine_rps"] * 0.25
                               + df["baggage_rps"] * cfg.baggage_legacy_share)
    df["boundary_gateway_rps"] = df["legacy_writes_rps"]

    df.attrs.update({
        "carrier": cfg.carrier,
        "schedule_source": sched.attrs.get("source_file"),
        "pax_per_departure": ppd,
        "pax_per_departure_source": ppd_src,
        "cloud_shape_source": "Azure Functions 2019 (HTTP-triggered)" if shape is not None else "none (flat)",
        "config": asdict(cfg),
    })
    return df


SERVICE_COLUMNS = {
    "reservations": "reservations_rps",
    "crew": "crew_rps",
    "baggage": "baggage_rps",
    "boundary-gateway": "boundary_gateway_rps",
}


def daily_service_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(SERVICE_COLUMNS.values()) + ["scheduled_departures_pm", "cancellations_pm"]
    g = df[cols].resample("1D")
    out = g.mean()
    out["scheduled_departures_pm"] = g["scheduled_departures_pm"].sum()
    out["cancellations_pm"] = g["cancellations_pm"].sum()
    out = out.rename(columns={"scheduled_departures_pm": "flights", "cancellations_pm": "cancelled"})
    for c in SERVICE_COLUMNS.values():
        out[c.replace("_rps", "_peak_rps")] = g[c].max()
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate real-data-driven airline workload")
    ap.add_argument("--carrier", default="WN")
    ap.add_argument("--month", default=None)
    ap.add_argument("--look-to-book", type=float, default=1000.0)
    ap.add_argument("--out", default=None, help="optional CSV path for per-minute output")
    a = ap.parse_args()
    wl = generate_workload(WorkloadConfig(carrier=a.carrier, month=a.month, look_to_book=a.look_to_book))
    print({k: v for k, v in wl.attrs.items() if k != "config"})
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(daily_service_summary(wl).round(1))
    if a.out:
        wl.to_csv(a.out)
