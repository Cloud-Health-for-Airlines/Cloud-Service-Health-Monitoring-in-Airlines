"""Loaders for real U.S. airline data from the Bureau of Transportation Statistics.

* BTS On-Time Performance  -> per-flight schedule, delays, cancellations
* BTS DB1B Market (10% ticket sample) -> passengers per carrier / market

These two sources answer "when do flights happen, how many people are on them,
and when did the airline's operation actually break down" with real numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import zipfile

import numpy as np
import pandas as pd

ONTIME_COLUMNS = [
    "FlightDate", "Reporting_Airline", "Flight_Number_Reporting_Airline", "Tail_Number",
    "Origin", "Dest", "CRSDepTime", "DepTime", "DepDelay", "CRSArrTime", "ArrDelay",
    "Cancelled", "CancellationCode", "Diverted",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]

DB1B_COLUMNS = ["Year", "Quarter", "Origin", "Dest", "TkCarrier", "OpCarrier", "RPCarrier",
                "Passengers", "MktFare", "MktDistance"]

# DB1B is a 10% sample of tickets.
DB1B_SAMPLE_RATE = 0.10


def _read_csv_from_zip(path: Path, usecols: List[str], **kw) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(member) as fh:
            header = pd.read_csv(fh, nrows=0).columns
        cols = [c for c in usecols if c in header]
        with zf.open(member) as fh:
            return pd.read_csv(fh, usecols=cols, low_memory=False, **kw)


def _hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """BTS stores clock times as hhmm integers (e.g. 1435, 2400 = midnight)."""
    v = pd.to_numeric(series, errors="coerce")
    return (v // 100) * 60 + (v % 100)


def load_ontime(paths: Iterable[Path], carriers: Optional[List[str]] = None,
                airports: Optional[List[str]] = None) -> pd.DataFrame:
    """Load one or more monthly On-Time zip files into a tidy per-flight frame.

    Adds ``sched_dep`` (local scheduled departure timestamp) and ``disrupted``
    (cancelled, diverted or departure delay >= 15 min).
    """
    frames = []
    for p in paths:
        df = _read_csv_from_zip(Path(p), ONTIME_COLUMNS)
        if carriers:
            df = df[df["Reporting_Airline"].isin(carriers)]
        if airports:
            df = df[df["Origin"].isin(airports) | df["Dest"].isin(airports)]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    dep_min = _hhmm_to_minutes(df["CRSDepTime"]).fillna(0)
    df["sched_dep"] = df["FlightDate"] + pd.to_timedelta(dep_min, unit="m")
    df["Cancelled"] = df["Cancelled"].fillna(0).astype(int)
    df["Diverted"] = df["Diverted"].fillna(0).astype(int)
    df["DepDelay"] = pd.to_numeric(df["DepDelay"], errors="coerce")
    df["disrupted"] = ((df["Cancelled"] == 1) | (df["Diverted"] == 1) | (df["DepDelay"] >= 15)).astype(int)
    return df.sort_values("sched_dep").reset_index(drop=True)


def daily_carrier_summary(ontime: pd.DataFrame) -> pd.DataFrame:
    """Per carrier per day: flights, cancellation rate, mean delay, delay-cause minutes."""
    g = ontime.groupby(["Reporting_Airline", "FlightDate"])
    out = g.agg(
        flights=("Cancelled", "size"),
        cancelled=("Cancelled", "sum"),
        diverted=("Diverted", "sum"),
        disrupted=("disrupted", "sum"),
        mean_dep_delay=("DepDelay", "mean"),
        carrier_delay_min=("CarrierDelay", "sum"),
        late_aircraft_delay_min=("LateAircraftDelay", "sum"),
        weather_delay_min=("WeatherDelay", "sum"),
    ).reset_index()
    out["cancel_rate"] = out["cancelled"] / out["flights"]
    out["disrupt_rate"] = out["disrupted"] / out["flights"]
    return out


@dataclass
class DisruptionWindow:
    carrier: str
    start: pd.Timestamp
    end: pd.Timestamp
    peak_cancel_rate: float
    baseline_cancel_rate: float
    cancelled_flights: int


def detect_disruption_windows(daily: pd.DataFrame, z_threshold: float = 3.0,
                              min_cancel_rate: float = 0.05) -> List[DisruptionWindow]:
    """Find runs of days where a carrier's cancellation rate is anomalous.

    A day is anomalous when its cancel rate exceeds both ``min_cancel_rate`` and
    the carrier's robust baseline (median + z * MAD). Consecutive anomalous days
    are merged. These are *real* operational meltdowns (e.g. Southwest
    2022-12-21..29, Delta 2024-07-19..24) used as ground-truth incident windows.
    """
    windows: List[DisruptionWindow] = []
    for carrier, grp in daily.groupby("Reporting_Airline"):
        grp = grp.sort_values("FlightDate")
        rate = grp["cancel_rate"].to_numpy()
        med = float(np.median(rate))
        mad = float(np.median(np.abs(rate - med))) * 1.4826 + 1e-4
        flag = (rate > med + z_threshold * mad) & (rate >= min_cancel_rate)
        days = grp["FlightDate"].to_numpy()
        cancelled = grp["cancelled"].to_numpy()
        i = 0
        while i < len(flag):
            if not flag[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(flag) and flag[j + 1]:
                j += 1
            windows.append(DisruptionWindow(
                carrier=carrier,
                start=pd.Timestamp(days[i]),
                end=pd.Timestamp(days[j]),
                peak_cancel_rate=float(rate[i:j + 1].max()),
                baseline_cancel_rate=med,
                cancelled_flights=int(cancelled[i:j + 1].sum()),
            ))
            i = j + 1
    return sorted(windows, key=lambda w: -w.cancelled_flights)


def schedule_series(ontime: pd.DataFrame, freq: str = "1min", carrier: Optional[str] = None,
                    airport: Optional[str] = None) -> pd.DataFrame:
    """Bucket flights onto a regular time grid by *scheduled* departure.

    Columns: departures, cancellations, delayed (>=15 min), disrupted, dep_delay_sum.
    """
    df = ontime
    if carrier:
        df = df[df["Reporting_Airline"] == carrier]
    if airport:
        df = df[df["Origin"] == airport]
    df = df.set_index("sched_dep")
    agg = pd.DataFrame({
        "departures": df["Cancelled"].resample(freq).size(),
        "cancellations": df["Cancelled"].resample(freq).sum(),
        "delayed": (df["DepDelay"] >= 15).resample(freq).sum(),
        "disrupted": df["disrupted"].resample(freq).sum(),
        "dep_delay_sum": df["DepDelay"].clip(lower=0).resample(freq).sum(),
    }).fillna(0)
    return agg


def load_db1b_market(path: Path) -> pd.DataFrame:
    df = _read_csv_from_zip(Path(path), DB1B_COLUMNS)
    df["Passengers"] = pd.to_numeric(df["Passengers"], errors="coerce").fillna(0)
    return df


def passengers_per_carrier(db1b: pd.DataFrame, carrier_col: str = "TkCarrier") -> pd.Series:
    """Estimated full-population passenger trips per carrier for the quarter."""
    return (db1b.groupby(carrier_col)["Passengers"].sum() / DB1B_SAMPLE_RATE).sort_values(ascending=False)


def passengers_per_departure(db1b: pd.DataFrame, ontime: pd.DataFrame, carrier: str,
                             months_in_ontime: int) -> float:
    """Average real passengers generated per scheduled departure for ``carrier``.

    DB1B covers a quarter (3 months); the on-time frame may cover fewer months,
    so departures are scaled to a quarter before dividing.
    """
    pax_quarter = float(passengers_per_carrier(db1b).get(carrier, np.nan))
    deps = int((ontime["Reporting_Airline"] == carrier).sum())
    if not deps or np.isnan(pax_quarter):
        return float("nan")
    deps_quarter = deps * (3.0 / max(1, months_in_ontime))
    return pax_quarter / deps_quarter


def top_markets(db1b: pd.DataFrame, carrier: str, n: int = 20) -> pd.DataFrame:
    sub = db1b[db1b["TkCarrier"] == carrier]
    mk = sub.groupby(["Origin", "Dest"]).agg(passengers=("Passengers", "sum"),
                                             mean_fare=("MktFare", "mean")).reset_index()
    mk["passengers"] = mk["passengers"] / DB1B_SAMPLE_RATE
    return mk.sort_values("passengers", ascending=False).head(n).reset_index(drop=True)
