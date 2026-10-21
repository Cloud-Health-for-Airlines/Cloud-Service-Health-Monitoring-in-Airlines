"""Loaders for real production cloud / cluster workload traces.

* Azure Functions 2019 -> per-minute request arrivals of real cloud functions.
  We use it for *shape*: diurnal profile and minute-level burstiness.
* Alibaba microservices 2022 (NodeMetrics / MSMetrics) -> real CPU / memory
  utilisation of the machines and containers running production microservices.
  We use it to calibrate how a cloud tier is provisioned (target utilisation)
  and how noisy utilisation is.
"""

from __future__ import annotations

from pathlib import Path
import tarfile
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440


# --------------------------------------------------------------------------- Azure

def _iter_tar_csv(path: Path, name_contains: str) -> Iterator[tuple[str, "object"]]:
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if m.isfile() and name_contains in m.name and m.name.endswith(".csv"):
                yield m.name, tf.extractfile(m)


def load_azure_invocations(path: Path, days: Optional[List[int]] = None,
                           trigger: Optional[str] = "http", top_n: int = 200) -> pd.DataFrame:
    """Per-minute invocation counts for the ``top_n`` busiest functions.

    Returns a frame indexed by absolute minute (day*1440 + minute) with one
    column per function (``HashFunction``). ``trigger='http'`` keeps only
    HTTP-triggered functions, the closest analogue of user-facing APIs such
    as flight search / booking.
    """
    path = Path(path)
    per_day: Dict[int, pd.DataFrame] = {}
    for name, fh in _iter_tar_csv(path, "invocations_per_function"):
        day = int(name.rsplit(".d", 1)[-1].split(".")[0])
        if days and day not in days:
            continue
        df = pd.read_csv(fh)
        if trigger:
            df = df[df["Trigger"] == trigger]
        per_day[day] = df.set_index("HashFunction")[[str(i) for i in range(1, MINUTES_PER_DAY + 1)]]
    if not per_day:
        raise FileNotFoundError(f"no invocation files matched in {path}")
    # keep functions present every selected day, rank by total volume
    common = set.intersection(*(set(d.index) for d in per_day.values()))
    totals = sum(per_day[d].loc[list(common)].sum(axis=1) for d in per_day)
    keep = totals.sort_values(ascending=False).head(top_n).index
    blocks = []
    for d in sorted(per_day):
        block = per_day[d].loc[keep].T.astype(float)
        block.index = (d - 1) * MINUTES_PER_DAY + np.arange(MINUTES_PER_DAY)
        blocks.append(block)
    return pd.concat(blocks)


def human_driven_functions(invocations: pd.DataFrame, min_swing: float = 2.0) -> pd.Index:
    """Functions whose traffic follows a day/night cycle (peak/trough hourly mean >= min_swing).

    Machine-driven HTTP functions (health checks, pollers, webhooks) are flat over
    the day; user-facing APIs - like flight search and booking - are not.
    """
    hour = (invocations.index % MINUTES_PER_DAY) // 60
    hourly = invocations.groupby(hour).mean()
    swing = hourly.max() / hourly.min().replace(0, np.nan)
    return swing[swing >= min_swing].index


def diurnal_profile(invocations: pd.DataFrame, human_driven_only: bool = True) -> np.ndarray:
    """Normalised (mean = 1) minute-of-day demand profile from real traffic.

    Each function is first normalised by its own mean so a few huge functions
    don't dominate; the profile is the median across (human-driven) functions.
    """
    if human_driven_only:
        keep = human_driven_functions(invocations)
        if len(keep) >= 10:
            invocations = invocations[keep]
    norm = invocations / invocations.mean().replace(0, np.nan)
    minute_of_day = invocations.index % MINUTES_PER_DAY
    prof = norm.groupby(minute_of_day).median().median(axis=1).to_numpy()
    prof = pd.Series(prof).rolling(15, center=True, min_periods=1).mean().to_numpy()
    return prof / prof.mean()


def burstiness_residuals(invocations: pd.DataFrame, window: int = 30) -> np.ndarray:
    """Multiplicative minute-level fluctuations of real traffic around its trend.

    residual = count / rolling_mean(count). Pooled over functions and
    returned as a flat array that can be resampled (bootstrap) to give
    synthetic-free, empirically-shaped burstiness.
    """
    trend = invocations.rolling(window, center=True, min_periods=1).mean()
    ratio = (invocations / trend.replace(0, np.nan)).to_numpy().ravel()
    ratio = ratio[np.isfinite(ratio)]
    # trim pathological near-empty functions
    return np.clip(ratio, 0.0, 10.0)


def residual_blocks(invocations: pd.DataFrame, window: int = 30) -> np.ndarray:
    """Per-function residual time series (minutes x functions), keeps autocorrelation."""
    trend = invocations.rolling(window, center=True, min_periods=1).mean()
    ratio = (invocations / trend.replace(0, np.nan)).fillna(1.0).clip(0.0, 10.0)
    keep = invocations.mean() >= 5.0  # functions with enough volume for stable ratios
    return ratio.loc[:, keep].to_numpy()


# --------------------------------------------------------------------------- Alibaba

def load_alibaba_node_metrics(path: Path, max_nodes: Optional[int] = 500) -> pd.DataFrame:
    """Load a NodeMetrics tar.gz -> columns timestamp(ms), nodeid, cpu_utilization, memory_utilization."""
    frames = []
    with tarfile.open(Path(path), "r:*") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".csv"):
                frames.append(pd.read_csv(tf.extractfile(m)))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if max_nodes:
        nodes = sorted(df["nodeid"].unique())[:max_nodes]
        df = df[df["nodeid"].isin(nodes)]
    return df.sort_values(["nodeid", "timestamp"]).reset_index(drop=True)


def utilisation_stats(node_metrics: pd.DataFrame, col: str = "cpu_utilization") -> Dict[str, float]:
    """Summary of how hard a real production cloud runs its machines."""
    v = pd.to_numeric(node_metrics[col], errors="coerce").dropna()
    if v.max() <= 1.0:  # normalised 0..1
        v = v * 100.0
    per_node_std = node_metrics.assign(_v=pd.to_numeric(node_metrics[col], errors="coerce"))\
        .groupby("nodeid")["_v"].std().median()
    scale = 100.0 if pd.to_numeric(node_metrics[col], errors="coerce").max() <= 1.0 else 1.0
    return {
        "mean": float(v.mean()),
        "p50": float(v.quantile(0.50)),
        "p90": float(v.quantile(0.90)),
        "p99": float(v.quantile(0.99)),
        "per_node_std_median": float(per_node_std * scale),
        "n_nodes": int(node_metrics["nodeid"].nunique()),
        "n_samples": int(len(v)),
    }


def utilisation_fluctuations(node_metrics: pd.DataFrame, col: str = "cpu_utilization") -> np.ndarray:
    """Per-node utilisation normalised by the node's mean (minute-level ratio)."""
    df = node_metrics.assign(_v=pd.to_numeric(node_metrics[col], errors="coerce")).dropna(subset=["_v"])
    means = df.groupby("nodeid")["_v"].transform("mean").replace(0, np.nan)
    r = (df["_v"] / means).to_numpy()
    return r[np.isfinite(r)]
