"""Real cloud service-health data: Google Cloud public incident history.

Turns https://status.cloud.google.com/incidents.json into a tidy incident
table and empirical distributions (duration, inter-arrival, severity) that the
trace-driven simulator samples from instead of inventing fault schedules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}

# Keyword map from incident products/descriptions to the failure mode it
# would look like at an airline's cloud/legacy boundary.
FAULT_KEYWORDS = {
    "network-delay": ["network", "latency", "interconnect", "load balanc", "dns", "vpc", "cdn"],
    "connection-drop": ["unavailable", "error", "connectivity", "outage", "5xx", "failed requests", "pub/sub"],
    "batch-job-stall": ["dataflow", "batch", "queue", "backlog", "scheduler", "job", "bigquery", "composer"],
}


def load_gcp_incidents(*paths: Path) -> pd.DataFrame:
    """Load one or more incidents.json files (live feed + archived snapshots), de-duplicated by id."""
    by_id: Dict[str, Dict] = {}
    for path in paths:
        for inc in json.loads(Path(path).read_text(encoding="utf-8")):
            prev = by_id.get(inc.get("id"))
            if prev is None or (inc.get("modified") or "") >= (prev.get("modified") or ""):
                by_id[inc.get("id")] = inc
    rows: List[Dict] = []
    for inc in by_id.values():
        begin = pd.to_datetime(inc.get("begin"), utc=True, errors="coerce")
        end = pd.to_datetime(inc.get("end"), utc=True, errors="coerce")
        products = [p.get("title", "") for p in inc.get("affected_products", []) or []]
        locations = {loc.get("id") for loc in (inc.get("previously_affected_locations") or [])
                     + (inc.get("currently_affected_locations") or []) if isinstance(loc, dict)}
        text = (inc.get("external_desc") or "") + " " + " ".join(products)
        rows.append({
            "id": inc.get("id"),
            "begin": begin,
            "end": end,
            "duration_min": (end - begin).total_seconds() / 60.0 if pd.notna(end) and pd.notna(begin) else np.nan,
            "severity": (inc.get("severity") or "").lower(),
            "status_impact": inc.get("status_impact"),
            "service_name": inc.get("service_name"),
            "products": products,
            "n_products": len(products),
            "n_locations": len(locations),
            "description": (inc.get("external_desc") or "").strip(),
            "fault_class": classify_fault(text),
        })
    df = pd.DataFrame(rows).dropna(subset=["begin"]).sort_values("begin").reset_index(drop=True)
    return df


def classify_fault(text: str) -> str:
    t = text.lower()
    scores = {k: sum(kw in t for kw in kws) for k, kws in FAULT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "connection-drop"


def incident_distributions(incidents: pd.DataFrame) -> Dict[str, object]:
    """Empirical distributions used by the simulator (all in minutes)."""
    dur = incidents["duration_min"].dropna()
    dur = dur[(dur > 0) & (dur < 60 * 24 * 14)]  # drop open / malformed records
    gaps = incidents["begin"].diff().dt.total_seconds().dropna() / 60.0
    return {
        "duration_min": dur.to_numpy(),
        "interarrival_min": gaps[gaps > 0].to_numpy(),
        "severity_share": incidents["severity"].value_counts(normalize=True).to_dict(),
        "fault_class_share": incidents["fault_class"].value_counts(normalize=True).to_dict(),
        "n_incidents": int(len(incidents)),
        "span_days": float((incidents["begin"].max() - incidents["begin"].min()).total_seconds() / 86400.0),
    }


def summary(incidents: pd.DataFrame) -> Dict[str, float]:
    d = incident_distributions(incidents)
    dur = d["duration_min"]
    return {
        "n_incidents": d["n_incidents"],
        "span_days": round(d["span_days"], 1),
        "incidents_per_30d": round(d["n_incidents"] / max(1.0, d["span_days"]) * 30, 2),
        "duration_median_min": round(float(np.median(dur)), 1),
        "duration_p90_min": round(float(np.quantile(dur, 0.9)), 1),
        "duration_mean_min": round(float(np.mean(dur)), 1),
    }
