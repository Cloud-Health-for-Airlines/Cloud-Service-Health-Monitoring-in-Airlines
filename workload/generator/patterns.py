"""Workload demand patterns, empirical diurnal profiles, and service topologies for BACCP.

Combines:
1. Public airline operational data (BTS/FAA hourly departure distributions and day-of-week seasonality)
2. Computing resource distributions informed by Google Cluster trace analysis
3. Dynamic workload shapers (diurnal, normal/peak/trough, sudden spikes, gradual ramp-up)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "sample_data"
BTS_PROFILE_PATH = SAMPLE_DATA_DIR / "bts_airline_demand_profile.json"

# Default fallback diurnal curve if BTS profile file is not loaded
DEFAULT_HOURLY_CURVE = {
    0: 0.14, 1: 0.11, 2: 0.08, 3: 0.08, 4: 0.12, 5: 0.28,
    6: 0.65, 7: 0.90, 8: 0.98, 9: 1.00, 10: 0.88, 11: 0.78,
    12: 0.75, 13: 0.74, 14: 0.76, 15: 0.82, 16: 0.91, 17: 0.95,
    18: 0.92, 19: 0.84, 20: 0.70, 21: 0.55, 22: 0.38, 23: 0.22,
}

DEFAULT_DAY_MULTIPLIERS = {
    "Monday": 1.05,
    "Tuesday": 0.92,
    "Wednesday": 0.94,
    "Thursday": 1.02,
    "Friday": 1.15,
    "Saturday": 0.88,
    "Sunday": 1.12,
}

# Canonical 5 services used in BACCP GNN / circuit breaker
CANONICAL_5_SERVICES = [
    "legacy-core",
    "boundary-gateway",
    "reservations",
    "crew",
    "baggage",
]

# Complete 9-service expanded airline IT topology
EXPANDED_9_SERVICES = [
    "legacy-core",
    "boundary-gateway",
    "reservations",
    "flight-search",
    "payment",
    "check-in",
    "crew",
    "baggage",
    "notifications",
]

SERVICE_TIERS = {
    "legacy-core": "legacy",
    "boundary-gateway": "boundary-gateway",
    "reservations": "cloud-native",
    "flight-search": "cloud-native",
    "payment": "cloud-native",
    "check-in": "cloud-native",
    "crew": "cloud-native",
    "baggage": "cloud-native",
    "notifications": "cloud-native",
}


def load_bts_airline_profile(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load empirical BTS airline demand profile from JSON."""
    target_path = Path(path) if path else BTS_PROFILE_PATH
    if target_path.exists():
        try:
            return json.loads(target_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "metadata": {
            "dataset_name": "BTS Airline Demand & Traffic Profile (Default Fallback)",
            "license": "Public Domain",
        },
        "hourly_demand_curve": DEFAULT_HOURLY_CURVE,
        "day_of_week_multipliers": DEFAULT_DAY_MULTIPLIERS,
    }


def get_diurnal_multiplier(hour: float, hourly_curve: Optional[Dict[Any, float]] = None) -> float:
    """Compute interpolated diurnal demand multiplier for any fractional hour of the day [0.0, 24.0).
    
    Uses smooth linear interpolation between empirical hourly reference points.
    """
    curve = hourly_curve or DEFAULT_HOURLY_CURVE
    h_norm = hour % 24.0
    h_floor = int(math.floor(h_norm))
    h_ceil = (h_floor + 1) % 24
    frac = h_norm - h_floor

    val_floor = curve.get(h_floor) or curve.get(str(h_floor), 0.5)
    val_ceil = curve.get(h_ceil) or curve.get(str(h_ceil), 0.5)
    return float(val_floor + frac * (val_ceil - val_floor))


def get_spike_multiplier(
    time_sec: float,
    spike_start: Optional[float],
    spike_duration: float = 60.0,
    spike_magnitude: float = 3.0,
) -> float:
    """Calculate demand multiplier for a sudden traffic surge (e.g. flash sale or weather rebooking).
    
    Smooth trapezoidal envelope with rapid ramp-up, sustained peak, and gradual decay.
    """
    if spike_start is None:
        return 1.0

    if time_sec < spike_start or time_sec > (spike_start + spike_duration):
        return 1.0

    offset = time_sec - spike_start
    # First 10% is ramp up
    ramp_up = spike_duration * 0.1
    # Last 20% is ramp down
    ramp_down = spike_duration * 0.2
    sustain_end = spike_duration - ramp_down

    if offset < ramp_up:
        factor = offset / ramp_up
        return 1.0 + factor * (spike_magnitude - 1.0)
    elif offset <= sustain_end:
        return float(spike_magnitude)
    else:
        factor = (spike_duration - offset) / ramp_down
        return 1.0 + factor * (spike_magnitude - 1.0)


def get_ramp_multiplier(
    time_sec: float,
    ramp_start: Optional[float],
    ramp_duration: float = 300.0,
    ramp_end_factor: float = 2.5,
) -> float:
    """Calculate demand multiplier for a gradual demand increase (e.g., holiday travel ramp-up).
    
    Produces a smooth S-curve ramp from 1.0 to ramp_end_factor.
    """
    if ramp_start is None or time_sec < ramp_start:
        return 1.0

    offset = time_sec - ramp_start
    if offset >= ramp_duration:
        return float(ramp_end_factor)

    progress = offset / ramp_duration
    # Smooth cosine-based S-curve
    smooth_factor = 0.5 * (1.0 - math.cos(progress * math.pi))
    return 1.0 + smooth_factor * (ramp_end_factor - 1.0)


def build_canonical_edges() -> List[Dict[str, Any]]:
    """Return the canonical edges for the 5-node BACCP dependency graph."""
    return [
        {"source": "boundary-gateway", "destination": "legacy-core", "protocol": "tcp", "destination_port": 9090, "observation_count": 3415},
        {"source": "legacy-core", "destination": "boundary-gateway", "protocol": "tcp", "destination_port": 8080, "observation_count": 3410},
        {"source": "reservations", "destination": "boundary-gateway", "protocol": "tcp", "destination_port": 8080, "observation_count": 1420},
        {"source": "boundary-gateway", "destination": "reservations", "protocol": "http", "destination_port": 8081, "observation_count": 1415},
        {"source": "crew", "destination": "boundary-gateway", "protocol": "tcp", "destination_port": 8080, "observation_count": 890},
        {"source": "baggage", "destination": "boundary-gateway", "protocol": "tcp", "destination_port": 8080, "observation_count": 1105},
        {"source": "reservations", "destination": "crew", "protocol": "http", "destination_port": 8082, "observation_count": 210},
        {"source": "reservations", "destination": "baggage", "protocol": "http", "destination_port": 8083, "observation_count": 185},
    ]


def build_expanded_edges() -> List[Dict[str, Any]]:
    """Return dependency edges for the complete 9-service airline IT topology."""
    edges = build_canonical_edges()
    expanded_additions = [
        {"source": "flight-search", "destination": "reservations", "protocol": "http", "destination_port": 8081, "observation_count": 5200},
        {"source": "payment", "destination": "reservations", "protocol": "http", "destination_port": 8081, "observation_count": 1250},
        {"source": "payment", "destination": "boundary-gateway", "protocol": "tcp", "destination_port": 8080, "observation_count": 980},
        {"source": "check-in", "destination": "boundary-gateway", "protocol": "tcp", "destination_port": 8080, "observation_count": 870},
        {"source": "check-in", "destination": "baggage", "protocol": "http", "destination_port": 8083, "observation_count": 760},
        {"source": "reservations", "destination": "notifications", "protocol": "http", "destination_port": 8089, "observation_count": 1400},
        {"source": "crew", "destination": "notifications", "protocol": "http", "destination_port": 8089, "observation_count": 320},
    ]
    return edges + expanded_additions
