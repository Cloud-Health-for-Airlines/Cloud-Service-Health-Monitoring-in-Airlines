"""Tier 1G: Multi-Objective Constrained Reward Design for Airline Circuit Breakers.

Formalizes the trade-offs in automated airline boundary mitigation:
1. Maximize preserved transaction throughput:
     Priority ordering: reservations (0.50) > crew (0.35) > baggage (0.15)
     (preserving booking checkout revenue is paramount, followed by crew legal duty hours, then baggage reconciliation)
2. Minimize cascade propagation risk
3. Penalize unnecessary gateway isolation / false-positive throttling on healthy traffic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# Airline service criticality weights (sum to 1.0)
CRITICALITY_WEIGHTS: Dict[str, float] = {
    "reservations": 0.50,  # Direct passenger booking & ticketing revenue
    "crew": 0.35,          # FAA/EASA legal crew duty flight scheduling
    "baggage": 0.15,       # Baggage tracking & reconciliation
}


def compute_service_throughput(throttle_rate: float) -> Dict[str, float]:
    """Calculate service retention rates according to business priority shedding.

    Priority shedding order:
    1. Low throttle (0.0 - 0.20): Shed non-critical baggage traffic first.
    2. Medium throttle (0.20 - 0.50): Shed all baggage, buffer crew duty scheduling.
    3. High throttle (0.50 - 1.0): Protect core ticketing/reservations until unavoidable.
    """
    th = max(0.0, min(1.0, float(throttle_rate)))

    # Baggage: sheds across [0.0, 0.20]
    if th <= 0.20:
        bag = 1.0 - (th / 0.20)
    else:
        bag = 0.0

    # Crew: sheds across [0.20, 0.50]
    if th <= 0.20:
        crew = 1.0
    elif th <= 0.50:
        crew = 1.0 - (th - 0.20) / 0.30
    else:
        crew = 0.0

    # Reservations: sheds across [0.50, 1.0]
    if th <= 0.50:
        res = 1.0
    else:
        res = 1.0 - (th - 0.50) / 0.50

    res = round(max(0.0, min(1.0, res)), 4)
    crew = round(max(0.0, min(1.0, crew)), 4)
    bag = round(max(0.0, min(1.0, bag)), 4)

    weighted = round(
        CRITICALITY_WEIGHTS["reservations"] * res
        + CRITICALITY_WEIGHTS["crew"] * crew
        + CRITICALITY_WEIGHTS["baggage"] * bag,
        4,
    )

    return {
        "reservations": res,
        "crew": crew,
        "baggage": bag,
        "weighted": weighted,
    }


def compute_reward(
    throttle_rate: float,
    cascade_risk: float,
    cascade_occurred: bool,
    throughput_info: Optional[Dict[str, float]] = None,
    service_latencies: Optional[Dict[str, float]] = None,
    nominal_latencies: Optional[Dict[str, float]] = None,
) -> float:
    """Convenience function returning scalar multi-objective reward."""
    tp = throughput_info or compute_service_throughput(throttle_rate)
    r_tp = tp["weighted"] * 2.0

    if cascade_occurred:
        r_cascade = -15.0
    elif cascade_risk > 0.50 and throttle_rate >= 0.25:
        r_cascade = 4.0  # Successfully avoided cascade under hazard
    else:
        r_cascade = 0.5

    # False alarm penalty: throttling healthy traffic
    if cascade_risk < 0.30 and throttle_rate > 0.05:
        r_false_alarm = -(throttle_rate * 5.0)
    else:
        r_false_alarm = 0.0

    return round(float(r_tp + r_cascade + r_false_alarm), 4)


class MultiObjectiveReward:
    """Computes scalarized multi-objective reward for boundary circuit breaker."""

    def __init__(
        self,
        w_throughput: float = 1.0,
        w_cascade_avoid: float = 2.0,
        w_latency: float = 0.8,
        w_false_alarm_penalty: float = 1.5,
    ):
        self.w_throughput = w_throughput
        self.w_cascade_avoid = w_cascade_avoid
        self.w_latency = w_latency
        self.w_false_alarm_penalty = w_false_alarm_penalty

    def calculate_reward(
        self,
        throttle_rate: float,
        cascade_risk: float,
        cascade_occurred: bool,
        nominal_traffic: float = 50.0,
        service_latencies: Optional[Dict[str, float]] = None,
        nominal_latencies: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute scalarized reward and breakdown components."""
        tp_info = compute_service_throughput(throttle_rate)
        r_throughput = tp_info["weighted"] * self.w_throughput

        if cascade_risk > 0.5:
            if throttle_rate >= 0.35 and not cascade_occurred:
                r_cascade = 1.0 * self.w_cascade_avoid
            elif cascade_occurred:
                r_cascade = -2.0 * self.w_cascade_avoid
            else:
                r_cascade = 0.0
        else:
            r_cascade = 0.2

        lat_penalty = 0.0
        if service_latencies and nominal_latencies:
            for svc, w_s in CRITICALITY_WEIGHTS.items():
                actual_lat = service_latencies.get(svc, 15.0)
                nom_lat = nominal_latencies.get(svc, 15.0)
                if actual_lat > nom_lat:
                    normalized_excess = min(5.0, (actual_lat - nom_lat) / nom_lat)
                    lat_penalty += w_s * normalized_excess

        r_latency = -lat_penalty * self.w_latency

        if cascade_risk < 0.35 and throttle_rate > 0.10:
            r_false_alarm = -(throttle_rate * 3.5) * self.w_false_alarm_penalty
        else:
            r_false_alarm = 0.0

        total_reward = r_throughput + r_cascade + r_latency + r_false_alarm

        components = {
            "r_throughput": round(float(r_throughput), 4),
            "r_cascade": round(float(r_cascade), 4),
            "r_latency": round(float(r_latency), 4),
            "r_false_alarm": round(float(r_false_alarm), 4),
            "r_total": round(float(total_reward), 4),
        }
        return float(total_reward), components
