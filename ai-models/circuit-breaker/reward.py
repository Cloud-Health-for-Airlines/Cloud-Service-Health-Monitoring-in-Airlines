"""Tier 1G: Multi-Objective Constrained Reward Design for Airline Circuit Breakers.

Formalizes the trade-offs in automated airline boundary mitigation:
1. Maximize preserved transaction throughput
2. Minimize cascade propagation risk
3. Minimize latency impact weighted by business criticality:
     Priority ordering: reservations (0.50) > crew (0.35) > baggage (0.15)
     (preserving booking checkout revenue is paramount, followed by crew legal duty hours, then baggage reconciliation)
4. Penalize unnecessary gateway isolation / false-positive throttling:
     Throttling healthy traffic strands passengers and costs airline revenue.
"""

from __future__ import annotations

from typing import Dict, Tuple


# Airline service criticality weights (must sum to 1.0)
CRITICALITY_WEIGHTS: Dict[str, float] = {
    "reservations": 0.50,  # Direct passenger booking & ticketing revenue
    "crew": 0.35,          # FAA/EASA legal crew duty flight scheduling
    "baggage": 0.15,       # Baggage tracking & reconciliation
}


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
        nominal_traffic: float,
        service_latencies: Dict[str, float],
        nominal_latencies: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute scalarized reward and breakdown components.
        
        Parameters:
            throttle_rate: float in [0.0, 1.0] applied by circuit breaker
            cascade_risk: model's predicted cascade probability [0.0, 1.0]
            cascade_occurred: whether downstream cascade occurred (ground truth)
            nominal_traffic: base throughput (e.g. 50 req/s)
            service_latencies: actual observed latencies per downstream service
            nominal_latencies: baseline healthy latencies (~10-15ms)
            
        Returns:
            total_reward: scalar float
            components: dict of individual reward terms
        """
        # 1. Throughput component: fraction of allowable traffic successfully passed
        passed_traffic_fraction = 1.0 - throttle_rate
        r_throughput = passed_traffic_fraction * self.w_throughput

        # 2. Cascade avoidance component
        if cascade_risk > 0.5:
            # High hazard state: throttling appropriately avoids catastrophic cascade
            if throttle_rate >= 0.5 and not cascade_occurred:
                r_cascade = 1.0 * self.w_cascade_avoid  # Successfully avoided
            elif throttle_rate < 0.25 and cascade_occurred:
                r_cascade = -1.5 * self.w_cascade_avoid  # Under-mitigated outage
            else:
                r_cascade = 0.0
        else:
            r_cascade = 0.2  # Nominal safe operation bonus

        # 3. Criticality-weighted latency penalty
        # R_lat = - sum_s (w_s * (lat_s - nom_s) / nom_s)
        lat_penalty = 0.0
        for svc, w_s in CRITICALITY_WEIGHTS.items():
            actual_lat = service_latencies.get(svc, 15.0)
            nom_lat = nominal_latencies.get(svc, 15.0)
            if actual_lat > nom_lat:
                normalized_excess = min(5.0, (actual_lat - nom_lat) / nom_lat)
                lat_penalty += w_s * normalized_excess

        r_latency = -lat_penalty * self.w_latency

        # 4. Unnecessary isolation / false-positive throttling penalty
        # If risk is low (< 0.35) but breaker throttles (> 0.10), penalize severely
        if cascade_risk < 0.35 and throttle_rate > 0.10:
            r_false_alarm = - (throttle_rate * 3.5) * self.w_false_alarm_penalty
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
