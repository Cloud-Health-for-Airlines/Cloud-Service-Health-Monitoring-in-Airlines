"""BACCP Shared Operational Alert Policy Configuration.

Defines the unified degradation event alert thresholds evaluated in Task 4:
- H=1 (20s): 0.280
- H=5 (100s): 0.300
- H=15 (300s): 0.320

PROVENANCE GAP DISCLOSURE:
These operational thresholds (0.280, 0.300, 0.320) were defined during Task 4
operational evaluation to achieve balanced early warning lead time and high actionable
recall across held-out splits. They were NOT derived via a formal validation-split grid
search or mathematical optimization, nor were they tuned on test data. They represent an
operational policy configuration establishing unified alerting across inference,
explanation, incident brief generation, alert-gated timing, and Prometheus metrics.
"""

from __future__ import annotations
from typing import Dict, Tuple

# Evaluated operational thresholds across supported horizons
ALERT_THRESHOLDS: Dict[int, float] = {
    1: 0.280,
    5: 0.300,
    15: 0.320,
}

DEFAULT_ALERT_THRESHOLD: float = 0.300


def get_alert_threshold(horizon: int) -> float:
    """Returns the configured alert threshold for the specified horizon."""
    return ALERT_THRESHOLDS.get(int(horizon), DEFAULT_ALERT_THRESHOLD)


def evaluate_alert_decision(probability: float, horizon: int) -> Tuple[bool, float]:
    """Evaluates whether an alert should be issued given calibrated event probability and horizon.

    Returns:
        (alert_issued, alert_threshold)
    """
    threshold = get_alert_threshold(horizon)
    alert_issued = bool(float(probability) >= threshold)
    return alert_issued, threshold
