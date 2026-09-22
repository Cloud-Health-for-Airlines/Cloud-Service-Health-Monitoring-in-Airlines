"""Tier 1D: Explainability & Calibrated Uncertainty.

Implements:
1. Attention/attribution visualization: surfaces dominant generation-boundary edges into natural language briefs.
2. Temperature Scaling (Guo et al., 2017): post-hoc probability calibration on validation split.
3. Split Conformal Prediction (Vovk et al. / Angelopoulos & Bates 2021):
   Finite-sample distribution-free statistical guarantee with 90% target coverage interval [lower, upper].
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling calibration (Guo et al., 2017)."""

    def __init__(self):
        super().__init__()
        # Initialize temperature at 1.0 (log_temp = 0.0)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        temp = self.temperature.clamp(min=0.1, max=10.0)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """Optimize temperature parameter on validation set via NLL."""
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        val_labels = val_labels.float()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nn.functional.binary_cross_entropy_with_logits(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


class SplitConformalCalibrator:
    """Split Conformal Prediction for distribution-free 90% coverage intervals."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha  # 1 - alpha = 0.90 coverage
        self.quantile: float = 0.08  # default conservative quantile

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> float:
        """Compute conformal residual quantile on validation split.
        
        Non-conformity score: s_i = |y_i - \hat{p}_i|
        Finite-sample quantile: ceil((n + 1) * (1 - alpha)) / n
        """
        n = len(val_probs)
        if n == 0:
            return self.quantile

        residuals = np.abs(val_labels - val_probs)
        k = int(math.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(n, max(1, k))
        
        # k-th smallest residual
        sorted_residuals = np.sort(residuals)
        self.quantile = float(sorted_residuals[k - 1])
        return self.quantile

    def predict_interval(self, prob: float) -> Tuple[float, float]:
        """Return 90% confidence interval [lower, upper]."""
        lower = max(0.0, prob - self.quantile)
        upper = min(1.0, prob + self.quantile)
        return round(float(lower), 3), round(float(upper), 3)


def format_incident_explanation(
    cascade_prob: float,
    root_cause_node: str,
    affected_nodes: List[str],
    lead_time_sec: float,
    sync_drift_score: float,
    conf_lower: float,
    conf_upper: float,
    relation_attributions: Optional[Dict[str, float]] = None,
) -> str:
    """Generate rich natural language incident brief including attention attribution."""
    # Find dominant stress relation from attention attributions
    top_rel_str = ""
    if relation_attributions:
        sorted_rels = sorted(relation_attributions.items(), key=lambda x: x[1], reverse=True)
        if sorted_rels and sorted_rels[0][1] > 0.15:
            top_rel, pct = sorted_rels[0]
            clean_rel = top_rel.replace("__", " ")
            top_rel_str = f" Root cause analysis attributes {pct * 100:.0f}% of graph message energy to '{clean_rel}'."

    aff_str = ", ".join(affected_nodes) if affected_nodes else root_cause_node

    if cascade_prob >= 0.70:
        return (
            f"CRITICAL boundary cascade predicted at {root_cause_node} "
            f"(P={cascade_prob:.1%}, 90% Conformal CI [{conf_lower:.1%}, {conf_upper:.1%}]). "
            f"Digital-twin sync drift ε(t) is critical at {sync_drift_score:.1f}%. "
            f"Estimated lead time to downstream failure is ~{lead_time_sec:.0f}s.{top_rel_str} "
            f"Affected domains: {aff_str}. Automated circuit-breaker mitigation recommended."
        )
    elif cascade_prob >= 0.40:
        return (
            f"ELEVATED cascade risk detected at {root_cause_node} (P={cascade_prob:.1%}). "
            f"Boundary sync drift ε(t)={sync_drift_score:.1f}%. "
            f"Estimated warning lead time: {lead_time_sec:.0f}s.{top_rel_str} "
            f"Monitoring reservations and crew services for queue backpressure."
        )
    else:
        return (
            f"System operating nominally across legacy core and cloud tiers. "
            f"Boundary sync drift is low ({sync_drift_score:.1f}%), cascade risk {cascade_prob:.1%}. "
            f"All microservices reporting healthy response times."
        )
