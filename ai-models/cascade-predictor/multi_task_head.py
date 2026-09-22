"""Tier 1C: Multi-Task Prediction Head for BACCP.

Consumes the shared representation from the Hetero-GNN encoder to simultaneously predict:
1. Cascade probability (binary classification, BCE)
2. Expected lead time in seconds (regression, Smooth-L1)
3. Most-likely root-cause / next-failure node (5-class classification, Cross-Entropy)
4. Cascade severity rating: LOW, MEDIUM, HIGH, CRITICAL (4-class classification, Cross-Entropy)

Loss formulation:
L_total = w_prob * L_prob + w_lead * L_lead + w_root * L_root + w_sev * L_sev
(Note: Kendall et al., 2018 homoscedastic uncertainty loss weighting is the documented upgrade path).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_NAMES = ["legacy-core", "boundary-gateway", "reservations", "crew", "baggage"]
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class MultiTaskHead(nn.Module):
    """Multi-task prediction heads operating on shared graph representation."""

    def __init__(
        self,
        in_dim: int = 48,
        num_nodes: int = 5,
        num_severities: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_severities = num_severities

        # Task 1: Cascade Probability Head
        self.prob_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Task 2: Lead Time Regression Head (in seconds)
        self.lead_time_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Task 3: Root Cause Node Classification Head
        self.root_cause_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_nodes),
        )

        # Task 4: Severity Level Classification Head
        self.severity_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_severities),
        )

    def forward(self, shared_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute multi-task predictions.
        
        shared_repr: [batch_size, in_dim]
        
        Returns:
            dict containing:
                cascade_probability: [batch_size] in [0.0, 1.0]
                lead_time_seconds: [batch_size] in seconds
                root_cause_logits: [batch_size, num_nodes]
                root_cause_probs: [batch_size, num_nodes]
                severity_logits: [batch_size, num_severities]
                severity_probs: [batch_size, num_severities]
        """
        if shared_repr.dim() == 1:
            shared_repr = shared_repr.unsqueeze(0)

        # 1. Probability
        p_logits = self.prob_head(shared_repr).squeeze(-1)
        prob = torch.sigmoid(p_logits)

        # 2. Lead time (in seconds; softplus ensures strictly positive output)
        # Scaled so default output centers around ~60-180 seconds
        lt_raw = self.lead_time_head(shared_repr).squeeze(-1)
        lead_time = F.softplus(lt_raw) * 50.0 + 8.0

        # 3. Root cause node classification
        rc_logits = self.root_cause_head(shared_repr)
        rc_probs = F.softmax(rc_logits, dim=-1)

        # 4. Severity classification
        sev_logits = self.severity_head(shared_repr)
        sev_probs = F.softmax(sev_logits, dim=-1)

        return {
            "cascade_probability": prob,
            "prob_logits": p_logits,
            "lead_time_seconds": lead_time,
            "root_cause_logits": rc_logits,
            "root_cause_probs": rc_probs,
            "severity_logits": sev_logits,
            "severity_probs": sev_probs,
        }

    def compute_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, Any],
        weights: Tuple[float, float, float, float] = (1.0, 0.005, 0.5, 0.5),
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute balanced multi-task loss."""
        w_prob, w_lead, w_root, w_sev = weights

        target_prob = torch.tensor(targets["cascade_occurred"], dtype=torch.float32, device=preds["prob_logits"].device)
        loss_prob = F.binary_cross_entropy_with_logits(preds["prob_logits"].view(-1), target_prob.view(-1))

        # Lead time loss: only computed for cascade runs
        target_lead = torch.tensor(targets["lead_time_seconds"], dtype=torch.float32, device=preds["lead_time_seconds"].device)
        loss_lead = F.smooth_l1_loss(preds["lead_time_seconds"].view(-1), target_lead.view(-1))

        # Root cause target
        rc_name = targets.get("root_cause_node", "boundary-gateway")
        rc_idx = NODE_NAMES.index(rc_name) if rc_name in NODE_NAMES else 1
        target_rc = torch.tensor([rc_idx], dtype=torch.long, device=preds["root_cause_logits"].device)
        loss_root = F.cross_entropy(preds["root_cause_logits"], target_rc)

        # Severity target
        sev_name = targets.get("severity", "LOW")
        sev_idx = SEVERITY_LEVELS.index(sev_name) if sev_name in SEVERITY_LEVELS else 0
        target_sev = torch.tensor([sev_idx], dtype=torch.long, device=preds["severity_logits"].device)
        loss_sev = F.cross_entropy(preds["severity_logits"], target_sev)

        total_loss = w_prob * loss_prob + w_lead * loss_lead + w_root * loss_root + w_sev * loss_sev

        loss_dict = {
            "loss_prob": float(loss_prob.item()),
            "loss_lead": float(loss_lead.item()),
            "loss_root": float(loss_root.item()),
            "loss_sev": float(loss_sev.item()),
            "loss_total": float(total_loss.item()),
        }
        return total_loss, loss_dict
