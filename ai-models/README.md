# ai-models/

Machine learning components for BACCP.

Planned contents:
- `graph-builder/` — automated trace-based dependency graph construction, with generation-based node typing
- `cascade-predictor/` — attention-weighted temporal GNN (GAT + GRU) for cross-generation cascade-probability prediction, with generation-gap edge features and boundary sync-drift scoring
- `circuit-breaker/` — DQN + A3C hybrid reinforcement-learning agent, scoped to boundary/gateway nodes
- `evaluation/` — training/evaluation scripts, baseline comparison harness (flat graph vs. domain-typed vs. generation-typed)

Status: not yet implemented — placeholder for Phase-II development.
