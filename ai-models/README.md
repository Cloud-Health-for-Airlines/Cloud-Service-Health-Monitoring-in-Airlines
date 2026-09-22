# ai-models/

Machine learning components for BACCP. Owner: Varad.

This folder is organized in tiers so scope stays manageable: **Tier 0** is the working baseline that must exist first, **Tier 1** is the recommended upgrade set (strong novelty-to-effort ratio, realistic in project time), **Tier 2** is stretch work (do these only if Tier 0+1 are solid and there's time left). Pick a stopping point deliberately rather than by running out of time — a well-evaluated Tier 1 system beats a half-working Tier 2 one.

---

## Tier 0 — Baseline (must exist first)

- `graph-builder/` — automated trace-based dependency graph construction, nodes typed `legacy` / `boundary-gateway` / `cloud-native`
- `cascade-predictor/` — attention-weighted temporal GNN (GAT + GRU) with generation-gap edge features and boundary sync-drift score
- `circuit-breaker/` — single-agent RL mitigation, scoped to boundary/gateway nodes
- `evaluation/` — training/evaluation scripts, baseline comparison harness

---

## Tier 1 — Recommended Upgrades

### 1A. Heterogeneous / generation-aware message passing (replaces plain GAT)
**Folder:** `cascade-predictor/hetero-gnn/`

The baseline GAT treats "generation gap" as an edge *feature* — the message-passing weights themselves are the same regardless of whether an edge is legacy→gateway, gateway→cloud, or cloud→cloud. This is the single highest-value upgrade, because it's the one that most directly answers the literature survey's own gap statement (no reviewed work types message-passing weights by technology generation, only by domain).

- **Approach:** move to a Relational GNN (RGCN) or Heterogeneous Graph Transformer (HGT), where `legacy↔gateway`, `gateway↔cloud`, and `cloud↔cloud` are distinct **relation types**, each with its own weight matrix / attention parameters — not just a scalar feature bolted onto a shared weight matrix.
- **Reference starting points:** RGCN (Schlichtkrull et al., 2018) for the simpler relation-specific-weight formulation; HGT (Hu et al., 2020) if you want full heterogeneous attention (type-aware Q/K/V projections) and have the time budget.
- **Complexity:** Medium-High. This should be the first Tier 1 item you build — everything else in the predictor stack sits on top of it.

### 1B. Continuous-time dynamic graph modeling (replaces discrete-snapshot GRU)
**Folder:** `cascade-predictor/temporal/`

The baseline GRU processes discrete time snapshots, which is a poor fit for this specific problem: legacy batch jobs run on a completely different clock (hourly/nightly batch windows) than cloud request traffic (sub-second). Discretizing both onto the same snapshot interval either wastes resolution on the cloud side or misses the batch-job side entirely.

- **Approach:** a Temporal Graph Network (TGN) or DySAT-style continuous-time model that updates each node's memory embedding on every individual eBPF event, rather than on a fixed clock tick.
- **Integration with timing prediction:** feed the TGN's continuously-updated node memory into a Hawkes-process-style intensity function (see Tier 2 below, or fold it in here directly if time allows) — this is a genuinely good combination, since TGN gives you *when the graph state changed* and Hawkes gives you *the probability of the next event given that state*, which together produce a much better-grounded lead-time estimate than either alone.
- **Reference starting points:** TGN (Rossi et al., 2020); DySAT (Sankar et al., 2020) as a lighter-weight alternative if TGN's memory-module complexity is too much for the time budget.
- **Complexity:** High.

### 1C. Multi-task / multi-horizon prediction head
**Folder:** `cascade-predictor/multi-task-head/`

Right now the model outputs one number (cascade probability). A multi-task head gets four outputs from the same shared representation for roughly the cost of one:
1. Probability of cascade crossing the boundary (binary classification)
2. Estimated lead time (regression)
3. Most likely next-failure node (node classification)
4. Cascade size / severity (regression)

- **Approach:** shared GNN encoder (from 1A+1B) → four task-specific heads → a multi-objective loss (start with fixed weighted sum; upgrade to uncertainty-weighted loss balancing — Kendall et al., 2018 — if the fixed weights are hard to tune).
- **Why it's worth it:** this is a cheap add once the encoder exists, and "predicts not just whether but where and how bad" is a strong, concrete claim for the report.
- **Complexity:** Medium (assuming 1A/1B already exist — most of the work is upstream of this).

### 1D. Explainability & calibrated uncertainty
**Folder:** `cascade-predictor/explainability/`

Two distinct things, both worth having:
- **Attention visualization:** since 1A uses attention-based message passing, the attention weights are directly interpretable — surface them (e.g., "62% of the model's attention on this prediction is on the MQ-queue-depth edge") for the dashboard/report.
- **Calibrated uncertainty:** the raw softmax/sigmoid output of a GNN is not a trustworthy probability. Use temperature scaling (cheap, one extra parameter, fit post-hoc) as the baseline, or evidential deep learning (predicts a distribution over probabilities, not just a point estimate) if you want a stronger, more citable method.

This is the mechanism that gets you to the "I am 87% confident a cascade will start at Gateway-X in ~4.2 minutes" statement — genuinely useful for an airline operator, not just a metrics-table number.

- **Reference starting points:** Guo et al., "On Calibration of Modern Neural Networks" (2017) for temperature scaling; Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty" (2018) for the evidential approach.
- **Relationship to the conformal prediction layer (below):** these are complementary, not redundant. Temperature scaling / evidential DL calibrates the model's *own* confidence output; conformal prediction (Tier 2 — carried over from the previous round of additions) wraps *any* model's output in a distribution-free statistical guarantee regardless of whether the model is well-calibrated. Doing both is reasonable: evidential/temperature-scaled uncertainty for the human-readable confidence statement, conformal prediction as the backstop that guarantees the false-alarm rate the circuit breaker actually acts on.
- **Complexity:** Low (temperature scaling) to Medium (evidential deep learning).

### 1E. Physics-/domain-informed features
**Folder:** `graph-builder/features/`

Extends the existing boundary sync-drift score rather than replacing it: inject it as a node/edge embedding bias term (not just a scalar feature concatenated in), and add topological features — betweenness centrality of gateway nodes, generation-aware centrality (a centrality measure that weights paths crossing a generation boundary more heavily than same-generation paths).

- **Complexity:** Low-Medium. Cheap to add once the graph builder exists; mostly feature engineering, not new model architecture.

---

## Tier 1 — RL Side

### 1F. PPO / SAC instead of DQN+A3C
**Folder:** `circuit-breaker/ppo-sac/`

Replace the baseline hybrid DQN+A3C with Proximal Policy Optimization or Soft Actor-Critic. Both are generally more stable to train and more sample-efficient than DQN-style methods, particularly once the action space includes continuous or high-dimensional choices (throttle rate as a continuous value, graceful-degradation level as an ordinal choice, isolate/don't-isolate as discrete) rather than the original paper's small discrete threshold-adjustment space.

- **Approach:** PPO if you want the safer, easier-to-tune default; SAC if the action space ends up meaningfully continuous and sample efficiency matters more than implementation simplicity.
- **Complexity:** Medium. This is a good first RL upgrade to make before attempting hierarchical/multi-agent — get single-agent PPO/SAC solid first.

### 1G. Multi-objective / constrained reward design
**Folder:** `circuit-breaker/reward-design/`

Instead of one scalar reward, explicitly track a vector: maximize remaining throughput, minimize cascade probability, minimize latency impact on critical paths (reservations should be weighted higher than baggage), and penalize unnecessary isolation of the gateway (false-positive throttling has a real operational cost).

- **Approach:** start with scalarization (weighted sum, tuned by hand or via grid search over weight vectors); a Pareto-front / multi-objective RL method (e.g. envelope Q-learning or a multi-objective PPO variant) is a stretch goal if scalarization proves too coarse.
- **Complexity:** Low-Medium for scalarization; High for true Pareto-front methods.

---

## Tier 2 — Stretch Goals (attempt only if Tier 0+1 are solid)

### 2A. Hierarchical / multi-agent RL
**Folder:** `circuit-breaker/hierarchical-multi-agent/`

This is the natural merge of two ideas: a high-level agent decides *whether* to act on a given boundary node at all, and a low-level agent decides the exact throttle/isolation parameters if it does — while multiple gateway nodes are themselves coordinated as separate agents (MAPPO, as in the previous round's addition), sharing a critic so a locally-optimal action at one gateway doesn't starve a dependent cloud service. This directly closes the gap the original DRL rate-limiting paper's authors named as their own future work, and is the single strongest "we closed a named gap" claim available across the whole project.

- **Reference starting points:** MAPPO (Yu et al., 2021) for the coordination layer; standard hierarchical RL (options framework / feudal RL) for the high/low-level split.
- **Complexity:** Very High. Budget the most remaining time here if you attempt it, and treat a working two-agent (not fully general N-agent) version as an acceptable scoped result.

### 2B. Safe RL / constrained MDP
**Folder:** `circuit-breaker/safe-rl/`

Airline-critical constraint: never fully isolate the reservation path, never exceed a maximum false-positive isolation rate. Formalize as a Constrained MDP and enforce with Lagrangian relaxation (a penalty term whose weight is itself learned to keep the constraint satisfied) rather than a hard-coded reward penalty, which tends to be brittle.

- **Reference starting points:** Altman, "Constrained Markov Decision Processes" (1999) for the formalism; Lagrangian-PPO variants (e.g. CPO, Achiam et al., 2017) for a modern implementation.
- **Complexity:** High, but a strong safety-oriented claim if you attempt even a simplified version (hard constraint check as a fallback, with the Lagrangian method as the "soft" primary mechanism).

### 2C. Offline-to-online RL
**Folder:** `circuit-breaker/offline-pretrain/`

Pre-train the circuit breaker on logged chaos-engineering trajectories (DeathStarBench-style + injected faults) before letting it learn online — more realistic than training entirely from scratch against a live/simulated environment, and mirrors how a real deployment would actually be bootstrapped.

- **Complexity:** Medium, mostly an engineering/pipeline change rather than a new algorithm (any of the above RL methods can be pretrained this way).

### 2D. Temporal point process cascade timing (carried over, still valid as a standalone addition if 1B's TGN+Hawkes combination isn't attempted)
**Folder:** `cascade-predictor/timing/`

If 1B (TGN) doesn't get built in time, a standalone Hawkes-process/RMTPP module can still sit on top of the Tier 0 GRU output to get a timing estimate — a lighter-weight fallback for the same capability.

### 2E. Conformal prediction calibration (carried over)
**Folder:** `cascade-predictor/calibration/`

Distribution-free statistical guarantee wrapper described in 1D above — cheap, do this one even if other Tier 2 items are dropped.

### 2F. LLM explanation agent (carried over)
**Folder:** `explainability/llm-agent/`

Turns pipeline output into a human-readable incident brief. Good demo milestone, mostly integration work rather than modeling.

### 2G. Self-supervised / contrastive pre-training
**Folder:** `cascade-predictor/pretraining/`

Pre-train the graph encoder via contrastive learning on normal (non-fault) traffic, so it learns a good representation of "healthy" legacy↔cloud interaction before fine-tuning on the relatively rare cascade-event labels — cascade events will always be a small fraction of your fault-injection data, so this helps sample efficiency.

- **Reference starting points:** graph contrastive learning frameworks such as GraphCL or DGI (Deep Graph Infomax).

### 2H. Causal / interventional modeling
**Folder:** `cascade-predictor/causal/`

Treat each chaos-engineering fault injection as an *intervention* rather than passive observation, and train the model to answer counterfactual questions ("what if we had throttled 30 seconds earlier?"). This is the most research-heavy stretch item — genuinely publishable-adjacent if it works, but also the easiest to run out of time on. Only attempt after everything else is stable.

### 2I. Mixture-of-Experts GNN
**Folder:** `cascade-predictor/moe/`

Route predictions through specialized sub-networks: one expert for legacy-origin cascades (failure starts in the mainframe, propagates outward), another for cloud-origin cascades that propagate backward through the gateway into the legacy side. A gating network decides which expert(s) to weight per prediction.

### 2J. Continual learning / concept drift
**Folder:** `cascade-predictor/continual/`

Handle new microservices or new mainframe batch jobs appearing after initial training without full retraining (e.g. elastic weight consolidation or a simple replay-buffer fine-tuning scheme). Lowest priority of the Tier 2 items — most relevant if you frame the system as production-deployable rather than a fixed evaluation.

---

## Evaluation — Lead-Time-Aware Metrics

Beyond standard precision/recall/F1, report metrics that are largely absent from the reviewed literature and are a genuine contribution on their own regardless of which modeling tiers get built:

- **Mean minutes of warning before cascade manifestation** — the headline number.
- **Precision@lead-time-threshold** — e.g. precision of predictions that fired at least 2 minutes / 5 minutes before manifestation, separately, since a technically-correct prediction that fires 3 seconds before an outage is operationally useless.
- Empirical coverage rate of calibrated/conformal intervals vs. target confidence level (if 1D/2E implemented).
- Global vs. local optimization comparison: coordinated multi-agent circuit breaker vs. single-agent baseline, specifically on scenarios where a locally-optimal throttle would starve a dependent service (if 2A implemented).

---

## Suggested Build Order

1. Tier 0 baseline (must exist first)
2. 1A — Heterogeneous/relational GNN (highest-value single upgrade, closes the named literature gap directly)
3. 1F — PPO/SAC swap-in (stabilizes the RL side before adding coordination complexity)
4. 1C — Multi-task head + 1E — domain-informed features (cheap once 1A exists)
5. 1D — Explainability & calibrated uncertainty (cheap, high report value)
6. 1B — TGN/continuous-time modeling (high effort — only after 1A/1C/1D are solid)
7. 1G — Multi-objective reward design
8. Tier 2, in whatever order fits remaining time — 2A (hierarchical multi-agent RL) and 2E (conformal prediction) are the best time-to-value picks if you can only do two.

---

## Implementation Status & Verification Summary

**Current Status:** Completed (Production-Grade Tier 0 Baseline & Tier 1 Advanced Architecture).

| Component | Target Tier | Module Path | Status | Verification / Artifact |
| :--- | :--- | :--- | :---: | :--- |
| **Graph Schema & Types** | Tier 0 | `graph-builder/schema.py` | Completed | 3 Node types, 3 Edge relations |
| **Sync-Drift & Features** | Tier 0 / 1E | `graph-builder/features.py` | Completed | $\epsilon(t)$ drift + generation centrality |
| **Baseline Cascade Predictor** | Tier 0 | `cascade-predictor/baseline_gat_gru.py` | Completed | GATv2Conv + Temporal GRU |
| **Baseline Circuit Breaker** | Tier 0 | `circuit-breaker/baseline_dqn.py` | Completed | Epsilon-greedy DQN Agent |
| **Architecture Tests** | Tier 0 | `tests/test_tier0_arch.py` | Passed | 4/4 Unit tests passing |
| **Dataset Generation** | Tier 0 / 1 | `data/generate_dataset.py` | Completed | Dual-mode (`live` eBPF + `synthetic` fallback) |
| **Dataset Artifact** | Tier 0 / 1 | `data/dataset/baccp_dataset.pt` | Generated | 60 experimental runs (35 cascades, 25 nominal) |
| **Hetero-RGCN Conv** | Tier 1A | `cascade-predictor/hetero_gnn.py` | Completed | Relation-specific $W_r$, LayerNorm |
| **Multi-Task Head** | Tier 1C | `cascade-predictor/multi_task_head.py` | Completed | Prob, lead-time, root cause, severity |
| **Explainability & Conformal** | Tier 1D / 2E | `cascade-predictor/explainability.py` | Completed | Attention weights, Temperature scaling, Split conformal [lower, upper] |
| **Continuous PPO Agent** | Tier 1F | `circuit-breaker/ppo_agent.py` | Completed | Continuous Beta policy, Actor-Critic |
| **Airline Objective Reward** | Tier 1G | `circuit-breaker/reward.py` | Completed | Weighted: reservations (1.5) > crew (1.0) > baggage (0.7) |
| **Inference Wrappers** | Production | `cascade-predictor/inference.py`, `circuit-breaker/inference.py` | Completed | Zero-dependency model singletons |
| **Backend Integration** | Production | `backend/cloud/sagemaker.py`, `backend/cloud/lambda_handler.py` | Passed | All 22/22 backend tests passing |
| **Evaluation Suite** | Tier 0 / 1 | `evaluation/evaluate.py` | Completed | 3-way graph comparison & policy benchmarks |

### Pinned Reproduction Commands

```bash
# 1. Generate Dataset (Dual-mode: --mode=live for Docker/eBPF, --mode=synthetic for Windows/fallback)
python ai-models/data/generate_dataset.py --mode=synthetic --runs=60

# 2. Run Architectural Verification Tests
python -m unittest ai-models/tests/test_tier0_arch.py

# 3. Train Models
python ai-models/evaluation/train_cascade_predictor.py
python ai-models/evaluation/train_circuit_breaker.py

# 4. Run Evaluation Benchmarks
python ai-models/evaluation/evaluate.py

# 5. Run Full System Backend Tests
python -m unittest backend/tests/test_backend.py
```

