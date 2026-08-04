# ai-models/

Machine learning components for BACCP. Owner: Varad.

## Core Pipeline

- `graph-builder/` — automated trace-based dependency graph construction, with generation-based node typing (`legacy` / `boundary-gateway` / `cloud-native`)
- `cascade-predictor/` — attention-weighted temporal GNN (GAT + GRU) for cross-generation cascade-probability prediction, with generation-gap edge features and boundary sync-drift scoring
- `circuit-breaker/` — reinforcement-learning mitigation agent(s), scoped to boundary/gateway nodes
- `evaluation/` — training/evaluation scripts, baseline comparison harness (flat graph vs. domain-typed vs. generation-typed)

## Advanced Components (added on top of the core pipeline)

### 1. Temporal Point Process module — cascade *timing*, not just probability
**Folder:** `cascade-predictor/timing/`

The GNN predicts *whether* a cascade will cross the legacy/cloud boundary; it is not built to predict *when*. A Hawkes-process-style temporal point process models cascades as self-exciting event sequences: each observed anomaly event raises the conditional intensity (instantaneous probability rate) of a subsequent event, which is exactly the mechanism behind cascading failures.

- **Approach:** feed the GNN's per-node risk embeddings into a neural Hawkes process (start with a Transformer Hawkes Process or the simpler RMTPP formulation before attempting a full Mamba Hawkes model) to estimate the conditional intensity function λ(t) of a boundary-crossing failure event.
- **Output:** an estimated time-to-cascade distribution, not just a binary/probability flag — this is what makes "lead time before manifestation" a rigorous, reportable metric instead of an informal one.
- **Reference starting points:** Recurrent Marked Temporal Point Processes (Du et al., KDD 2016) for the baseline formulation; Transformer Hawkes Process (Zuo et al., ICML 2020) for the modern attention-based version.
- **Complexity:** Medium. RMTPP is implementable in a few weeks; only attempt the Transformer/Mamba variants if time allows.

### 2. Multi-Agent RL circuit breaker — coordinated, not per-node
**Folder:** `circuit-breaker/multi-agent/`

This directly replaces the single-agent DQN+A3C circuit breaker with a multi-agent system, closing a gap the DRL rate-limiting paper's own authors named as unsolved future work (single-service rate limiting can locally optimize one node while starving a dependent one — precisely the cascading-failure scenario this project targets).

- **Approach:** one RL agent per boundary/gateway node, coordinated via a shared critic or communication channel (MAPPO — Multi-Agent Proximal Policy Optimization — is the most stable and well-documented starting point; MADDPG is a valid alternative). Each agent still acts locally (throttle/isolate its own gateway) but the reward function incorporates downstream cloud-service impact, not just local throughput/latency.
- **Output:** globally-aware throttling decisions instead of locally-optimal ones.
- **Reference starting points:** MAPPO (Yu et al., 2021) as the base algorithm; frame the state/action/reward design directly off the original single-agent formulation (8-dim state vector, 7-action threshold-adjustment space) from the DRL rate-limiting paper in the literature review, extended with a coordination term.
- **Complexity:** High. This is the most implementation-heavy addition — budget the most time for it. It is also the strongest "we closed a named gap" claim in the report.

### 3. LLM explanation agent — natural-language incident briefs
**Folder:** `explainability/llm-agent/`

Turns the GNN + Hawkes + RL pipeline's raw output (cascade probability, estimated lead time, which boundary node, which mitigation action taken) into a human-readable diagnostic brief, in the style of current agentic AIOps systems (e.g. AWS DevOps Agent, Azure SRE Agent, and academic multi-agent RCA systems like RCAFlow).

- **Approach:** a retrieval-augmented LLM agent that is given (a) the current dependency graph state, (b) the cascade-probability + timing output, (c) recent eBPF trace snippets from the boundary gateway, and produces a structured incident summary — e.g. "boundary-gateway node X shows 87% cascade probability toward reservations-service in ~40s, driven by MQ queue depth exceeding threshold."
- **Output:** demo-ready, human-facing text — this is the component most worth showing live in a presentation.
- **Reference starting points:** any current-generation LLM API (function-calling / tool-use pattern) with a prompt template constrained to the graph + telemetry context; keep the first version rule-based/templated if time is short, then add LLM generation on top.
- **Complexity:** Low-Medium — this is the fastest to get a visible demo out of, since it's mostly prompt/integration engineering rather than model training.

### 4. Conformal Prediction calibration layer — statistically-guaranteed alerting
**Folder:** `cascade-predictor/calibration/`

Wraps the GNN + Hawkes pipeline's output probability in a model-agnostic, distribution-free calibration layer so that every "trigger the circuit breaker" decision comes with a stated, statistically-valid confidence bound, rather than an uncalibrated raw score. This is a post-processing step — it does not require changing the GNN architecture.

- **Approach:** split conformal prediction with a sliding calibration window (temporal quantile adjustment) to handle distribution shift in the fault-injection data over time; report false-alarm-rate guarantees rather than a bare accuracy number.
- **Output:** a calibrated confidence interval / prediction set alongside every cascade alert, and an explicit, reportable false-alarm-rate bound.
- **Reference starting points:** Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2021) for the foundational method; adaptive/temporal conformal variants (Gibbs & Candès, 2021/2024) for the time-series-appropriate version.
- **Complexity:** Low. This is the cheapest of the four to add — it's a wrapper around whatever the GNN/Hawkes model already outputs, and is a good first addition to get working before tackling multi-agent RL.

## Suggested Build Order

1. Core GNN pipeline (baseline, must exist first)
2. Conformal calibration layer (cheap, wraps the baseline)
3. Temporal point process timing module (medium effort, extends the baseline's output)
4. LLM explanation agent (medium effort, mostly integration — good demo milestone)
5. Multi-agent RL circuit breaker (highest effort, save the most time for this)

## Evaluation additions

In addition to the metrics already listed in the project proposal (lead time, precision/recall/F1, z-score resilience, SLA compliance), report:
- Empirical coverage rate of the conformal prediction intervals vs. the target confidence level
- Time-to-cascade prediction error (Hawkes process) vs. ground-truth fault-injection timestamps
- Global vs. local optimization comparison: multi-agent RL circuit breaker vs. the original single-agent baseline, specifically on scenarios where a locally-optimal throttle would starve a dependent service

Status: not yet implemented — placeholder for Phase-II development.
