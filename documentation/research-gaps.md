# Research Gap Analysis (Consolidated)

Synthesized across all 15 reviewed papers (5 from the airline-focused review, 5 from the phase-1 review, 5 from the cross-domain technical deep dive).

## Gap 1 — No legacy/cloud technology-generation boundary in any dependency graph model
Every cascade-prediction and dependency-graph paper reviewed (Krasnovsky & Zorkin; Li 2026; Tang et al./I³) types nodes by service identity or physical infrastructure domain, never by technology generation. None model a legacy-mainframe node differently from a cloud-native node. Barua & Kaiser's airline microservices proposals assume a clean rebuild and never address mainframe/cloud coexistence.

## Gap 2 — No zero-instrumentation observability mechanism applied to the legacy boundary specifically
ZeroTracer (Yang et al.) proves kernel-level eBPF tracing can reconstruct request causality without any application instrumentation — but it targets HTTP microservice tracing generally, not the specific, harder problem of a system that categorically cannot be instrumented (a legacy mainframe).

## Gap 3 — Cascade prediction stays reactive rather than predictive
Zhang et al.'s 98-paper survey confirms the field is still dominated by reactive, post-hoc diagnosis. Li 2026 and Tang et al./I³ move toward prediction, but neither is evaluated on a legacy/cloud boundary scenario, and neither reports a "lead time before manifestation" metric — the most operationally relevant number for a boundary-crossing failure.

## Gap 4 — Mitigation is generic, not boundary-aware
Lyu et al.'s DRL rate limiter is validated at genuine production scale (500M req/day) but operates per-service, with the authors' own stated future work being multi-service coordination — it has no concept of a boundary/gateway node needing distinct treatment from a regular service.

## Gap 5 — Cross-domain heterogeneity has been proven valuable but never applied to this axis
Tang et al.'s I³ model proves that explicitly modeling heterogeneous infrastructure types (electric/road/comms/building) measurably improves cascade prediction (+22.73% F1, +31.94% AUC over baseline) — strong outside evidence that the same approach, applied to a generation-type axis instead of a domain-type axis, should work for the legacy/cloud problem, but nobody has done it.

## Where BACCP Sits

BACCP is positioned exactly at the intersection of Gaps 1–5: it is the first proposed system to (a) type nodes by technology generation, (b) use zero-instrumentation tracing specifically to observe the un-instrumentable legacy boundary, (c) report cascade lead-time as a first-class predictive metric, and (d) scope automated mitigation to the boundary node rather than applying it generically.
