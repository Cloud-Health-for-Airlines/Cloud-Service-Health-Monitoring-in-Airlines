# testbed/

Chaos-engineering fault-injection setup used to validate BACCP without real airline operational data.

Planned contents:
- Kubernetes-based testbed definition (DeathStarBench-style microservices + emulated legacy mainframe stub behind a gateway)
- Chaos Mesh (or equivalent) fault-injection scenarios: network delay, connection drop, batch-job stall, at multiple intensity levels (25/50/75/100%)
- Scripts to run systematic fault-injection sweeps and collect ground-truth cascade data for model evaluation

Status: not yet implemented — placeholder for Phase-II development.
