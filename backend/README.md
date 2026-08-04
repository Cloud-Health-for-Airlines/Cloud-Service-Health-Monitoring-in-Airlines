# backend/

Backend services for BACCP: microservice simulators, the integration gateway, and API layer connecting the frontend, database, and ai-models components.

Planned contents:
- `gateway/` — integration gateway simulator (MQ/EDI/batch-to-API adapter) with eBPF tracepoints attached
- `services/` — reservation, crew-scheduling, and baggage-handling microservice simulators (DeathStarBench-style)
- `api/` — REST/GraphQL API exposing graph state, alerts, and telemetry to the frontend
- AWS wiring: CloudWatch/X-Ray instrumentation, Lambda action handlers, SNS alert publishing

Status: not yet implemented — placeholder for Phase-II development.
