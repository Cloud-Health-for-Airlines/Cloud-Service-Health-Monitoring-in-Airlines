# RCAEval RE1-TT excerpt (test fixture)

Three real Train-Ticket fault-injection cases from RCAEval RE1, cut by
`python ai-models/real_data/rcaeval.py --make-fixture`: 18 of the ~65 services,
5 minutes either side of the injection, one row every 10 s.

| case | fault | label under our rule |
|---|---|---|
| `ts-route-service_delay/1` | network delay on route | cascade |
| `ts-train-service_loss/1` | packet loss on train | cascade |
| `ts-order-service_disk/1` | disk stress on order | no cascade |

Source: Pham et al., "RCAEval: A Benchmark for Root Cause Analysis of Microservice
Systems", WWW 2025 companion, arXiv 2412.17015. Zenodo 10.5281/zenodo.14590730.
Licence: CC-BY 4.0. Values are unchanged apart from the column/row subsetting.
