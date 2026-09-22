# BACCP Cascade Predictor: Training & Evaluation Report

**Model Architecture**: Heterogeneous Relational Graph Convolutional Network (RGCN) + Temporal GRU (Tier 1A)  
**Evaluation Mode**: Unseen Holdout Test Set (Zero Leakage, Seed 42)  
**Evaluation Timestamp**: 2026-09-22T17:03:33.889908+00:00  
**Checkpoint**: `cascade_predictor_tier1a.pt`  
**Test Samples**: 180 trajectories (5 temporal snapshots per trajectory)

---

## 1. Executive Summary & Key Results

| Metric | Result | Target Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **89.44%** | $\ge 85.0\%$ | **EXCEEDED** |
| **Precision** | **82.24%** | $\ge 80.0\%$ | **EXCEEDED** |
| **Recall (Detection Rate)** | **100.00%** | $\ge 85.0\%$ | **EXCEEDED** |
| **F1-Score** | **0.9026** | $\ge 0.8500$ | **EXCEEDED** |
| **ROC-AUC** | **0.9818** | $\ge 0.8800$ | **EXCEEDED** |
| **Root-Cause Accuracy** | **73.33%** | $\ge 70.0\%$ | **EXCEEDED** |
| **Severity Level Accuracy** | **88.89%** | $\ge 80.0\%$ | **EXCEEDED** |

---

## 2. Lead-Time-Aware Evaluation

In airline IT operations, a prediction is only valuable if it provides adequate warning for automated circuit breakers or human SRE intervention before catastrophic boundary saturation.

* **Mean Warning Time on Impending Cascades**: **176.8 seconds** (~2.9 minutes of advance warning).
* **Mean Lead-Time Absolute Error**: **47.9 seconds**.
* **Precision @ 2-Minute Lead Time ($T_{\\text{lead}} \\ge 120\\text{s}$)**: **71.21%**.
* **Precision @ 5-Minute Lead Time ($T_{\\text{lead}} \\ge 300\\text{s}$)**: **37.04%**.

---

## 3. Confusion Matrix (Binary Cascade Classification)

```
                       Actual Non-Cascade (0)    Actual Cascade (1)
Predicted Non-Cascade (0)        73                        0                   
Predicted Cascade (1)            19                        88                  
```

* **True Positives (TP)**: 88 (Cascades correctly identified and alerted)
* **True Negatives (TN)**: 73 (Nominal/contained operations correctly passed)
* **False Positives (FP)**: 19 (Spurious alerts triggered)
* **False Negatives (FN)**: 0 (Missed cascading failures)

---

## 4. Multi-Task Heads Performance

1. **Cascade Binary Head**: F1 score of **0.9026** with zero false negatives on severe boundary desync.
2. **Lead-Time Regression Head**: Predicts continuous warning horizon with MAE of 47.9s.
3. **Root-Cause Identification**: Accurately isolates the initiating fault domain (`boundary-gateway`, `legacy-core`, or cloud microservice) with **73.33%** accuracy.
4. **Severity Classification**: Maps degradation into standard ITIL operational tiers (`NOMINAL`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) with **88.89%** accuracy.

---

## 5. Architectural Alignment & Verification

* **Input Topology**: 5 nodes (`reservations`, `crew`, `baggage`, `boundary-gateway`, `legacy-core`), 12 directed relation edges.
* **Message Passing**: Heterogeneous linear transformations $W_r$ per relation type with boundary drift bias injection $\epsilon(t)$.
* **Temporal Tracking**: 2-layer GRU over temporal snapshot window $T=5$.
* **Output Conformal Bounds**: Evaluated outputs include $90\%$ conformal prediction interval brackets $[L, U]$ guaranteeing controlled false alarm rates.
