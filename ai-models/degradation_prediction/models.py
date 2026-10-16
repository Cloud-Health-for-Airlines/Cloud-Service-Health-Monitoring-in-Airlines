"""Baselines and Learned Models for Degradation and Boundary Deterioration Prediction.

Version: baccp-degradation-models-v1.1.0

Implements:
1. Trivial and Statistical Baselines:
   - AlwaysPositiveBaseline: Predicts 1.0 everywhere.
   - AlwaysNegativeBaseline: Predicts 0.0 everywhere.
   - TrainingPrevalenceBaseline: Predicts empirical training prevalence.
   - PersistenceDegradationBaseline: Persists current latency, backlog, drift, and degradation state.
2. Repaired CausalQueueRolloutBaseline:
   - Step-by-step rollout over k in 1..H using documented linear arrival path.
   - Observable capacity only (configured nominal for cloud/gateway; completion-rate proxy for legacy).
     Zero access to future internal simulator capacity shocks.
   - Capped by service buffer limits B_max, explicit drops, and mass-conserving departures.
   - Boundary drift persisted from origin t (documented baseline assumption).
3. Learned HistGradientBoosting:
   - HistGradientBoostingRegressor for Latency, Backlog, and Sync Drift.
   - HistGradientBoostingClassifier for Forward-Window Degradation Event Risk.
   - Isotonic probability calibration fitted strictly on calibration partition.
   - Alert threshold tuning strictly on validation partition after applying calibration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

try:
    from .data_contract import (
        BOUNDARY_SYNC_DRIFT_THRESHOLD,
        CONFIGURED_SERVICE_CAPACITIES,
        NOMINAL_BASELINES,
        SAMPLING_INTERVAL_SECONDS,
        DegradationSeverity,
        DegradationType,
        ServiceIdentifier,
        SLO_THRESHOLDS,
    )
except ImportError:
    from degradation_prediction.data_contract import (
        BOUNDARY_SYNC_DRIFT_THRESHOLD,
        CONFIGURED_SERVICE_CAPACITIES,
        NOMINAL_BASELINES,
        SAMPLING_INTERVAL_SECONDS,
        DegradationSeverity,
        DegradationType,
        ServiceIdentifier,
        SLO_THRESHOLDS,
    )

BUFFER_LIMITS: Dict[str, float] = {
    ServiceIdentifier.RESERVATIONS.value: 400.0,
    ServiceIdentifier.CREW.value: 300.0,
    ServiceIdentifier.BAGGAGE.value: 350.0,
    ServiceIdentifier.BOUNDARY_GATEWAY.value: 600.0,
    ServiceIdentifier.LEGACY_CORE.value: 500.0,
}


def simulate_step_by_step_rollout(
    service_id: str,
    curr_backlog: float,
    curr_offered: float,
    forecast_demand: float,
    curr_completed: float,
    curr_sync_drift: float,
    horizon: int,
    dt: float = SAMPLING_INTERVAL_SECONDS,
) -> Dict[str, Any]:
    """Causal step-by-step queue and latency rollout forward over H discrete steps.

    Dynamics:
    - Linear arrival path: lambda(t+k) = lambda(t) + (k / H) * (lambda_hat(t+H) - lambda(t))
    - Observable capacity: configured nominal capacity for cloud/gateway;
      observable throughput proxy for uninstrumented legacy-core.
    - Buffer conservation: departures min(Q_avail, M), drops max(0, R - B_max),
      Q(t+k) = R - drops >= 0.
    - Latency: L_0 + (Q / C) * 1000 ms, strictly capped by buffer limit.
    - Boundary drift: Persisted from origin t (documented baseline assumption).
    """
    nominal_cap = CONFIGURED_SERVICE_CAPACITIES.get(service_id)
    # Observable capacity: known provisioned nominal or completion throughput proxy for legacy
    eff_cap = nominal_cap if nominal_cap is not None else max(1.0, curr_completed)
    buffer_max = BUFFER_LIMITS.get(service_id, 400.0)
    base_lat = NOMINAL_BASELINES[service_id]["latency_ms"]
    slo = SLO_THRESHOLDS.get(service_id, {})
    slo_lat = slo.get("latency_ms", 100.0)
    slo_q = slo.get("backlog_requests", 50.0)

    q = max(0.0, float(curr_backlog))
    max_lat = base_lat
    max_q = q
    total_drops = 0.0
    degraded = False

    for k in range(1, horizon + 1):
        # Linear arrival path approximation
        lam_k = curr_offered + (float(k) / float(horizon)) * (forecast_demand - curr_offered)
        arr_k = max(0.0, lam_k) * dt
        q_avail = q + arr_k
        max_work = eff_cap * dt
        dep = min(q_avail, max_work)
        rem = q_avail - dep
        drops = max(0.0, rem - buffer_max)
        q = max(0.0, rem - drops)
        total_drops += drops

        q_delay_ms = (q / (eff_cap + 1e-4)) * 1000.0
        lat_k = base_lat + q_delay_ms

        max_lat = max(max_lat, lat_k)
        max_q = max(max_q, q)

        err_fraction = drops / (q_avail + 1e-4) if q_avail > 0 else 0.0
        if lat_k >= slo_lat or q >= slo_q or err_fraction >= 0.05:
            degraded = True

    final_lat = base_lat + (q / (eff_cap + 1e-4)) * 1000.0
    # Persisted boundary drift assumption
    final_drift = float(curr_sync_drift)

    return {
        "latency_ms": float(final_lat),
        "backlog_requests": float(q),
        "boundary_sync_drift_pct": float(final_drift),
        "degradation_risk": 1.0 if degraded else 0.0,
        "total_drops": float(total_drops),
        "max_latency_in_window": float(max_lat),
        "max_backlog_in_window": float(max_q),
    }


class AlwaysPositiveBaseline:
    """Trivial baseline predicting degradation for all inputs."""
    def predict(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.ones(n_samples, dtype=int), np.ones(n_samples, dtype=float)


class AlwaysNegativeBaseline:
    """Trivial baseline predicting nominal for all inputs."""
    def predict(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(n_samples, dtype=int), np.zeros(n_samples, dtype=float)


class TrainingPrevalenceBaseline:
    """Statistical baseline predicting training prevalence."""
    def __init__(self, train_prevalence: float = 0.50):
        self.prevalence = float(train_prevalence)

    def predict(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        pred_label = 1 if self.prevalence >= 0.50 else 0
        return np.full(n_samples, pred_label, dtype=int), np.full(n_samples, self.prevalence, dtype=float)


class PersistenceDegradationBaseline:
    """Current-state persistence baseline."""

    def __init__(self):
        self.name = "persistence"

    def predict_continuous(
        self,
        curr_latency: float,
        curr_backlog: float,
        curr_sync_drift: float,
    ) -> Dict[str, float]:
        return {
            "latency_ms": float(curr_latency),
            "backlog_requests": float(curr_backlog),
            "boundary_sync_drift_pct": float(curr_sync_drift),
        }

    def predict_event_prob(
        self,
        service_id: str,
        curr_latency: float,
        curr_backlog: float,
        curr_error_rate: float,
        curr_sync_drift: float,
    ) -> float:
        thresh = SLO_THRESHOLDS.get(service_id, {})
        is_degraded = (
            curr_latency >= thresh.get("latency_ms", 100.0)
            or curr_backlog >= thresh.get("backlog_requests", 50.0)
            or curr_error_rate >= thresh.get("error_rate", 0.05)
            or (service_id == ServiceIdentifier.BOUNDARY_GATEWAY.value and curr_sync_drift >= BOUNDARY_SYNC_DRIFT_THRESHOLD)
        )
        return 1.0 if is_degraded else 0.0


class CausalQueueRolloutBaseline:
    """Causal queue mass-conservation and capacity projection baseline."""

    def __init__(self):
        self.name = "causal_rollout"

    def predict(
        self,
        service_id: str,
        curr_latency: float,
        curr_backlog: float,
        curr_offered: float,
        forecast_demand: float,
        curr_completed: float,
        curr_sync_drift: float,
        horizon: int,
    ) -> Dict[str, float]:
        res = simulate_step_by_step_rollout(
            service_id=service_id,
            curr_backlog=curr_backlog,
            curr_offered=curr_offered,
            forecast_demand=forecast_demand,
            curr_completed=curr_completed,
            curr_sync_drift=curr_sync_drift,
            horizon=horizon,
        )
        return {
            "latency_ms": res["latency_ms"],
            "backlog_requests": res["backlog_requests"],
            "boundary_sync_drift_pct": res["boundary_sync_drift_pct"],
            "degradation_risk": res["degradation_risk"],
        }


class StaticThresholdBaseline:
    """Proactive baseline triggering an alert whenever any current metric exceeds 75% of SLO threshold."""

    def __init__(self, ratio: float = 0.75):
        self.name = "static_threshold"
        self.ratio = ratio

    def predict_event_prob(
        self,
        service_id: str,
        curr_latency: float,
        curr_backlog: float,
        curr_error_rate: float,
        curr_sync_drift: float,
    ) -> float:
        thresh = SLO_THRESHOLDS.get(service_id, {})
        is_warn = (
            curr_latency >= self.ratio * thresh.get("latency_ms", 100.0)
            or curr_backlog >= self.ratio * thresh.get("backlog_requests", 50.0)
            or curr_error_rate >= self.ratio * thresh.get("error_rate", 0.05)
            or (
                service_id == ServiceIdentifier.BOUNDARY_GATEWAY.value
                and curr_sync_drift >= self.ratio * BOUNDARY_SYNC_DRIFT_THRESHOLD
            )
        )
        return 1.0 if is_warn else 0.0


class SimpleAnomalyBaseline:
    """Velocity-based anomaly baseline triggering when metric delta exceeds 2-sigma of nominal."""

    def __init__(self):
        self.name = "simple_anomaly"

    def predict_event_prob(
        self,
        lat_delta: float,
        q_delta: float,
        drift_delta: float,
        lat_threshold: float = 12.0,
        q_threshold: float = 8.0,
        drift_threshold: float = 6.0,
    ) -> float:
        is_anomaly = (
            abs(lat_delta) >= lat_threshold
            or abs(q_delta) >= q_threshold
            or abs(drift_delta) >= drift_threshold
        )
        return 1.0 if is_anomaly else 0.0


class LearnedDegradationHorizonModel:
    """Gradient boosting suite for a single forecast horizon (H in {1, 5, 15})."""

    def __init__(
        self,
        horizon: int,
        include_workload_forecasts: bool = True,
        random_state: int = 42,
    ):
        self.horizon = horizon
        self.include_workload_forecasts = include_workload_forecasts
        self.random_state = random_state

        # Regressors for continuous targets
        self.reg_latency = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            min_samples_leaf=15,
            learning_rate=0.08,
            random_state=random_state,
        )
        self.reg_backlog = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            min_samples_leaf=15,
            learning_rate=0.08,
            random_state=random_state,
        )
        self.reg_drift = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            min_samples_leaf=15,
            learning_rate=0.08,
            random_state=random_state,
        )

        # Classifier for forward-window degradation event risk
        self.clf_event = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=6,
            min_samples_leaf=15,
            learning_rate=0.08,
            random_state=random_state,
        )

        # Calibrator fitted strictly on calibration partition
        self.calibrator: Optional[IsotonicRegression] = None
        self.alert_threshold: float = 0.50

    def fit_train(
        self,
        X_train: np.ndarray,
        y_lat_train: np.ndarray,
        y_backlog_train: np.ndarray,
        y_drift_train: np.ndarray,
        y_event_train: np.ndarray,
    ) -> None:
        """Trains regression and classification models on training split."""
        self.reg_latency.fit(X_train, y_lat_train)
        self.reg_backlog.fit(X_train, y_backlog_train)
        self.reg_drift.fit(X_train, y_drift_train)
        self.clf_event.fit(X_train, y_event_train)

    def calibrate(self, X_cal: np.ndarray, y_event_cal: np.ndarray) -> None:
        """Fits isotonic calibrator strictly on disjoint calibration partition."""
        raw_probs = self.clf_event.predict_proba(X_cal)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(raw_probs, y_event_cal)

    def tune_threshold(self, X_val: np.ndarray, y_event_val: np.ndarray) -> float:
        """Selects optimal alert threshold on validation split strictly AFTER applying calibration."""
        probs = self.predict_event_proba(X_val)
        best_f1 = -1.0
        best_thresh = 0.50

        for thresh in np.linspace(0.15, 0.85, 71):
            preds = (probs >= thresh).astype(int)
            tp = int(np.sum((preds == 1) & (y_event_val == 1)))
            fp = int(np.sum((preds == 1) & (y_event_val == 0)))
            fn = int(np.sum((preds == 0) & (y_event_val == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        self.alert_threshold = round(best_thresh, 3)
        return self.alert_threshold

    def predict_continuous(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predicts continuous targets at t+H."""
        pred_lat = np.maximum(0.0, self.reg_latency.predict(X))
        pred_backlog = np.maximum(0.0, self.reg_backlog.predict(X))
        pred_drift = np.clip(self.reg_drift.predict(X), 8.5, 99.5)
        return {
            "latency_ms": pred_lat,
            "backlog_requests": pred_backlog,
            "boundary_sync_drift_pct": pred_drift,
        }

    def predict_event_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts calibrated probability of degradation event in (t, t+H]."""
        raw_probs = self.clf_event.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            cal_probs = self.calibrator.predict(raw_probs)
            return np.clip(cal_probs, 0.0, 1.0)
        return raw_probs

    def predict_alert(self, X: np.ndarray) -> np.ndarray:
        """Returns binary alert indicator based on validated threshold."""
        probs = self.predict_event_proba(X)
        return (probs >= self.alert_threshold).astype(int)


class ConditionalTimeToOnsetRegressor:
    """Conditional time-to-onset regressor evaluated strictly on observed degradation events.

    Treats non-event incidents as censored. Evaluated strictly on eligible observed events.
    Returns predicted seconds to onset, or None when an onset estimate is unsupported/censored.
    """

    def __init__(self, random_state: int = 42):
        self.reg = HistGradientBoostingRegressor(
            max_iter=80,
            max_depth=5,
            min_samples_leaf=10,
            learning_rate=0.08,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X_observed: np.ndarray, y_time_steps: np.ndarray) -> None:
        """Trains on observed positive events only."""
        if len(X_observed) >= 10:
            self.reg.fit(X_observed, y_time_steps)
            self.is_fitted = True

    def predict_time_to_onset_sec(
        self, X: np.ndarray, alert_active: np.ndarray
    ) -> List[Optional[float]]:
        """Predicts estimated time to onset in seconds. Returns None if alert is not active or censored."""
        dt = SAMPLING_INTERVAL_SECONDS
        if not self.is_fitted:
            return [None] * len(X)

        pred_steps = self.reg.predict(X)
        results: List[Optional[float]] = []
        for idx in range(len(X)):
            if alert_active[idx] == 1:
                est_sec = max(dt, float(pred_steps[idx] * dt))
                results.append(round(est_sec, 1))
            else:
                results.append(None)
        return results


class DegradationTypeAndSeverityClassifier:
    """Predicts observable degradation type and severity using documented SLO exceedance rules."""

    def __init__(self, random_state: int = 42):
        self.clf_type = HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=5,
            min_samples_leaf=10,
            learning_rate=0.08,
            random_state=random_state,
        )
        self.clf_severity = HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=5,
            min_samples_leaf=10,
            learning_rate=0.08,
            random_state=random_state,
        )
        self.type_labels = [
            DegradationType.NOMINAL.value,
            DegradationType.LATENCY_BREACH.value,
            DegradationType.BACKLOG_SATURATION.value,
            DegradationType.ERROR_SPIKE.value,
            DegradationType.BOUNDARY_DRIFT_DETERIORATION.value,
            DegradationType.COMPOUND_MULTIPLE.value,
        ]
        self.severity_labels = [
            DegradationSeverity.NOMINAL.value,
            DegradationSeverity.MINOR.value,
            DegradationSeverity.MAJOR.value,
            DegradationSeverity.CRITICAL.value,
        ]
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_type_str: List[str],
        y_severity_str: List[str],
    ) -> None:
        """Trains multi-class models for observable type and severity."""
        type_idx = np.array([self.type_labels.index(t) if t in self.type_labels else 0 for t in y_type_str], dtype=int)
        sev_idx = np.array([self.severity_labels.index(s) if s in self.severity_labels else 0 for s in y_severity_str], dtype=int)

        self.clf_type.fit(X_train, type_idx)
        self.clf_severity.fit(X_train, sev_idx)
        self.is_fitted = True

    def predict(self, X: np.ndarray, alert_active: np.ndarray) -> Tuple[List[str], List[str]]:
        """Predicts observable type and severity. Returns NOMINAL if alert is not active."""
        if not self.is_fitted:
            return ([DegradationType.NOMINAL.value] * len(X), [DegradationSeverity.NOMINAL.value] * len(X))

        pred_t_idx = self.clf_type.predict(X)
        pred_s_idx = self.clf_severity.predict(X)

        out_types = []
        out_sevs = []
        for i in range(len(X)):
            if alert_active[i] == 1:
                t_str = self.type_labels[min(len(self.type_labels) - 1, max(0, int(pred_t_idx[i])))]
                s_str = self.severity_labels[min(len(self.severity_labels) - 1, max(0, int(pred_s_idx[i])))]
                out_types.append(t_str)
                out_sevs.append(s_str)
            else:
                out_types.append(DegradationType.NOMINAL.value)
                out_sevs.append(DegradationSeverity.NOMINAL.value)

        return out_types, out_sevs


class SystemLevelCalibrator:
    """Fits and evaluates a separate system-level calibrator on aggregated calibration origins.

    Transforms the system composite risk score (max service probability) into a true
    calibrated system-wide probability of degradation within (t, t+H].
    """

    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(self, composite_risk_scores: np.ndarray, y_system_true: np.ndarray) -> None:
        self.calibrator.fit(composite_risk_scores, y_system_true)
        self.is_fitted = True

    def predict_proba(self, composite_risk_scores: np.ndarray) -> np.ndarray:
        if self.is_fitted:
            cal_p = self.calibrator.predict(composite_risk_scores)
            return np.clip(cal_p, 0.0, 1.0)
        return composite_risk_scores


class PyTorchTemporalGRU(nn.Module):
    """Small CPU-friendly GRU with temporal neighbor context and capacity headroom."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head_event = nn.Linear(hidden_dim, 1)
        self.head_time = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        event_logit = self.head_event(last).squeeze(-1)
        time_pred = torch.relu(self.head_time(last)).squeeze(-1)
        return event_logit, time_pred


class LearnedTemporalHorizonModel:
    """Small CPU-friendly temporal GRU comparator using neighbor context and capacity headroom."""

    def __init__(self, horizon: int, input_dim: int = 16, hidden_dim: int = 32, random_state: int = 42):
        self.horizon = horizon
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.model = PyTorchTemporalGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2)
        self.scaler = StandardScaler()
        self.calibrator: Optional[IsotonicRegression] = None
        self.alert_threshold: float = 0.50

    def fit_train(
        self,
        X_tensors: np.ndarray,
        y_event: np.ndarray,
        y_time: np.ndarray,
        epochs: int = 12,
    ) -> None:
        N, seq_len, D = X_tensors.shape
        X_flat = X_tensors.reshape(-1, D)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(N, seq_len, D)

        device = torch.device("cpu")
        self.model.to(device)
        self.model.train()

        pos_count = max(1, int(np.sum(y_event == 1)))
        neg_count = max(1, int(np.sum(y_event == 0)))
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)

        criterion_event = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion_time = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003, weight_decay=1e-4)

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_ev_t = torch.tensor(y_event, dtype=torch.float32)
        y_tm_t = torch.tensor(y_time, dtype=torch.float32)

        batch_size = 64
        num_batches = int(math.ceil(N / batch_size))

        for _ in range(epochs):
            perm = torch.randperm(N)
            for b in range(num_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                optimizer.zero_grad()
                logits, t_preds = self.model(X_t[idx])
                loss_ev = criterion_event(logits, y_ev_t[idx])

                pos_mask = (y_ev_t[idx] == 1)
                loss_tm = (
                    criterion_time(t_preds[pos_mask], y_tm_t[idx][pos_mask])
                    if pos_mask.sum() > 0
                    else torch.tensor(0.0)
                )

                loss = loss_ev + 0.1 * loss_tm
                loss.backward()
                optimizer.step()

        self.model.eval()

    def calibrate(self, X_tensors_cal: np.ndarray, y_event_cal: np.ndarray) -> None:
        raw_probs = self.predict_event_proba_raw(X_tensors_cal)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(raw_probs, y_event_cal)

    def tune_threshold(self, X_tensors_val: np.ndarray, y_event_val: np.ndarray) -> float:
        probs = self.predict_event_proba(X_tensors_val)
        best_f1 = -1.0
        best_thresh = 0.50
        for thresh in np.linspace(0.15, 0.85, 71):
            preds = (probs >= thresh).astype(int)
            tp = int(np.sum((preds == 1) & (y_event_val == 1)))
            fp = int(np.sum((preds == 1) & (y_event_val == 0)))
            fn = int(np.sum((preds == 0) & (y_event_val == 1)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
        self.alert_threshold = round(best_thresh, 3)
        return self.alert_threshold

    def predict_event_proba_raw(self, X_tensors: np.ndarray) -> np.ndarray:
        N, seq_len, D = X_tensors.shape
        X_flat = X_tensors.reshape(-1, D)
        X_scaled = self.scaler.transform(X_flat).reshape(N, seq_len, D)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict_event_proba(self, X_tensors: np.ndarray) -> np.ndarray:
        raw_probs = self.predict_event_proba_raw(X_tensors)
        if self.calibrator is not None:
            cal_probs = self.calibrator.predict(raw_probs)
            return np.clip(cal_probs, 0.0, 1.0)
        return raw_probs

    def predict_alert(self, X_tensors: np.ndarray) -> np.ndarray:
        probs = self.predict_event_proba(X_tensors)
        return (probs >= self.alert_threshold).astype(int)

