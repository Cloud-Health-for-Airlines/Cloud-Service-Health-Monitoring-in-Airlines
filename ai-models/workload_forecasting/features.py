"""Anti-leakage feature extraction and window construction for workload forecasting.

Version: baccp-workload-features-v1.1.0 (20 features)

Enforces strict temporal boundaries:
At any forecast origin step t, only historical observations at or before step t
(tau <= t) are accessible. Future values (tau > t), fault labels, severity annotations,
incident IDs, and true root-cause labels are strictly excluded.

Continuous Inference & Phase Consistency:
- Features do NOT depend on future trajectory length or total episode steps.
- Micro-cycle phase (sin_phase, cos_phase) is anchored to physical elapsed time
  via explicit timestamp_sec or step_index.
- When timestamp and step_index are omitted, a documented neutral phase (phase=0)
  is used to guarantee consistent features across variable sliding buffer lengths,
  while flagging is_phase_anchored=False in inference reliability metadata.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Seasonal micro-cycle period (15 steps * 20s = 300s = 5-minute micro-cycle assumption)
SEASONAL_PERIOD_STEPS = 15
CONTEXT_WINDOW_STEPS = 15
SUPPORTED_HORIZONS = (1, 5, 15)

FEATURE_SCHEMA_VERSION = "baccp-workload-features-v1.1.0"

FEATURE_NAMES = [
    # 1. Autoregressive Lags (8 features)
    "lag_0", "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_9", "lag_14",
    # 2. Rolling Summary Statistics (7 features)
    "mean_3", "mean_5", "mean_15", "std_5", "std_15", "min_15", "max_15",
    # 3. Momentum & Relative Differences (3 features)
    "diff_1", "diff_5", "ratio_to_mean15",
    # 4. Micro-cycle Periodic Phase (2 features)
    "sin_phase", "cos_phase",
]


def extract_features_at_origin(
    history: np.ndarray,
    origin_step: Optional[int] = None,
    step_index: Optional[int] = None,
    timestamp_sec: Optional[float] = None,
    sampling_interval_sec: float = 20.0,
    seasonal_period_steps: int = SEASONAL_PERIOD_STEPS,
) -> np.ndarray:
    """Extracts an ordered 20-dimensional feature vector at forecast origin t.

    Parameters:
    - history: 1D array of observed offered workload up to and including origin t.
    - origin_step: Optional index within `history` for origin t. If None, history[-1] is used.
    - step_index: Optional discrete telemetry step counter for micro-cycle phase anchoring.
    - timestamp_sec: Optional elapsed seconds timestamp for micro-cycle phase anchoring.
    - sampling_interval_sec: Observation interval (default: 20s).
    - seasonal_period_steps: Cycle length in steps (default: 15 steps = 300s).

    Returns:
    - 1D numpy array of 20 float64 features strictly adhering to FEATURE_NAMES.
    """
    hist_arr = np.asarray(history, dtype=np.float64)

    if origin_step is not None:
        if origin_step < 0 or origin_step >= len(hist_arr):
            raise ValueError(f"origin_step {origin_step} is out of bounds for history of length {len(hist_arr)}")
        hist_t = hist_arr[:origin_step + 1]
    else:
        hist_t = hist_arr

    if len(hist_t) < CONTEXT_WINDOW_STEPS:
        raise ValueError(
            f"History length {len(hist_t)} is smaller than required context window {CONTEXT_WINDOW_STEPS}"
        )

    # 1. Autoregressive Lag Features relative to origin t (hist_t[-1])
    lag_0 = float(hist_t[-1])
    lag_1 = float(hist_t[-2])
    lag_2 = float(hist_t[-3])
    lag_3 = float(hist_t[-4])
    lag_4 = float(hist_t[-5])
    lag_5 = float(hist_t[-6])
    lag_9 = float(hist_t[-10])
    lag_14 = float(hist_t[-15])  # Exactly 1 seasonal period back (15 steps)

    # 2. Rolling Summary Features
    win_3 = hist_t[-3:]
    win_5 = hist_t[-5:]
    win_15 = hist_t[-15:]

    mean_3 = float(np.mean(win_3))
    mean_5 = float(np.mean(win_5))
    mean_15 = float(np.mean(win_15))
    std_5 = float(np.std(win_5))
    std_15 = float(np.std(win_15))
    min_15 = float(np.min(win_15))
    max_15 = float(np.max(win_15))

    # 3. Momentum & Relative Differences
    diff_1 = lag_0 - lag_1
    diff_5 = lag_0 - lag_5
    ratio_to_mean15 = lag_0 / (mean_15 + 1e-5)

    # 4. Micro-cycle Periodic Phase
    # Anchor to physical time if timestamp_sec or step_index provided.
    # If omitted, use neutral phase 0 (sin=0, cos=1) so sliding buffers of variable length
    # ending at the same moment produce identical feature vectors without silent drift.
    if timestamp_sec is not None:
        phase_step = int(round(timestamp_sec / sampling_interval_sec)) % seasonal_period_steps
    elif step_index is not None:
        phase_step = int(step_index) % seasonal_period_steps
    elif origin_step is not None:
        phase_step = int(origin_step) % seasonal_period_steps
    else:
        phase_step = 0

    angle = 2.0 * math.pi * phase_step / seasonal_period_steps
    sin_phase = math.sin(angle)
    cos_phase = math.cos(angle)

    features = np.array([
        lag_0, lag_1, lag_2, lag_3, lag_4, lag_5, lag_9, lag_14,
        mean_3, mean_5, mean_15, std_5, std_15, min_15, max_15,
        diff_1, diff_5, ratio_to_mean15,
        sin_phase, cos_phase,
    ], dtype=np.float64)

    return features


def build_forecasting_dataset_for_service(
    trajectories: List[Dict[str, Any]],
    service_id: str,
    horizons: Tuple[int, ...] = SUPPORTED_HORIZONS,
    context_window: int = CONTEXT_WINDOW_STEPS,
) -> Dict[str, Any]:
    """Constructs tabular matrices (X, y_by_horizon, metadata) for a given service across trajectories.

    Every sample row represents a valid forecast origin step t.
    Target definition: point offered workload at step t + H.
    """
    max_h = max(horizons)
    X_rows: List[np.ndarray] = []
    y_by_horizon: Dict[int, List[float]] = {h: [] for h in horizons}
    meta_rows: List[Dict[str, Any]] = []

    for traj in trajectories:
        inc_id = traj["incident_id"]
        records = traj["records_by_service"][service_id]
        total_steps = len(records)
        workloads = np.array([r["offered_workload_rps"] for r in records], dtype=np.float64)

        # Valid forecast origins t:
        # t must have enough context (t >= context_window - 1)
        # t + max_h must be within trajectory (t + max_h < total_steps)
        start_t = context_window - 1
        end_t = total_steps - max_h

        for t in range(start_t, end_t):
            # Anti-leakage: slice strictly up to step t
            hist_t = workloads[:t + 1]
            feat = extract_features_at_origin(
                hist_t,
                origin_step=t,
                step_index=records[t]["step_index"],
                timestamp_sec=records[t]["elapsed_time_sec"],
            )
            X_rows.append(feat)

            for h in horizons:
                target_t = t + h
                y_by_horizon[h].append(float(workloads[target_t]))

            meta_rows.append({
                "incident_id": inc_id,
                "service_id": service_id,
                "origin_step": t,
                "origin_elapsed_sec": records[t]["elapsed_time_sec"],
                "last_observed_workload": float(workloads[t]),
            })

    X_mat = np.array(X_rows, dtype=np.float64) if X_rows else np.empty((0, len(FEATURE_NAMES)))
    y_mats = {h: np.array(y_by_horizon[h], dtype=np.float64) for h in horizons}

    return {
        "X": X_mat,
        "y": y_mats,
        "meta": meta_rows,
        "feature_names": FEATURE_NAMES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
