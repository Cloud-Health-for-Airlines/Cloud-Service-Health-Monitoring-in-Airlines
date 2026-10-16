"""Comprehensive Evaluation Suite for BACCP Workload Forecasting.

Version: baccp-evaluation-v1.1.0

Evaluates all models across:
1. Standard Held-Out Test Split (25 independent incidents)
2. Diagnostic Unseen Extreme Surge Split (25 independent incidents, examined in Task 1)
3. Locked Unseen Extreme Surge Split (25 fresh, independently seeded locked incidents)

Evaluated Models:
- persistence: Last observed workload baseline
- seasonal_naive: S=15 seasonal lag baseline
- autoregressive_ridge: Regularized linear autoregression
- gradient_boosting: HistGradientBoostingRegressor
- auto_fallback: Training-derived adaptive policy (switches from GB to Ridge AR when workload > training max)

Computes:
- MAE, RMSE, WAPE (with explicit zero-denominator policy)
- Fallback activation rates
- Interval coverage and mean width under stationary and surge conditions
- 95% Confidence Intervals via whole-incident bootstrap (500 resamples)
- Publication-quality diagnostic plots under results/workload_forecasting/plots/
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data_contract import CONFIGURED_SERVICE_CAPACITIES, ServiceIdentifier
from .features import SUPPORTED_HORIZONS, build_forecasting_dataset_for_service
from .train import load_dataset_by_splits

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "ai-models" / "data" / "workload_forecasting" / "workload_dataset.json"
WEIGHTS_PATH = REPO_ROOT / "ai-models" / "weights" / "workload_forecasting" / "workload_models_bundle.pkl"
RESULTS_DIR = REPO_ROOT / "results" / "workload_forecasting"
PLOTS_DIR = RESULTS_DIR / "plots"


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes MAE, RMSE, WAPE, and prediction interval coverage & width."""
    errors = np.abs(y_true - y_pred)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Zero-denominator policy:
    # If sum(y_true) == 0: return 0.0 when total absolute error is 0.0; otherwise return None (undefined).
    denom = float(np.sum(y_true))
    total_abs_error = float(np.sum(errors))
    if denom == 0.0:
        if total_abs_error == 0.0:
            wape = 0.0
            wape_explanation = "Zero total actual demand with zero absolute error -> WAPE=0.0%"
        else:
            wape = None
            wape_explanation = "Undefined: zero total actual demand with non-zero forecast error"
    else:
        wape = round(float(total_abs_error / denom * 100.0), 2)
        wape_explanation = None

    out: Dict[str, Any] = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape_pct": wape,
        "wape_explanation": wape_explanation,
    }

    if lower_bound is not None and upper_bound is not None:
        covered = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage_pct = float(np.mean(covered) * 100.0)
        mean_width = float(np.mean(upper_bound - lower_bound))
        out["interval_coverage_pct"] = round(coverage_pct, 2)
        out["interval_mean_width"] = round(mean_width, 3)

    return out


def evaluate_model_on_split(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    svc_models: Dict[str, Any],
    svc_calibrators: Dict[str, Any],
    surge_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Computes predictions, intervals, and fallback activation rate for a specified model."""
    n_samples = len(X)
    if model_name == "auto_fallback":
        # Adaptive policy: if lag_0 (X[:, 0]) > surge_threshold, use Ridge AR; else GB
        y_t = X[:, 0]
        surge_mask = y_t > surge_threshold
        fallback_rate = float(np.mean(surge_mask) * 100.0)

        preds = np.empty(n_samples, dtype=np.float64)
        lower = np.empty(n_samples, dtype=np.float64)
        upper = np.empty(n_samples, dtype=np.float64)

        gb_preds = svc_models["gradient_boosting"].predict(X)
        gb_l, gb_u = svc_calibrators["gradient_boosting"].get_intervals(gb_preds)

        ridge_preds = svc_models["autoregressive_ridge"].predict(X)
        ridge_l, ridge_u = svc_calibrators["autoregressive_ridge"].get_intervals(ridge_preds)

        # Standard condition: Gradient Boosting
        preds[~surge_mask] = gb_preds[~surge_mask]
        lower[~surge_mask] = gb_l[~surge_mask]
        upper[~surge_mask] = gb_u[~surge_mask]

        # Surge condition: Autoregressive Ridge
        preds[surge_mask] = ridge_preds[surge_mask]
        lower[surge_mask] = ridge_l[surge_mask]
        upper[surge_mask] = ridge_u[surge_mask]

        return preds, lower, upper, fallback_rate

    else:
        model = svc_models[model_name]
        calibrator = svc_calibrators[model_name]
        preds = model.predict(X)
        lower, upper = calibrator.get_intervals(preds)
        return preds, lower, upper, 0.0


def bootstrap_incident_metrics(
    trajectories: List[Dict[str, Any]],
    service_id: str,
    horizon: int,
    model_name: str,
    svc_models: Dict[str, Any],
    svc_calibrators: Dict[str, Any],
    surge_threshold: float,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Computes 95% confidence intervals by bootstrapping whole incidents."""
    rng = np.random.RandomState(seed)
    n_inc = len(trajectories)
    if n_inc == 0:
        return {}

    inc_results = []
    for traj in trajectories:
        data = build_forecasting_dataset_for_service([traj], service_id, (horizon,))
        X = data["X"]
        y = data["y"][horizon]
        if len(X) == 0:
            continue
        p, l, u, _ = evaluate_model_on_split(model_name, X, y, svc_models, svc_calibrators, surge_threshold)
        inc_results.append((y, p, l, u))

    if not inc_results:
        return {}

    boot_maes = []
    boot_rmses = []
    boot_wapes = []
    boot_covs = []

    for _ in range(n_bootstraps):
        sampled_indices = rng.choice(len(inc_results), size=len(inc_results), replace=True)
        y_all = np.concatenate([inc_results[idx][0] for idx in sampled_indices])
        p_all = np.concatenate([inc_results[idx][1] for idx in sampled_indices])
        l_all = np.concatenate([inc_results[idx][2] for idx in sampled_indices])
        u_all = np.concatenate([inc_results[idx][3] for idx in sampled_indices])

        m = compute_metrics(y_all, p_all, l_all, u_all)
        boot_maes.append(m["mae"])
        boot_rmses.append(m["rmse"])
        if m["wape_pct"] is not None:
            boot_wapes.append(m["wape_pct"])
        boot_covs.append(m.get("interval_coverage_pct", 0.0))

    out: Dict[str, Tuple[float, float]] = {
        "mae_ci95": (round(float(np.percentile(boot_maes, 2.5)), 3), round(float(np.percentile(boot_maes, 97.5)), 3)),
        "rmse_ci95": (round(float(np.percentile(boot_rmses, 2.5)), 3), round(float(np.percentile(boot_rmses, 97.5)), 3)),
        "coverage_ci95": (round(float(np.percentile(boot_covs, 2.5)), 2), round(float(np.percentile(boot_covs, 97.5)), 2)),
    }
    if boot_wapes:
        out["wape_ci95"] = (round(float(np.percentile(boot_wapes, 2.5)), 2), round(float(np.percentile(boot_wapes, 97.5)), 2))
    return out


def evaluate_all(
    dataset_path: Path = DATA_PATH,
    weights_path: Path = WEIGHTS_PATH,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """Master evaluation routine across all services, horizons, models, and splits."""
    results_dir.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(weights_path, "rb") as f:
        bundle = pickle.load(f)

    models_bundle = bundle["models"]
    calibrators_bundle = bundle["calibrators"]
    training_bounds = bundle["metadata"].get("training_domain_bounds", {})

    partitioned, manifest = load_dataset_by_splits(dataset_path)
    test_std_trajs = partitioned["test_standard"]
    test_diag_trajs = partitioned["test_unseen_surge_diagnostic"]
    test_locked_trajs = partitioned["test_unseen_surge_locked"]

    services = [s.value for s in ServiceIdentifier]
    horizons = SUPPORTED_HORIZONS
    model_names = [
        "persistence",
        "seasonal_naive",
        "autoregressive_ridge",
        "gradient_boosting",
        "auto_fallback",
    ]

    print(f"[*] Running Comprehensive Evaluation (Task 1A)...")
    print(f"    Standard Test Incidents:    {len(test_std_trajs)}")
    print(f"    Diagnostic Surge Incidents: {len(test_diag_trajs)}")
    print(f"    Locked Surge Incidents:     {len(test_locked_trajs)}")

    evaluation_report: Dict[str, Any] = {
        "metadata": {
            "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "contract_version": manifest.get("contract_version"),
            "dataset_sha256": manifest.get("dataset_sha256"),
            "counts": {
                "standard_test": len(test_std_trajs),
                "diagnostic_surge_test": len(test_diag_trajs),
                "locked_surge_test": len(test_locked_trajs),
            },
            "horizons": list(horizons),
            "services": services,
            "models": model_names,
            "zero_denominator_policy": (
                "If sum(y_true)==0 and sum(|y-y_hat|)==0 then WAPE=0.0%; "
                "if sum(y_true)==0 and sum(|y-y_hat|)>0 then WAPE=null (undefined)"
            ),
        },
        "standard_test": {},
        "diagnostic_surge_test": {},
        "locked_surge_test": {},
    }

    eval_splits = [
        ("standard_test", test_std_trajs),
        ("diagnostic_surge_test", test_diag_trajs),
        ("locked_surge_test", test_locked_trajs),
    ]

    for split_key, trajs in eval_splits:
        evaluation_report[split_key] = {}

        for svc in services:
            evaluation_report[split_key][svc] = {}
            split_data = build_forecasting_dataset_for_service(trajs, svc, horizons)
            X = split_data["X"]
            surge_th = training_bounds.get(svc, {}).get("surge_threshold_rps", 1e9)

            for h in horizons:
                y = split_data["y"][h]
                evaluation_report[split_key][svc][str(h)] = {}

                svc_models = models_bundle[svc][h]
                svc_cals = calibrators_bundle[svc][h]

                # Persistence baseline
                p_preds, p_l, p_u, _ = evaluate_model_on_split(
                    "persistence", X, y, svc_models, svc_cals, surge_th
                )
                persist_metrics = compute_metrics(y, p_preds, p_l, p_u)
                persist_mae = persist_metrics["mae"]

                for m_name in model_names:
                    preds, lower, upper, fallback_rate = evaluate_model_on_split(
                        m_name, X, y, svc_models, svc_cals, surge_th
                    )
                    m = compute_metrics(y, preds, lower, upper)
                    m["fallback_activation_rate_pct"] = round(fallback_rate, 2)

                    # Improvement vs Persistence: positive = better, negative = worse
                    rel_improvement = ((persist_mae - m["mae"]) / max(1e-5, persist_mae)) * 100.0
                    m["rel_improvement_vs_persistence_pct"] = round(rel_improvement, 2)

                    # Whole-incident bootstrap 95% CI
                    ci = bootstrap_incident_metrics(
                        trajs, svc, h, m_name, svc_models, svc_cals, surge_th, n_bootstraps=500, seed=42
                    )
                    m["bootstrap_ci95"] = ci

                    evaluation_report[split_key][svc][str(h)][m_name] = m

    # Save machine-readable JSON
    summary_path = results_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(evaluation_report, indent=2))
    print(f"[+] Saved metrics summary to {summary_path}")

    # Generate plots
    generate_evaluation_plots(partitioned, bundle, evaluation_report)

    # Generate Markdown Report
    generate_markdown_report(evaluation_report, results_dir / "evaluation_report.md")

    return evaluation_report


def generate_evaluation_plots(
    partitioned: Dict[str, List[Dict[str, Any]]],
    bundle: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    """Generates all 4 publication-quality diagnostic plots."""
    models_bundle = bundle["models"]
    calibrators_bundle = bundle["calibrators"]
    training_bounds = bundle["metadata"].get("training_domain_bounds", {})
    test_std_trajs = partitioned["test_standard"]
    test_locked_trajs = partitioned["test_unseen_surge_locked"]

    services = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]
    horizons = [1, 5, 15]

    # 1. Traces Plot: Standard Test (Reservations, H=5)
    sample_traj = test_std_trajs[0]
    records = sample_traj["records_by_service"]["reservations"]
    times = [r["elapsed_time_sec"] for r in records]
    actual_workload = np.array([r["offered_workload_rps"] for r in records])

    h5_data = build_forecasting_dataset_for_service([sample_traj], "reservations", (5,))
    X_h5 = h5_data["X"]
    gb_model = models_bundle["reservations"][5]["gradient_boosting"]
    gb_cal = calibrators_bundle["reservations"][5]["gradient_boosting"]
    gb_preds = gb_model.predict(X_h5)
    gb_l, gb_u = gb_cal.get_intervals(gb_preds)
    target_times = [records[m["origin_step"] + 5]["elapsed_time_sec"] for m in h5_data["meta"]]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.plot(times, actual_workload, label="Observed Ground Truth (Offered RPS)", color="#2563eb", linewidth=1.8)
    ax.plot(target_times, gb_preds, label="Gradient Boosting Forecast (H=5, 100s)", color="#059669", linestyle="--", linewidth=1.6)
    ax.fill_between(target_times, gb_l, gb_u, color="#059669", alpha=0.20, label="Nominal 90% Conformal Interval")
    ax.axhline(100.0, color="#dc2626", linestyle=":", linewidth=1.2, label="Configured Capacity (100 RPS)")
    ax.set_title(f"BACCP Workload Forecast — Reservations API ({sample_traj['scenario_name']})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax.set_ylabel("Offered Workload (requests/sec)", fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    plot1_path = PLOTS_DIR / "workload_forecast_traces.png"
    fig.savefig(plot1_path)
    plt.close(fig)
    print(f"[+] Saved trace plot to {plot1_path}")

    # 2. Error by Service and Horizon (Bar plot: Standard Test MAE across all models)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300, sharey=False)
    for idx, h in enumerate(horizons):
        ax = axes[idx]
        x_indices = np.arange(len(services))
        width = 0.16

        per_maes = [report["standard_test"][s][str(h)]["persistence"]["mae"] for s in services]
        sn_maes = [report["standard_test"][s][str(h)]["seasonal_naive"]["mae"] for s in services]
        ar_maes = [report["standard_test"][s][str(h)]["autoregressive_ridge"]["mae"] for s in services]
        gb_maes = [report["standard_test"][s][str(h)]["gradient_boosting"]["mae"] for s in services]
        fb_maes = [report["standard_test"][s][str(h)]["auto_fallback"]["mae"] for s in services]

        ax.bar(x_indices - 2 * width, per_maes, width, label="Persistence", color="#94a3b8")
        ax.bar(x_indices - width, sn_maes, width, label="Seasonal Naive", color="#f59e0b")
        ax.bar(x_indices, ar_maes, width, label="Ridge AR", color="#8b5cf6")
        ax.bar(x_indices + width, gb_maes, width, label="Gradient Boosting", color="#10b981")
        ax.bar(x_indices + 2 * width, fb_maes, width, label="Auto Fallback", color="#0284c7")

        ax.set_title(f"Horizon H={h} ({h*20}s)", fontsize=11, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(["Res", "Crew", "Bag", "Gateway", "Legacy"], fontsize=9)
        ax.set_ylabel("MAE (requests/sec)" if idx == 0 else "")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=7.5)

    fig.suptitle("Forecasting MAE Across Services and Horizons (Standard Test)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plot2_path = PLOTS_DIR / "error_by_service_and_horizon.png"
    fig.savefig(plot2_path)
    plt.close(fig)
    print(f"[+] Saved error plot to {plot2_path}")

    # 3. Interval Coverage by Service and Horizon (Standard vs Locked Surge)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300, sharey=True)
    for s_idx, (s_name, s_title) in enumerate([("standard_test", "Standard Held-Out Test"), ("locked_surge_test", "Locked Fresh Surge Test")]):
        ax = axes[s_idx]
        x_indices = np.arange(len(services))
        width = 0.25

        cov_h1 = [report[s_name][s]["1"]["auto_fallback"]["interval_coverage_pct"] for s in services]
        cov_h5 = [report[s_name][s]["5"]["auto_fallback"]["interval_coverage_pct"] for s in services]
        cov_h15 = [report[s_name][s]["15"]["auto_fallback"]["interval_coverage_pct"] for s in services]

        ax.bar(x_indices - width, cov_h1, width, label="H=1 (20s)", color="#38bdf8")
        ax.bar(x_indices, cov_h5, width, label="H=5 (100s)", color="#818cf8")
        ax.bar(x_indices + width, cov_h15, width, label="H=15 (300s)", color="#c084fc")

        ax.axhline(90.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Nominal 90% Target")
        ax.set_title(s_title, fontsize=11, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(["Res", "Crew", "Bag", "Gateway", "Legacy"], fontsize=9)
        ax.set_ylabel("Empirical Coverage (%)" if s_idx == 0 else "")
        ax.set_ylim(0.0, 105.0)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Conformal 90% Interval Coverage: Stationary vs Distribution Shift", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plot3_path = PLOTS_DIR / "interval_coverage_by_horizon.png"
    fig.savefig(plot3_path)
    plt.close(fig)
    print(f"[+] Saved coverage plot to {plot3_path}")

    # 4. Held-out Surge Example: Fallback Policy vs Gradient Boosting Saturation
    locked_traj = test_locked_trajs[0]
    recs = locked_traj["records_by_service"]["reservations"]
    t_axis = [r["elapsed_time_sec"] for r in recs]
    y_actual = np.array([r["offered_workload_rps"] for r in recs])

    surge_h1 = build_forecasting_dataset_for_service([locked_traj], "reservations", (1,))
    X_s1 = surge_h1["X"]
    y_s1 = surge_h1["y"][1]
    surge_th = training_bounds["reservations"]["surge_threshold_rps"]

    # Model predictions at H=1
    gb_p1 = models_bundle["reservations"][1]["gradient_boosting"].predict(X_s1)
    fb_p1, fb_l1, fb_u1, _ = evaluate_model_on_split(
        "auto_fallback", X_s1, y_s1, models_bundle["reservations"][1], calibrators_bundle["reservations"][1], surge_th
    )
    t_targets = [recs[m["origin_step"] + 1]["elapsed_time_sec"] for m in surge_h1["meta"]]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.plot(t_axis, y_actual, label="Observed Surge Workload (Offered RPS)", color="#e11d48", linewidth=2.0)
    ax.plot(t_targets, gb_p1, label="Unmodified Gradient Boosting (Saturates at Leaf Max)", color="#64748b", linestyle=":", linewidth=1.8)
    ax.plot(t_targets, fb_p1, label="Auto Fallback Policy (Ridge AR Linear Extrapolation)", color="#0284c7", linestyle="--", linewidth=1.8)
    ax.axhline(100.0, color="#b91c1c", linestyle="--", linewidth=1.2, label="Configured Capacity (100 RPS)")
    ax.axhline(surge_th, color="#eab308", linestyle="-.", linewidth=1.2, label=f"Training Max Threshold ({surge_th:.1f} RPS)")
    ax.set_title(f"Locked Surge Evaluation — Fallback Policy vs Tree Saturation (Reservations H=1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax.set_ylabel("Offered Workload (requests/sec)", fontsize=10)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    plot4_path = PLOTS_DIR / "heldout_surge_case_study.png"
    fig.savefig(plot4_path)
    plt.close(fig)
    print(f"[+] Saved held-out surge plot to {plot4_path}")


def generate_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    """Writes detailed GitHub-flavored markdown report summarizing all metrics."""
    std = report["standard_test"]
    diag = report["diagnostic_surge_test"]
    lock = report["locked_surge_test"]
    services = report["metadata"]["services"]
    horizons = ["1", "5", "15"]
    models = ["persistence", "seasonal_naive", "autoregressive_ridge", "gradient_boosting", "auto_fallback"]

    lines = [
        "# BACCP Workload Forecasting Comprehensive Evaluation Report (Task 1A Audit)",
        "",
        f"**Generated:** {report['metadata']['evaluation_timestamp']}  ",
        f"**Dataset SHA-256:** `{report['metadata']['dataset_sha256']}`  ",
        f"**Contract Version:** `{report['metadata']['contract_version']}`  ",
        f"**Zero-Denominator Policy:** `{report['metadata']['zero_denominator_policy']}`  ",
        "",
        "---",
        "",
        "## 1. Standard Held-Out Test Split (25 Incidents, Stationary Operations)",
        "",
        "| Service | Horizon | Model | MAE [95% CI] | RMSE [95% CI] | WAPE (%) | Rel. Imprv vs Persist | 90% Cov (%) [95% CI] | Mean Width | Fallback Rate |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for svc in services:
        for h in horizons:
            for m in models:
                d = std[svc][h][m]
                b = d.get("bootstrap_ci95", {})
                mae_str = f"{d['mae']:.3f} [{b.get('mae_ci95', ('-','-'))[0]}-{b.get('mae_ci95', ('-','-'))[1]}]"
                rmse_str = f"{d['rmse']:.3f} [{b.get('rmse_ci95', ('-','-'))[0]}-{b.get('rmse_ci95', ('-','-'))[1]}]"
                cov_str = f"{d.get('interval_coverage_pct', '-')}% [{b.get('coverage_ci95', ('-','-'))[0]}-{b.get('coverage_ci95', ('-','-'))[1]}]"
                width_str = f"{d.get('interval_mean_width', '-')}"
                imprv_str = f"{d['rel_improvement_vs_persistence_pct']:+.1f}%" if m != "persistence" else "Baseline"
                fb_str = f"{d['fallback_activation_rate_pct']:.1f}%"

                lines.append(
                    f"| `{svc}` | H={h} ({int(h)*20}s) | **{m}** | {mae_str} | {rmse_str} | {d['wape_pct']}% | {imprv_str} | {cov_str} | {width_str} | {fb_str} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Locked Fresh Unseen Surge Split (25 Incidents, 2.2x - 2.85x Multiplier)",
        "",
        "| Service | Horizon | Model | MAE | RMSE | WAPE (%) | Rel. Imprv vs Persist | 90% Cov (%) | Mean Width | Fallback Rate |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for svc in services:
        for h in horizons:
            for m in models:
                d = lock[svc][h][m]
                imprv_str = f"{d['rel_improvement_vs_persistence_pct']:+.1f}%" if m != "persistence" else "Baseline"
                fb_str = f"{d['fallback_activation_rate_pct']:.1f}%"
                lines.append(
                    f"| `{svc}` | H={h} ({int(h)*20}s) | **{m}** | {d['mae']:.3f} | {d['rmse']:.3f} | {d['wape_pct']}% | {imprv_str} | {d.get('interval_coverage_pct', '-')}% | {d.get('interval_mean_width', '-')} | {fb_str} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Diagnostic Unseen Surge Split (25 Incidents, Examined in Task 1)",
        "",
        "| Service | Horizon | Model | MAE | RMSE | WAPE (%) | Rel. Imprv vs Persist | 90% Cov (%) | Mean Width | Fallback Rate |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for svc in services:
        for h in horizons:
            for m in models:
                d = diag[svc][h][m]
                imprv_str = f"{d['rel_improvement_vs_persistence_pct']:+.1f}%" if m != "persistence" else "Baseline"
                fb_str = f"{d['fallback_activation_rate_pct']:.1f}%"
                lines.append(
                    f"| `{svc}` | H={h} ({int(h)*20}s) | **{m}** | {d['mae']:.3f} | {d['rmse']:.3f} | {d['wape_pct']}% | {imprv_str} | {d.get('interval_coverage_pct', '-')}% | {d.get('interval_mean_width', '-')} | {fb_str} |"
                )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"[+] Saved evaluation report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BACCP Workload Forecasters")
    parser.add_argument("--dataset", type=Path, default=DATA_PATH)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    evaluate_all(dataset_path=args.dataset, weights_path=args.weights, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
