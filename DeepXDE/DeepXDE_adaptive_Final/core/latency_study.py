"""Compare controllable sources of adaptive-PINN boundary-event latency."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parameters import (
    LATENCY_EXPERIMENTS,
    LATENCY_RECOVERY_CONSECUTIVE_INSTANCES,
    LATENCY_RECOVERY_FRACTION,
    LatencyExperiment,
    config_for_latency_experiment,
    make_config,
)
from pinn_workflow import adapt_online, predict, rmse_by_time, train_baseline
from reference_data import generate_reference_dataset
from switch_solver import generate_switch_dataset


def _settling_metrics(
    times,
    errors,
    switch_tau,
    fraction,
    consecutive,
    earliest_update_tau=None,
):
    """Return a disturbance-settling metric on the causal prior error curve."""
    switch_index = int(np.searchsorted(times, switch_tau, side="left"))
    pre_start = max(0, switch_index - 4)
    pre_level = float(np.median(errors[pre_start:switch_index]))
    post_errors = errors[switch_index:]
    peak_index = switch_index + int(np.nanargmax(post_errors))
    peak_error = float(errors[peak_index])
    target = pre_level + (1.0 - fraction) * max(0.0, peak_error - pre_level)

    first_eligible_index = peak_index
    if earliest_update_tau is not None:
        # causal_prior is evaluated before the update at earliest_update_tau;
        # the first prediction affected by that update is the following sample.
        first_eligible_index = max(
            first_eligible_index,
            int(np.searchsorted(times, earliest_update_tau, side="right")),
        )

    recovery_index = None
    for index in range(first_eligible_index, len(times) - consecutive + 1):
        if np.all(errors[index : index + consecutive] <= target):
            recovery_index = index
            break

    if recovery_index is None:
        return {
            "pre_event_error": pre_level,
            "post_event_peak_error": peak_error,
            "recovery_target_error": target,
            "recovery_tau": None,
            "recovery_delay_tau": None,
            "recovery_delay_intervals": None,
        }
    return {
        "pre_event_error": pre_level,
        "post_event_peak_error": peak_error,
        "recovery_target_error": target,
        "recovery_tau": float(times[recovery_index]),
        "recovery_delay_tau": float(times[recovery_index] - switch_tau),
        "recovery_delay_intervals": int(recovery_index - switch_index),
    }


def _first_event_update(history, times, switch_tau):
    row = next(item for item in history if item["time_end"] >= switch_tau)
    changed_samples = int(
        np.count_nonzero((times >= switch_tau) & (times <= row["time_end"]))
    )
    return float(row["time_end"]), changed_samples


def _plot_results(results, output_dir, switch_tau):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(results)))
    for color, result in zip(colors, results):
        times = np.asarray(result["times"])
        errors = np.asarray(result["causal_prior_time_rmse"])
        axes[0].plot(times, errors, marker="o", markersize=2.5, color=color, label=result["name"])
    axes[0].axvline(switch_tau, color="tab:red", linestyle="--", label="boundary change")
    axes[0].set(
        title="Causal prediction error after the boundary event",
        xlabel="Dimensionless time",
        ylabel="Full-field temperature RMSE before update",
        yscale="log",
        xlim=(max(0.0, switch_tau - 10.0), None),
    )
    axes[0].legend(fontsize=7)

    labels = [result["name"] for result in results]
    recovery = [
        np.nan if result["recovery_delay_tau"] is None else result["recovery_delay_tau"]
        for result in results
    ]
    y = np.arange(len(results))
    axes[1].barh(y, np.nan_to_num(recovery, nan=0.0), color=colors)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set(
        title=f"Time to {LATENCY_RECOVERY_FRACTION:.0%} disturbance recovery",
        xlabel="Dimensionless time after boundary change (lower is faster)",
    )
    for row, value in enumerate(recovery):
        label = "not recovered" if np.isnan(value) else f"{value:g}"
        axes[1].text(0 if np.isnan(value) else value, row, f"  {label}", va="center", fontsize=8)

    effort = np.asarray([result["total_adaptive_iterations"] for result in results])
    post_rmse = np.asarray([result["post_switch_causal_rmse"] for result in results])
    for color, result, x_value, y_value in zip(colors, results, effort, post_rmse):
        axes[2].scatter(x_value, y_value, s=52, color=color)
        axes[2].annotate(
            result["name"],
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axes[2].set(
        title="Latency/accuracy versus update effort",
        xlabel="total adaptive Adam iterations",
        ylabel="Post-switch causal temperature RMSE",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Boundary-change latency study: every curve uses the model state available online"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "07_latency_parameter_sweep.png", dpi=190)
    plt.close(fig)


def run_study(profile: str, output_dir: Path, all_experiments: bool = False):
    base = make_config(profile)
    if profile == "smoke":
        # The full study is defined in physical tau units.  Smoke mode keeps a
        # short event while exercising the causal metrics and plotting path.
        experiments = (
            LatencyExperiment("smoke n=2", base.tau_final / 10.0, 2, adaptive_iterations_per_batch=10),
            LatencyExperiment("smoke n=1", base.tau_final / 10.0, 1, adaptive_iterations_per_batch=10),
        )
    elif all_experiments:
        experiments = LATENCY_EXPERIMENTS
    else:
        experiments = tuple(
            experiment
            for experiment in LATENCY_EXPERIMENTS
            if experiment.name in {"balanced", "low_latency"}
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    reference = generate_reference_dataset(base, output_dir)
    print("Training one shared offline baseline...")
    baseline = train_baseline(
        base,
        reference.field_points,
        reference.field_values,
        reference.times,
        verbose=0,
    )

    results = []
    for index, experiment in enumerate(experiments, start=1):
        cfg = config_for_latency_experiment(base, experiment)
        cfg.validate()
        print(f"[{index}/{len(experiments)}] {experiment.name}")
        switch_data = generate_switch_dataset(cfg)
        adaptive = adapt_online(
            cfg,
            baseline.network_state,
            switch_data.sensor_points,
            switch_data.sensor_values,
            switch_data.field_points,
            switch_data.field_values,
            switch_data.times,
            verbose=0,
        )
        baseline_prediction = predict(baseline.model, switch_data.field_points, base)
        baseline_errors = rmse_by_time(
            switch_data.field_points,
            switch_data.field_values,
            baseline_prediction,
            switch_data.times,
        )
        post_mask = switch_data.times >= switch_data.switch_tau
        first_update_tau, changed_samples = _first_event_update(
            adaptive.history, switch_data.times, switch_data.switch_tau
        )
        settling = _settling_metrics(
            switch_data.times,
            adaptive.causal_prior_time_rmse,
            switch_data.switch_tau,
            LATENCY_RECOVERY_FRACTION,
            LATENCY_RECOVERY_CONSECUTIVE_INSTANCES,
            earliest_update_tau=first_update_tau,
        )
        result = {
            "name": experiment.name,
            **asdict(experiment),
            "time_instances": cfg.time_instances,
            "sensor_count": len(cfg.sensor_x) * len(cfg.sensor_y),
            "switch_tau": switch_data.switch_tau,
            "first_event_update_tau": first_update_tau,
            "first_update_delay_tau": first_update_tau - switch_data.switch_tau,
            "changed_samples_before_first_update": changed_samples,
            "post_switch_offline_rmse": float(np.sqrt(np.mean(baseline_errors[post_mask] ** 2))),
            "post_switch_causal_rmse": float(
                np.sqrt(np.mean(adaptive.causal_prior_time_rmse[post_mask] ** 2))
            ),
            "post_switch_posterior_rmse": float(
                np.sqrt(np.mean(adaptive.causal_posterior_time_rmse[post_mask] ** 2))
            ),
            "total_adaptive_iterations": int(
                len(adaptive.history) * cfg.adaptive_iterations_per_batch
            ),
            **settling,
            "times": switch_data.times.tolist(),
            "causal_prior_time_rmse": adaptive.causal_prior_time_rmse.tolist(),
            "causal_posterior_time_rmse": adaptive.causal_posterior_time_rmse.tolist(),
        }
        results.append(result)

    ranked = sorted(
        results,
        key=lambda row: (
            float("inf") if row["recovery_delay_tau"] is None else row["recovery_delay_tau"],
            row["post_switch_causal_rmse"],
        ),
    )
    summary = {
        "metric_definition": {
            "causal_prior": "prediction before assimilating the current batch",
            "recovery_fraction": LATENCY_RECOVERY_FRACTION,
            "consecutive_instances": LATENCY_RECOVERY_CONSECUTIVE_INSTANCES,
        },
        "maintained_default": "balanced",
        "lowest_latency_configuration": ranked[0]["name"],
        "experiments": results,
    }
    (output_dir / "latency_parameter_sweep.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    excluded = {"times", "causal_prior_time_rmse", "causal_posterior_time_rmse", "sensor_x", "sensor_y"}
    columns = [key for key in results[0] if key not in excluded]
    with (output_dir / "latency_parameter_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in columns})

    _plot_results(results, output_dir, results[0]["switch_tau"])
    print("Maintained default: balanced")
    print(f"Lowest measured latency: {ranked[0]['name']}")
    print(f"Results: {output_dir.resolve()}")
    return summary


def recalculate_saved_results(output_dir: Path):
    """Refresh derived latency metrics/plot without retraining the networks."""
    json_path = output_dir / "latency_parameter_sweep.json"
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    results = summary["experiments"]
    for result in results:
        result.update(
            _settling_metrics(
                np.asarray(result["times"]),
                np.asarray(result["causal_prior_time_rmse"]),
                result["switch_tau"],
                LATENCY_RECOVERY_FRACTION,
                LATENCY_RECOVERY_CONSECUTIVE_INSTANCES,
                earliest_update_tau=result["first_event_update_tau"],
            )
        )
    ranked = sorted(
        results,
        key=lambda row: (
            float("inf") if row["recovery_delay_tau"] is None else row["recovery_delay_tau"],
            row["post_switch_causal_rmse"],
        ),
    )
    summary.pop("recommended_configuration", None)
    summary["maintained_default"] = "balanced"
    summary["lowest_latency_configuration"] = ranked[0]["name"]
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    excluded = {"times", "causal_prior_time_rmse", "causal_posterior_time_rmse", "sensor_x", "sensor_y"}
    columns = [key for key in results[0] if key not in excluded]
    with (output_dir / "latency_parameter_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in columns})
    _plot_results(results, output_dir, results[0]["switch_tau"])
    print("Maintained default: balanced")
    print(f"Lowest measured latency: {ranked[0]['name']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--recalculate-existing",
        action="store_true",
        help="recompute derived metrics and plots from an existing study JSON",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="also rerun historical sweep configurations; default compares maintained profiles only",
    )
    args = parser.parse_args()
    base = make_config(args.profile)
    output_dir = args.output_dir or base.output_dir / "latency_study"
    if args.recalculate_existing:
        recalculate_saved_results(output_dir)
    else:
        run_study(args.profile, output_dir, all_experiments=args.all_experiments)


if __name__ == "__main__":
    main()
