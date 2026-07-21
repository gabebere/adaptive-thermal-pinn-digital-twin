"""Report-quality plots for the validated offline and adaptive workflows."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from parameters import WorkflowConfig


def _slice_matrix(points, values, times, x_values, y_value=0.5):
    matrix = np.empty((len(times), len(x_values)))
    for row, tau in enumerate(times):
        for column, x in enumerate(x_values):
            mask = (
                np.isclose(points[:, 0], x)
                & np.isclose(points[:, 1], y_value)
                & np.isclose(points[:, 2], tau)
            )
            if not np.any(mask):
                raise ValueError(f"Missing slice point X={x}, Y={y_value}, tau={tau}")
            matrix[row, column] = values[mask][0, 0]
    return matrix


def plot_offline_comparison(cfg, reference, baseline, output_dir: Path):
    truth = _slice_matrix(
        reference.field_points, reference.field_values, reference.times, reference.x
    )
    prediction = _slice_matrix(
        reference.field_points, baseline.field_prediction, reference.times, reference.x
    )
    error = np.abs(prediction - truth)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, values, title, cmap in (
        (axes[0], truth, "Validated numerical reference", "magma"),
        (axes[1], prediction, "Offline PINN", "magma"),
        (axes[2], error, "Offline absolute error", "viridis"),
    ):
        image = ax.pcolormesh(reference.x, reference.times, values, shading="auto", cmap=cmap)
        fig.colorbar(image, ax=ax)
        ax.set(xlabel="X", ylabel="tau", title=title)
        ax.set_yscale("symlog", linthresh=0.1)
    axes[3].plot(reference.times, baseline.time_rmse, "o-", markersize=3)
    axes[3].set(
        xlabel="tau",
        ylabel="full-field RMSE",
        title=f"Offline PINN error (global={baseline.field_rmse:.3g})",
        xscale="symlog",
    )
    axes[3].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "03_offline_pinn_validation.png", dpi=180)
    plt.close(fig)


def plot_streaming_map(cfg, reference, output_dir: Path):
    truth = _slice_matrix(
        reference.field_points, reference.field_values, reference.times, reference.x
    )
    fig, ax = plt.subplots(figsize=(9, 5.3))
    image = ax.pcolormesh(reference.x, reference.times, truth, shading="auto", cmap="magma")
    fig.colorbar(image, ax=ax, label="validated reference theta at Y=0.5")
    center_y_mask = np.isclose(reference.sensor_points[:, 1], 0.5)
    center_points = reference.sensor_points[center_y_mask]
    batches = [
        reference.times[start : start + cfg.batch_size_n]
        for start in range(0, len(reference.times), cfg.batch_size_n)
    ]
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(batches)))
    for index, batch_times in enumerate(batches, start=1):
        mask = np.isin(np.round(center_points[:, 2], 12), np.round(batch_times, 12))
        ax.scatter(
            center_points[mask, 0],
            center_points[mask, 2],
            s=22,
            color=colors[index - 1],
            edgecolor="white",
            linewidth=0.3,
            label=f"batch {index}",
        )
    ax.set(
        xlabel="X at sensor row Y=0.5",
        ylabel="tau",
        title=f"Streaming numerical inputs: n={cfg.batch_size_n} time instances per update",
    )
    ax.set_yscale("symlog", linthresh=0.1)
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "04_streaming_numerical_inputs.png", dpi=180)
    plt.close(fig)


def plot_adaptive_comparison(cfg, reference, baseline, adaptive, output_dir: Path):
    improvement = baseline.time_rmse - adaptive.time_rmse
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    axes[0].plot(reference.times, baseline.time_rmse, "o-", markersize=3)
    axes[0].set(title="Offline PINN RMSE", ylabel="full-field RMSE")
    axes[1].plot(reference.times, adaptive.time_rmse, "o-", markersize=3)
    axes[1].set(title="Adaptive PINN RMSE")
    axes[2].axhline(0.0, color="0.5", linewidth=0.8)
    axes[2].plot(reference.times, improvement, "o-", markersize=3)
    axes[2].set(title="Offline RMSE - adaptive RMSE", ylabel="positive means adaptation helped")
    axes[3].plot(
        reference.times,
        adaptive.causal_prior_time_rmse,
        "o-",
        markersize=3,
        label="prediction before update",
    )
    axes[3].plot(
        reference.times,
        adaptive.causal_posterior_time_rmse,
        "o-",
        markersize=3,
        label="fit after update",
    )
    axes[3].set(title="Causal online full-field RMSE", ylabel="full-field RMSE")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("tau")
        ax.set_xscale("symlog")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"Online adaptation with n={cfg.batch_size_n}: global RMSE "
        f"{baseline.field_rmse:.3g} -> {adaptive.field_rmse:.3g}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "05_adaptive_pinn_error.png", dpi=180)
    plt.close(fig)


def plot_switch_test(cfg, switch_data, baseline_prediction, adaptive, output_dir: Path):
    baseline_time_rmse = []
    for tau in switch_data.times:
        mask = np.isclose(switch_data.field_points[:, 2], tau)
        baseline_time_rmse.append(
            np.sqrt(
                np.mean(
                    (
                        baseline_prediction[mask]
                        - switch_data.field_values[mask]
                    )
                    ** 2
                )
            )
        )
    baseline_time_rmse = np.asarray(baseline_time_rmse)
    improvement = baseline_time_rmse - adaptive.causal_prior_time_rmse
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    axes[0].plot(switch_data.times, baseline_time_rmse, label="offline PINN")
    axes[0].plot(
        switch_data.times,
        adaptive.causal_prior_time_rmse,
        label="adaptive prediction before update",
    )
    axes[0].plot(
        switch_data.times,
        adaptive.causal_posterior_time_rmse,
        label="adaptive fit after update",
        alpha=0.75,
    )
    axes[0].set(title="Causal full-field error through event", ylabel="RMSE")
    axes[0].legend()
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].plot(switch_data.times, improvement)
    axes[1].set(
        title="Offline RMSE - next adaptive prediction",
        ylabel="positive means adaptation helped",
    )
    batch_ends = [row["time_end"] for row in adaptive.history]
    axes[2].plot(
        batch_ends,
        [row["before_batch_rmse"] for row in adaptive.history],
        "o-",
        label="before update",
    )
    axes[2].plot(
        batch_ends,
        [row["after_batch_rmse"] for row in adaptive.history],
        "o-",
        label="after update",
    )
    axes[2].set(title="Sensor error seen by online updates", ylabel="sensor RMSE")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvline(switch_data.switch_tau, color="tab:red", linestyle="--", label="boundary change")
        ax.set_xlabel("tau")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"Unknown boundary change: {cfg.boundary_set.name} to "
        f"{cfg.changed_boundary_set.name} at tau={switch_data.switch_tau:g}; "
        "each point uses the model state available at that time"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "06_boundary_change_response.png", dpi=180)
    plt.close(fig)


def save_metrics(cfg, literature, baseline, adaptive, switch_baseline_rmse, switch_adaptive, output_dir):
    metrics = {
        "configuration": {
            "tau_final": cfg.tau_final,
            "time_instances": cfg.time_instances,
            "batch_size_n": cfg.batch_size_n,
            "sensor_count": len(cfg.sensor_x) * len(cfg.sensor_y),
            "boundary_set": cfg.boundary_set.name,
            "changed_boundary_set": cfg.changed_boundary_set.name,
            "switch_fraction": cfg.switch_fraction,
        },
        "literature_validation": literature,
        "offline_pinn_global_rmse": baseline.field_rmse,
        "adaptive_pinn_global_rmse": adaptive.field_rmse,
        "adaptive_update_history": adaptive.history,
        "switch_offline_global_rmse": switch_baseline_rmse,
        "switch_adaptive_global_rmse": switch_adaptive.field_rmse,
        "switch_update_history": switch_adaptive.history,
    }
    (output_dir / "workflow_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
