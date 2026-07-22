"""Report-quality plots for the validated offline and adaptive workflows."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from parameters import WorkflowConfig


TEMPERATURE_LABEL = "Dimensionless temperature"
TIME_LABEL = r"Dimensionless time, $\tau$"


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
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    temperature_min = float(min(np.min(truth), np.min(prediction)))
    temperature_max = float(max(np.max(truth), np.max(prediction)))
    levels = np.linspace(temperature_min, temperature_max, 61)
    for ax, values, title in (
        (axes[0], truth, "Analytical reference"),
        (axes[1], prediction, "Offline PINN prediction"),
    ):
        image = ax.contourf(
            reference.x,
            reference.times,
            values,
            levels=levels,
            cmap="magma",
            extend="both",
        )
        fig.colorbar(image, ax=ax, label=TEMPERATURE_LABEL)
        ax.set(xlabel=r"Dimensionless position, $X$", ylabel=TIME_LABEL, title=title)
    error_image = axes[2].contourf(
        reference.x,
        reference.times,
        error,
        levels=np.linspace(0.0, float(np.max(error)), 61),
        cmap="viridis",
        extend="max",
    )
    fig.colorbar(error_image, ax=axes[2], label="Absolute temperature error")
    axes[2].set(
        xlabel=r"Dimensionless position, $X$",
        ylabel=TIME_LABEL,
        title="Absolute prediction error",
    )
    axes[3].plot(reference.times, baseline.time_rmse, "o-", markersize=3)
    axes[3].set(
        xlabel=TIME_LABEL,
        ylabel="Full-field temperature RMSE",
        title=f"Offline PINN error (global={baseline.field_rmse:.3g})",
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
    image = ax.contourf(
        reference.x,
        reference.times,
        truth,
        levels=np.linspace(float(np.min(truth)), float(np.max(truth)), 61),
        cmap="magma",
    )
    fig.colorbar(image, ax=ax, label=TEMPERATURE_LABEL + r" at $Y=0.5$")
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
        xlabel=r"Sensor position $X$ along the center row ($Y=0.5$)",
        ylabel=TIME_LABEL,
        title=f"Sensor observations supplied to each update (n={cfg.batch_size_n} times)",
    )
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "04_streaming_numerical_inputs.png", dpi=180)
    plt.close(fig)


def plot_adaptive_comparison(cfg, reference, baseline, adaptive, output_dir: Path):
    improvement = baseline.time_rmse - adaptive.causal_prior_time_rmse
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    axes[0].plot(reference.times, baseline.time_rmse, "o-", markersize=3, label="offline PINN")
    axes[0].plot(
        reference.times,
        adaptive.causal_prior_time_rmse,
        "o-",
        markersize=3,
        label="adaptive PINN before seeing current batch",
    )
    axes[0].set(title="True online prediction error", ylabel="Full-field temperature RMSE")
    axes[0].legend(fontsize=7)
    axes[1].plot(
        reference.times,
        adaptive.causal_prior_time_rmse,
        "o-",
        markersize=3,
        label="before update",
    )
    axes[1].plot(
        reference.times,
        adaptive.causal_posterior_time_rmse,
        "o-",
        markersize=3,
        label="after update",
    )
    axes[1].set(title="Effect of assimilating the current batch")
    axes[1].legend(fontsize=7)
    axes[2].axhline(0.0, color="0.5", linewidth=0.8)
    axes[2].plot(reference.times, improvement, "o-", markersize=3)
    axes[2].set(
        title="Online improvement over offline PINN",
        ylabel="RMSE reduction (positive is better)",
    )
    batch_ends = [row["time_end"] for row in adaptive.history]
    axes[3].plot(
        batch_ends,
        [row["before_batch_rmse"] for row in adaptive.history],
        "o-",
        markersize=3,
        label="before update",
    )
    axes[3].plot(
        batch_ends,
        [row["after_batch_rmse"] for row in adaptive.history],
        "o-",
        markersize=3,
        label="after update",
    )
    axes[3].set(title="Error at newly revealed sensors", ylabel="Sensor temperature RMSE")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel(TIME_LABEL)
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"Causal online adaptation with n={cfg.batch_size_n} time instances per update"
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
    axes[0].set(title="Prediction error available in real time", ylabel="Full-field temperature RMSE")
    axes[0].legend()
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].plot(switch_data.times, improvement)
    axes[1].set(
        title="Adaptive improvement over offline PINN",
        ylabel="RMSE reduction (positive is better)",
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
    axes[2].set(title="Error at newly revealed sensors", ylabel="Sensor temperature RMSE")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvline(switch_data.switch_tau, color="tab:red", linestyle="--", label="boundary change")
        ax.set_xlabel(TIME_LABEL)
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
            "sensor_x": cfg.sensor_x,
            "sensor_y": cfg.sensor_y,
            "observation_window_batches": cfg.observation_window_batches,
            "hidden_layers": cfg.hidden_layers,
            "activation": cfg.activation,
            "initializer": cfg.initializer,
            "baseline_iterations": cfg.baseline_iterations,
            "adaptive_iterations_per_batch": cfg.adaptive_iterations_per_batch,
            "baseline_learning_rate": cfg.baseline_learning_rate,
            "adaptive_learning_rate": cfg.adaptive_learning_rate,
            "pde_loss_weight": cfg.pde_loss_weight,
            "boundary_loss_weight": cfg.boundary_loss_weight,
            "initial_loss_weight": cfg.initial_loss_weight,
            "data_loss_weight": cfg.data_loss_weight,
            "num_domain": cfg.num_domain,
            "num_boundary": cfg.num_boundary,
            "num_initial": cfg.num_initial,
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
