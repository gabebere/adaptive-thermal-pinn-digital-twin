from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .evaluate import EvaluationResult
from .reference import ReferenceSolution


def _figure_dir(cfg: ExperimentConfig) -> Path:
    path = cfg.output_dir / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _x_mm(reference: ReferenceSolution, cfg: ExperimentConfig) -> np.ndarray:
    return 1.0e3 * reference.x


def plot_reference_field(reference: ReferenceSolution, cfg: ExperimentConfig) -> Path:
    fig_dir = _figure_dir(cfg)
    path = fig_dir / "reference_temperature_field.png"
    plt.figure(figsize=(7, 4))
    extent = [0.0, 1.0e3 * cfg.L_wall, reference.t[0], reference.t[-1]]
    plt.imshow(reference.T - 273.15, aspect="auto", origin="lower", extent=extent, cmap="inferno")
    plt.colorbar(label="Temperature (C)")
    plt.xlabel("Wall coordinate x (mm)")
    plt.ylabel("Time (s)")
    plt.title("Finite-difference reference temperature")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_profiles(
    reference: ReferenceSolution,
    result: EvaluationResult,
    cfg: ExperimentConfig,
    name: str,
) -> Path:
    fig_dir = _figure_dir(cfg)
    path = fig_dir / f"{name}_profiles.png"
    plt.figure(figsize=(7, 4))
    for t_value in cfg.selected_times:
        idx = int(np.argmin(np.abs(reference.t - t_value)))
        label_t = f"t={reference.t[idx]:.2f}s"
        plt.plot(_x_mm(reference, cfg), reference.T[idx] - 273.15, "-", label=f"ref {label_t}")
        plt.plot(_x_mm(reference, cfg), result.T_pred[idx] - 273.15, "--", label=f"{name} {label_t}")
    plt.xlabel("Wall coordinate x (mm)")
    plt.ylabel("Temperature (C)")
    plt.title(f"{name.capitalize()} PINN vs reference")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_error_over_time(
    reference: ReferenceSolution,
    baseline: EvaluationResult,
    adaptive: EvaluationResult,
    cfg: ExperimentConfig,
) -> Path:
    fig_dir = _figure_dir(cfg)
    path = fig_dir / "relative_l2_error_over_time.png"
    plt.figure(figsize=(7, 4))
    plt.semilogy(reference.t, baseline.relative_l2_time, label="offline PINN")
    plt.semilogy(reference.t, adaptive.relative_l2_time, label="adaptive PINN")
    plt.xlabel("Time (s)")
    plt.ylabel("Relative L2 error of temperature rise")
    plt.title("Error over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_loss_curves(histories: dict[str, list[dict[str, float]]], cfg: ExperimentConfig) -> Path:
    fig_dir = _figure_dir(cfg)
    path = fig_dir / "loss_curves.png"
    plt.figure(figsize=(7, 4))
    offset = 0
    for label, history in histories.items():
        if not history:
            continue
        epochs = np.asarray([row["epoch"] + offset for row in history])
        for key in ("total", "pde", "bc", "ic", "sensor"):
            values = np.asarray([max(row[key], 1.0e-14) for row in history])
            if key == "sensor" and np.allclose(values, 0.0):
                continue
            plt.semilogy(epochs, values, label=f"{label} {key}")
        offset = int(epochs[-1])
    plt.xlabel("Logged epoch index")
    plt.ylabel("MSE loss")
    plt.title("Training loss components")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_noise_comparison(noise_summary: list[dict[str, float]], cfg: ExperimentConfig) -> Path:
    fig_dir = _figure_dir(cfg)
    path = fig_dir / "sensor_noise_comparison.png"
    levels = [100.0 * row["noise_level"] for row in noise_summary]
    l2 = [row["relative_l2_global"] for row in noise_summary]
    hot = [row["hot_side_max_abs_error"] for row in noise_summary]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(levels, l2, "o-", color="tab:blue", label="global relative L2")
    ax1.set_xlabel("Sensor noise (% of delta_T)")
    ax1.set_ylabel("Global relative L2", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(levels, hot, "s--", color="tab:red", label="hot-side max error")
    ax2.set_ylabel("Hot-side max error (K)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    plt.title("Adaptive PINN robustness to noisy sensors")
    fig.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_plots(
    reference: ReferenceSolution,
    baseline: EvaluationResult,
    adaptive: EvaluationResult,
    histories: dict[str, list[dict[str, float]]],
    noise_summary: list[dict[str, float]],
    cfg: ExperimentConfig,
) -> list[Path]:
    return [
        plot_reference_field(reference, cfg),
        plot_profiles(reference, baseline, cfg, "baseline"),
        plot_profiles(reference, adaptive, cfg, "adaptive"),
        plot_error_over_time(reference, baseline, adaptive, cfg),
        plot_loss_curves(histories, cfg),
        plot_noise_comparison(noise_summary, cfg),
    ]
