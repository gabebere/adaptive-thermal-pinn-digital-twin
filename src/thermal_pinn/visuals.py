from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from .config import ExperimentConfig
from .evaluate import EvaluationResult
from .reference import ReferenceSolution
from .sensors import SensorBatch


def _visual_dir(cfg: ExperimentConfig) -> Path:
    path = cfg.output_dir / "figures" / "visual_explainer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_physical_setup(cfg: ExperimentConfig) -> Path:
    """Schematic showing the wall, heat flux, coolant, and sensor locations."""
    path = _visual_dir(cfg) / "01_physical_setup.png"
    fig, ax = plt.subplots(figsize=(9, 3.2))

    wall = Rectangle((0.0, 0.25), 1.0, 0.5, facecolor="#d8dee9", edgecolor="#2e3440", linewidth=1.5)
    ax.add_patch(wall)
    ax.text(0.5, 0.5, "1D throat wall\nheat conduction", ha="center", va="center", fontsize=12)

    for y in np.linspace(0.32, 0.68, 4):
        ax.add_patch(
            FancyArrowPatch(
                (-0.22, y),
                (-0.01, y),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2,
                color="#bf616a",
            )
        )
    ax.text(-0.24, 0.82, "hot gas side\nimposed q_hot(t)", ha="left", va="center", color="#8f2832")

    for y in np.linspace(0.32, 0.68, 4):
        ax.add_patch(
            FancyArrowPatch(
                (1.22, y),
                (1.01, y),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2,
                color="#5e81ac",
            )
        )
    ax.text(1.02, 0.82, "regenerative coolant\nh_cool, T_cool", ha="left", va="center", color="#2f5f8f")

    for xh in cfg.sensor_x_hat:
        ax.plot(xh, 0.5, "o", markersize=9, color="#a3be8c", markeredgecolor="#2e3440")
        ax.plot([xh, xh], [0.23, 0.77], ":", color="#4c566a", linewidth=1)
    ax.text(0.5, 0.08, "sparse wall-temperature sensors", ha="center", va="center", color="#3b5f2b")

    ax.text(0.0, 0.16, "x=0", ha="center", fontsize=9)
    ax.text(1.0, 0.16, "x=L_wall", ha="center", fontsize=9)
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_boundary_forcing(reference: ReferenceSolution, cfg: ExperimentConfig) -> Path:
    """Show imposed heat flux and resulting hot/cool surface temperatures."""
    path = _visual_dir(cfg) / "02_boundary_forcing_and_surfaces.png"
    fig, ax1 = plt.subplots(figsize=(8, 4))
    q_mw = np.asarray(cfg.q_hot(reference.t), dtype=float) / 1.0e6
    ax1.plot(reference.t, q_mw, color="#bf616a", label="q_hot(t)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Hot-side heat flux (MW/m^2)", color="#bf616a")
    ax1.tick_params(axis="y", labelcolor="#bf616a")

    ax2 = ax1.twinx()
    ax2.plot(reference.t, reference.T[:, 0] - 273.15, color="#d08770", label="hot surface T")
    ax2.plot(reference.t, reference.T[:, -1] - 273.15, color="#5e81ac", label="coolant surface T")
    ax2.set_ylabel("Surface temperature (C)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.title("Boundary forcing drives the wall temperature response")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_temperature_evolution(reference: ReferenceSolution, cfg: ExperimentConfig) -> Path:
    """Draw many reference profiles to make the transient diffusion visible."""
    path = _visual_dir(cfg) / "03_reference_profiles_over_time.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_mm = 1.0e3 * reference.x
    indices = np.linspace(0, len(reference.t) - 1, 12, dtype=int)
    cmap = plt.get_cmap("viridis")
    for j, idx in enumerate(indices):
        color = cmap(j / max(1, len(indices) - 1))
        ax.plot(x_mm, reference.T[idx] - 273.15, color=color, label=f"{reference.t[idx]:.2f}s")
    ax.set_xlabel("Wall coordinate x (mm)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Reference solution: heat diffuses through the wall")
    ax.legend(title="time", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sensor_stream(reference: ReferenceSolution, batches: list[SensorBatch], cfg: ExperimentConfig) -> Path:
    """Show where and when the adaptive model receives data."""
    path = _visual_dir(cfg) / "04_sensor_streaming_map.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    extent = [0.0, 1.0e3 * cfg.L_wall, reference.t[0], reference.t[-1]]
    ax.imshow(reference.T - 273.15, aspect="auto", origin="lower", extent=extent, cmap="inferno", alpha=0.72)
    for batch in batches:
        ax.scatter(
            1.0e3 * cfg.L_wall * batch.x_hat.ravel(),
            cfg.t_final * batch.t_hat.ravel(),
            s=18,
            edgecolor="#2e3440",
            linewidth=0.35,
            label=f"window {batch.window_index}",
        )
    ax.set_xlabel("Wall coordinate x (mm)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Streaming sparse sensors over the reference field")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_error_heatmaps(
    reference: ReferenceSolution,
    baseline: EvaluationResult,
    adaptive: EvaluationResult,
    cfg: ExperimentConfig,
) -> Path:
    """Compare baseline/adaptive error fields and where adaptation helped."""
    path = _visual_dir(cfg) / "05_error_heatmaps.png"
    extent = [0.0, 1.0e3 * cfg.L_wall, reference.t[0], reference.t[-1]]
    vmax = max(np.max(np.abs(baseline.error)), np.max(np.abs(adaptive.error)))
    improvement = np.abs(baseline.error) - np.abs(adaptive.error)
    impmax = max(np.max(np.abs(improvement)), 1.0e-9)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    images = [
        axes[0].imshow(baseline.error, aspect="auto", origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax),
        axes[1].imshow(adaptive.error, aspect="auto", origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax),
        axes[2].imshow(improvement, aspect="auto", origin="lower", extent=extent, cmap="BrBG", vmin=-impmax, vmax=impmax),
    ]
    titles = ["Offline PINN error", "Adaptive PINN error", "|baseline error| - |adaptive error|"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("time (s)")
    fig.colorbar(images[0], ax=axes[:2], label="Prediction error (K)", shrink=0.88)
    fig.colorbar(images[2], ax=axes[2], label="Positive means adaptive helped (K)", shrink=0.88)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_visual_explanation_plots(
    reference: ReferenceSolution,
    baseline: EvaluationResult,
    adaptive: EvaluationResult,
    batches: list[SensorBatch],
    cfg: ExperimentConfig,
) -> list[Path]:
    return [
        plot_physical_setup(cfg),
        plot_boundary_forcing(reference, cfg),
        plot_temperature_evolution(reference, cfg),
        plot_sensor_stream(reference, batches, cfg),
        plot_error_heatmaps(reference, baseline, adaptive, cfg),
    ]
