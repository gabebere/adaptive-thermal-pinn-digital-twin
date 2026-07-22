"""Stage 3: sample the validated analytical temperature reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytical_solution import make_field_points, make_log_time_grid, temperature
from parameters import WorkflowConfig


@dataclass
class ReferenceDataset:
    field_points: np.ndarray
    field_values: np.ndarray
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    sensor_points: np.ndarray
    sensor_values: np.ndarray


def save_reference_csv(dataset: ReferenceDataset, path: Path) -> None:
    """Export the full analytical reference field in a readable table."""
    points = np.asarray(dataset.field_points, dtype=float)
    values = np.asarray(dataset.field_values, dtype=float).reshape(-1, 1)
    np.savetxt(
        path,
        np.column_stack((points, values)),
        delimiter=",",
        header="x,y,tau,temperature",
        comments="",
        fmt="%.16g",
    )


def make_sensor_points(cfg: WorkflowConfig, times: np.ndarray) -> np.ndarray:
    xx, yy, tt = np.meshgrid(cfg.sensor_x, cfg.sensor_y, times, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel(), tt.ravel()))


def generate_reference_dataset(cfg: WorkflowConfig, output_dir: Path) -> ReferenceDataset:
    output_dir.mkdir(parents=True, exist_ok=True)
    times = make_log_time_grid(cfg.tau_final, cfg.time_instances)
    field_points, x, y = make_field_points(cfg.field_nx, cfg.field_ny, times)
    field_values = temperature(field_points, cfg.boundary_set, cfg.series_terms)
    sensor_points = make_sensor_points(cfg, times)
    sensor_values = temperature(sensor_points, cfg.boundary_set, cfg.series_terms)
    dataset = ReferenceDataset(
        field_points, field_values, times, x, y, sensor_points, sensor_values
    )
    np.savez_compressed(
        output_dir / "02_validated_reference_dataset.npz",
        field_points=field_points,
        field_values=field_values,
        times=times,
        x=x,
        y=y,
        sensor_points=sensor_points,
        sensor_values=sensor_values,
    )
    save_reference_csv(dataset, output_dir / "02_validated_reference_dataset.csv")

    # X-time slice through the center Y=0.5.
    slice_points = np.column_stack(
        (
            np.tile(x, len(times)),
            np.full(len(x) * len(times), 0.5),
            np.repeat(times, len(x)),
        )
    )
    slice_values = temperature(slice_points, cfg.boundary_set, cfg.series_terms).reshape(
        len(times), len(x)
    )
    fig, ax = plt.subplots(figsize=(8.5, 5))
    image = ax.contourf(
        x,
        times,
        slice_values,
        levels=np.linspace(float(np.min(slice_values)), float(np.max(slice_values)), 61),
        cmap="magma",
    )
    fig.colorbar(
        image,
        ax=ax,
        label=r"Dimensionless temperature at $Y=0.5$",
    )
    ax.set(
        xlabel=r"Dimensionless position, $X$",
        ylabel=r"Dimensionless time, $\tau$",
        title=f"Validated analytical centerline temperature (0 ≤ τ ≤ {cfg.tau_final:g})",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "02_reference_field.png", dpi=180)
    plt.close(fig)
    return dataset
