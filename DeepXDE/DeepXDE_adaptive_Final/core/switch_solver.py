"""Analytical reference for an unexpected mid-run boundary change.

The module name is retained for compatibility with existing entry points. The
reference is no longer a finite-difference solve: both sides of the event use
the paper's reproduced series equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analytical_solution import make_field_points, switched_temperature
from parameters import WorkflowConfig


@dataclass
class SwitchDataset:
    field_points: np.ndarray
    field_values: np.ndarray
    sensor_points: np.ndarray
    sensor_values: np.ndarray
    times: np.ndarray
    x: np.ndarray
    y: np.ndarray
    switch_tau: float


def save_switch_csv(dataset: SwitchDataset, path: Path) -> None:
    """Export the analytical boundary-change reference as a readable table."""
    points = np.asarray(dataset.field_points, dtype=float)
    values = np.asarray(dataset.field_values, dtype=float).reshape(-1, 1)
    is_post_switch = (points[:, 2] >= dataset.switch_tau).astype(int).reshape(-1, 1)
    np.savetxt(
        path,
        np.column_stack((points, values, is_post_switch)),
        delimiter=",",
        header="x,y,tau,temperature,is_post_switch",
        comments="",
        fmt=("%.10g", "%.10g", "%.10g", "%.16g", "%d"),
    )


def _sensor_points(cfg: WorkflowConfig, times: np.ndarray) -> np.ndarray:
    locations = np.asarray(
        [(x, y) for y in cfg.sensor_y for x in cfg.sensor_x], dtype=float
    )
    return np.vstack(
        [
            np.column_stack((locations, np.full(len(locations), time)))
            for time in times
        ]
    )


def generate_switch_dataset(cfg: WorkflowConfig) -> SwitchDataset:
    """Generate a continuous-interior, piecewise analytical event reference."""
    times = np.linspace(0.0, cfg.tau_final, cfg.time_instances)
    switch_tau = cfg.switch_fraction * cfg.tau_final
    field_points, x, y = make_field_points(cfg.field_nx, cfg.field_ny, times)
    sensor_points = _sensor_points(cfg, times)

    evaluation = dict(
        before=cfg.boundary_set,
        after=cfg.changed_boundary_set,
        switch_tau=switch_tau,
        terms=cfg.series_terms,
        reset_boundary_clock=cfg.reset_boundary_clock_at_switch,
    )
    field_values = switched_temperature(field_points, **evaluation)
    sensor_values = switched_temperature(sensor_points, **evaluation)
    if cfg.sensor_noise_std > 0.0:
        rng = np.random.default_rng(cfg.seed + 101)
        sensor_values = sensor_values + rng.normal(
            0.0, cfg.sensor_noise_std, size=sensor_values.shape
        )
    return SwitchDataset(
        field_points=field_points,
        field_values=field_values,
        sensor_points=sensor_points,
        sensor_values=sensor_values,
        times=times,
        x=x,
        y=y,
        switch_tau=switch_tau,
    )
