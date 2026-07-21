"""Independent Crank-Nicolson reference for a mid-run boundary change."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from parameters import BoundarySet, WorkflowConfig, boundary_spatial_profile, initial_condition


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


def _boundary_time(
    tau: float, switch_tau: float, switched: bool, reset_clock: bool
) -> float:
    return tau - switch_tau if switched and reset_clock else tau


def _boundary_arrays(
    x: np.ndarray,
    y: np.ndarray,
    tau: float,
    switch_tau: float,
    before: BoundarySet,
    after: BoundarySet,
    reset_clock: bool,
):
    switched = tau >= switch_tau
    boundaries = after if switched else before
    local_tau = _boundary_time(tau, switch_tau, switched, reset_clock)
    d1, d2, d3, d4 = boundaries.decays
    a1, a2, a3, a4 = boundaries.amplitudes
    gx = boundary_spatial_profile(x)
    gy = boundary_spatial_profile(y)
    return (
        a1 * gy * np.exp(-d1 * local_tau),
        a2 * gy * np.exp(-d2 * local_tau),
        a3 * gx * np.exp(-d3 * local_tau),
        a4 * gx * np.exp(-d4 * local_tau),
    )


def _set_boundaries(field, left, right, bottom, top):
    field[:, 0] = left
    field[:, -1] = right
    field[0, :] = bottom
    field[-1, :] = top
    # Paper functions are zero at all four corners.
    field[0, 0] = field[0, -1] = field[-1, 0] = field[-1, -1] = 0.0


def _boundary_rhs(left, right, bottom, top, interior_n, h):
    rhs = np.zeros((interior_n, interior_n))
    rhs[:, 0] += left[1:-1] / h**2
    rhs[:, -1] += right[1:-1] / h**2
    rhs[0, :] += bottom[1:-1] / h**2
    rhs[-1, :] += top[1:-1] / h**2
    return rhs.ravel()


def _bilinear(field, locations, h):
    values = []
    for x_value, y_value in locations:
        fx, fy = x_value / h, y_value / h
        i0, j0 = int(np.floor(fx)), int(np.floor(fy))
        i1, j1 = min(i0 + 1, field.shape[1] - 1), min(j0 + 1, field.shape[0] - 1)
        wx, wy = fx - i0, fy - j0
        values.append(
            (1 - wx) * (1 - wy) * field[j0, i0]
            + wx * (1 - wy) * field[j0, i1]
            + (1 - wx) * wy * field[j1, i0]
            + wx * wy * field[j1, i1]
        )
    return np.asarray(values)


def generate_switch_dataset(cfg: WorkflowConfig) -> SwitchDataset:
    n = cfg.switch_solver_grid
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]
    times = np.linspace(0.0, cfg.tau_final, cfg.time_instances)
    switch_tau = cfg.switch_fraction * cfg.tau_final
    xx, yy = np.meshgrid(x, y)
    field = initial_condition(xx, yy)
    _set_boundaries(
        field,
        *_boundary_arrays(
            x,
            y,
            0.0,
            switch_tau,
            cfg.boundary_set,
            cfg.changed_boundary_set,
            cfg.reset_boundary_clock_at_switch,
        ),
    )

    interior_n = n - 2
    one = np.ones(interior_n)
    d2 = sparse.diags((one[:-1], -2 * one, one[:-1]), (-1, 0, 1)) / h**2
    identity = sparse.eye(interior_n)
    laplacian = sparse.kron(identity, d2) + sparse.kron(d2, identity)
    fields = [field.copy()]

    for start, stop in zip(times[:-1], times[1:]):
        sub_times = np.linspace(start, stop, cfg.switch_solver_substeps_per_interval + 1)
        for t0, t1 in zip(sub_times[:-1], sub_times[1:]):
            dt = t1 - t0
            b0 = _boundary_arrays(
                x,
                y,
                t0,
                switch_tau,
                cfg.boundary_set,
                cfg.changed_boundary_set,
                cfg.reset_boundary_clock_at_switch,
            )
            b1 = _boundary_arrays(
                x,
                y,
                t1,
                switch_tau,
                cfg.boundary_set,
                cfg.changed_boundary_set,
                cfg.reset_boundary_clock_at_switch,
            )
            forcing0 = _boundary_rhs(*b0, interior_n, h)
            forcing1 = _boundary_rhs(*b1, interior_n, h)
            lhs = sparse.eye(interior_n**2) - 0.5 * dt * laplacian
            rhs_matrix = sparse.eye(interior_n**2) + 0.5 * dt * laplacian
            old = field[1:-1, 1:-1].ravel()
            new = spsolve(lhs.tocsr(), rhs_matrix @ old + 0.5 * dt * (forcing0 + forcing1))
            field[1:-1, 1:-1] = new.reshape(interior_n, interior_n)
            _set_boundaries(field, *b1)
        fields.append(field.copy())

    field_values = np.stack(fields)
    t_mesh, y_mesh, x_mesh = np.meshgrid(times, y, x, indexing="ij")
    field_points = np.column_stack((x_mesh.ravel(), y_mesh.ravel(), t_mesh.ravel()))
    sensor_locations = np.array([(sx, sy) for sy in cfg.sensor_y for sx in cfg.sensor_x])
    sensor_values = np.vstack(
        [_bilinear(one_field, sensor_locations, h) for one_field in field_values]
    )
    sensor_points = np.vstack(
        [
            np.column_stack(
                (sensor_locations, np.full(len(sensor_locations), one_time))
            )
            for one_time in times
        ]
    )
    return SwitchDataset(
        field_points,
        field_values.reshape(-1, 1),
        sensor_points,
        sensor_values.reshape(-1, 1),
        times,
        x,
        y,
        switch_tau,
    )
