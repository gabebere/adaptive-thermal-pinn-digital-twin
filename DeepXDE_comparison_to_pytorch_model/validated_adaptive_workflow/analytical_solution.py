"""General series solution for the paper's parabolic 2-D heat example."""

from __future__ import annotations

import numpy as np

from parameters import BoundarySet, boundary_spatial_profile, initial_condition


def _g(coordinate):
    return boundary_spatial_profile(coordinate)


def boundary_values(points: np.ndarray, boundaries: BoundarySet) -> dict[str, np.ndarray]:
    """Evaluate all four Dirichlet functions at point times."""
    points = np.asarray(points, dtype=float)
    x, y, tau = points[:, 0], points[:, 1], points[:, 2]
    d1, d2, d3, d4 = boundaries.decays
    a1, a2, a3, a4 = boundaries.amplitudes
    return {
        "left": a1 * _g(y) * np.exp(-d1 * tau),
        "right": a2 * _g(y) * np.exp(-d2 * tau),
        "bottom": a3 * _g(x) * np.exp(-d3 * tau),
        "top": a4 * _g(x) * np.exp(-d4 * tau),
    }


def _convolution(decay: float, eigenvalue: float, tau: np.ndarray) -> np.ndarray:
    """Integral of exp(-eigenvalue*(tau-s))*exp(-decay*s) ds."""
    if np.isclose(decay, eigenvalue):
        return tau * np.exp(-eigenvalue * tau)
    return (np.exp(-decay * tau) - np.exp(-eigenvalue * tau)) / (
        eigenvalue - decay
    )


def temperature(
    points: np.ndarray, boundaries: BoundarySet, terms: int = 20
) -> np.ndarray:
    """Evaluate theta(X,Y,tau) using a boundary lifting and sine series.

    This is the general-X,Y form of the method used for paper Eqs. (108)-(111).
    It satisfies the four space-time Dirichlet conditions exactly in the
    infinite-series limit and uses a homogeneous double-sine expansion for the
    lifted transient.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3) with columns X,Y,tau")
    if terms < 1 or np.any(points[:, :2] < 0) or np.any(points[:, :2] > 1):
        raise ValueError("invalid series terms or spatial point outside [0,1]^2")
    if np.any(points[:, 2] < 0):
        raise ValueError("tau must be nonnegative")

    x, y, tau = points[:, 0], points[:, 1], points[:, 2]
    d1, d2, d3, d4 = boundaries.decays
    a1, a2, a3, a4 = boundaries.amplitudes

    # Boundary lifting B. At tau=0 and unit amplitudes it equals the paper IC.
    result = (
        ((1 - x) * a1 * np.exp(-d1 * tau) + x * a2 * np.exp(-d2 * tau))
        * _g(y)
        + ((1 - y) * a3 * np.exp(-d3 * tau) + y * a4 * np.exp(-d4 * tau))
        * _g(x)
    )

    pi = np.pi
    for m in range(1, terms + 1):
        sin_mx = np.sin(m * pi * x)
        parity_m = (-1) ** m
        g_m = 2.0 * (1.0 - parity_m) / (m**3 * pi**3)
        one_m = (1.0 - parity_m) / (m * pi)
        for n in range(1, terms + 1):
            parity_n = (-1) ** n
            g_n = 2.0 * (1.0 - parity_n) / (n**3 * pi**3)
            one_n = (1.0 - parity_n) / (n * pi)
            eigenvalue = (m * m + n * n) * pi**2

            # Projection of Delta(B)-B_tau. The -2 term comes from
            # d2/dY2(Y-Y^2)=-2 (and likewise in X); it is not multiplied
            # by the parabolic g function.
            c1 = 4.0 * a1 * (d1 * g_n - 2.0 * one_n) / (m * pi)
            c2 = (
                4.0
                * a2
                * (d2 * g_n - 2.0 * one_n)
                * (-1) ** (m + 1)
                / (m * pi)
            )
            c3 = 4.0 * a3 * (d3 * g_m - 2.0 * one_m) / (n * pi)
            c4 = (
                4.0
                * a4
                * (d4 * g_m - 2.0 * one_m)
                * (-1) ** (n + 1)
                / (n * pi)
            )
            modal_time = (
                c1 * _convolution(d1, eigenvalue, tau)
                + c2 * _convolution(d2, eigenvalue, tau)
                + c3 * _convolution(d3, eigenvalue, tau)
                + c4 * _convolution(d4, eigenvalue, tau)
            )
            result += modal_time * sin_mx * np.sin(n * pi * y)
    return result[:, None]


def switched_temperature(
    points: np.ndarray,
    before: BoundarySet,
    after: BoundarySet,
    switch_tau: float,
    terms: int = 20,
    reset_boundary_clock: bool = True,
    quadrature_order: int | None = None,
) -> np.ndarray:
    """Piecewise analytical temperature for an unexpected boundary change.

    Before ``switch_tau`` this is the verified paper-series solution for
    ``before``. After the event, the same series equations are evaluated for
    ``after`` and corrected by a homogeneous double-sine expansion. The
    correction carries the pre-event interior field into the new problem, so
    the plate temperature does not reset when the boundary functions change.

    A discontinuous boundary event is permitted: the interior is continuous
    at the event, while points on an affected edge immediately take the new
    Dirichlet value.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3) with columns X,Y,tau")
    if switch_tau <= 0.0 or np.any(points[:, 2] < 0.0):
        raise ValueError("switch_tau must be positive and point times nonnegative")

    result = np.empty((len(points), 1), dtype=float)
    before_mask = points[:, 2] < switch_tau
    if np.any(before_mask):
        result[before_mask] = temperature(points[before_mask], before, terms)

    after_mask = ~before_mask
    if not np.any(after_mask):
        return result

    # Project the difference between the true pre-event interior and the
    # initial field implied by the new boundary problem onto homogeneous modes.
    order = quadrature_order or max(48, 2 * terms + 8)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    coordinates = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    switch_points = np.column_stack(
        (xx.ravel(), yy.ravel(), np.full(xx.size, switch_tau))
    )
    pre_event = temperature(switch_points, before, terms).reshape(order, order)

    new_initial_time = 0.0 if reset_boundary_clock else switch_tau
    new_initial_points = switch_points.copy()
    new_initial_points[:, 2] = new_initial_time
    new_initial = temperature(new_initial_points, after, terms).reshape(order, order)
    difference = pre_event - new_initial

    modes = np.arange(1, terms + 1, dtype=float)
    sine = np.sin(np.pi * coordinates[:, None] * modes[None, :])
    weighted_difference = difference * weights[:, None] * weights[None, :]
    coefficients = 4.0 * sine.T @ weighted_difference.T @ sine

    event_points = points[after_mask].copy()
    elapsed = event_points[:, 2] - switch_tau
    event_points[:, 2] = (
        elapsed if reset_boundary_clock else event_points[:, 2]
    )
    event_values = temperature(event_points, after, terms)[:, 0]
    sine_x = np.sin(np.pi * event_points[:, 0:1] * modes[None, :])
    sine_y = np.sin(np.pi * event_points[:, 1:2] * modes[None, :])
    for m_index, m in enumerate(modes):
        for n_index, n in enumerate(modes):
            decay = np.exp(-np.pi**2 * (m * m + n * n) * elapsed)
            event_values += (
                coefficients[m_index, n_index]
                * sine_x[:, m_index]
                * sine_y[:, n_index]
                * decay
            )

    # At the event instant, preserve the interior exactly instead of exposing
    # finite-series Gibbs error from the discontinuous edge jump. Dirichlet
    # boundary points retain the new boundary values from ``temperature``.
    spatially_interior = np.all(
        (event_points[:, :2] > 0.0) & (event_points[:, :2] < 1.0), axis=1
    )
    event_interior = np.isclose(elapsed, 0.0) & spatially_interior
    if np.any(event_interior):
        pre_event_points = points[after_mask][event_interior].copy()
        pre_event_points[:, 2] = switch_tau
        event_values[event_interior] = temperature(
            pre_event_points, before, terms
        )[:, 0]
    result[after_mask, 0] = event_values
    return result


def make_log_time_grid(tau_final: float, count: int) -> np.ndarray:
    """Resolve the fast early transient while still reaching large tau."""
    scaled = np.linspace(0.0, 1.0, count)
    return np.expm1(scaled * np.log1p(tau_final))


def make_field_points(nx: int, ny: int, times: np.ndarray):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy, tt = np.meshgrid(x, y, times, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), tt.ravel()))
    return points, x, y
