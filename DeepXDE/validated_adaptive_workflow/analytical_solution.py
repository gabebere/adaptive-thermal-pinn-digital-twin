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
