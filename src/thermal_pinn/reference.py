from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig


@dataclass
class ReferenceSolution:
    x: np.ndarray
    t: np.ndarray
    T: np.ndarray
    theta: np.ndarray


def _second_derivative_matrix(cfg: ExperimentConfig, dx: float) -> np.ndarray:
    n = cfg.nx
    A = np.zeros((n, n), dtype=np.float64)
    inv_dx2 = 1.0 / dx**2

    for i in range(1, n - 1):
        A[i, i - 1] = inv_dx2
        A[i, i] = -2.0 * inv_dx2
        A[i, i + 1] = inv_dx2

    # Hot side: -k dT/dx = q_hot(t), implemented with a ghost point.
    A[0, 0] = -2.0 * inv_dx2
    A[0, 1] = 2.0 * inv_dx2

    # Coolant side: -k dT/dx = h_cool * (T - T_cool).
    A[-1, -2] = 2.0 * inv_dx2
    A[-1, -1] = -2.0 * inv_dx2 - 2.0 * cfg.h_cool / (cfg.k * dx)
    return A


def _boundary_source(cfg: ExperimentConfig, t: float, dx: float) -> np.ndarray:
    b = np.zeros(cfg.nx, dtype=np.float64)
    b[0] = 2.0 * float(cfg.q_hot(t)) / (cfg.k * dx)
    b[-1] = 2.0 * cfg.h_cool * cfg.T_cool / (cfg.k * dx)
    return b


def generate_reference_solution(cfg: ExperimentConfig) -> ReferenceSolution:
    """Generate a Crank-Nicolson finite-difference reference solution."""
    x = np.linspace(0.0, cfg.L_wall, cfg.nx)
    t = np.linspace(0.0, cfg.t_final, cfg.nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    A = _second_derivative_matrix(cfg, dx)
    I = np.eye(cfg.nx, dtype=np.float64)
    lhs = I - 0.5 * dt * cfg.alpha * A
    rhs_matrix = I + 0.5 * dt * cfg.alpha * A

    T = np.empty((cfg.nt, cfg.nx), dtype=np.float64)
    T[0, :] = cfg.T0
    for n in range(cfg.nt - 1):
        b_now = _boundary_source(cfg, float(t[n]), dx)
        b_next = _boundary_source(cfg, float(t[n + 1]), dx)
        rhs = rhs_matrix @ T[n, :] + 0.5 * dt * cfg.alpha * (b_now + b_next)
        T[n + 1, :] = np.linalg.solve(lhs, rhs)

    theta = (T - cfg.T_cool) / cfg.delta_T
    return ReferenceSolution(x=x, t=t, T=T, theta=theta)
