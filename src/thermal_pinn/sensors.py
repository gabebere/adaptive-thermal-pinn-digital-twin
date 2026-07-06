from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .reference import ReferenceSolution


@dataclass
class SensorBatch:
    x_hat: np.ndarray
    t_hat: np.ndarray
    theta: np.ndarray
    noise_level: float
    window_index: int


def simulate_sensor_data(
    reference: ReferenceSolution,
    cfg: ExperimentConfig,
    noise_level: float | None = None,
) -> list[SensorBatch]:
    """Create streaming sparse sensor measurements from the reference solution."""
    rng = np.random.default_rng(cfg.seed + 101)
    noise = cfg.sensor_noise if noise_level is None else noise_level

    x_hat_grid = reference.x / cfg.L_wall
    sensor_indices = [int(np.argmin(np.abs(x_hat_grid - xh))) for xh in cfg.sensor_x_hat]
    time_indices = np.arange(1, cfg.nt, max(1, cfg.sensor_stride))
    windows = np.array_split(time_indices, cfg.sensor_windows)

    batches: list[SensorBatch] = []
    for window_index, window in enumerate(windows):
        xs, ts, values = [], [], []
        for ti in window:
            for xi in sensor_indices:
                xs.append(x_hat_grid[xi])
                ts.append(reference.t[ti] / cfg.t_final)
                T_value = reference.T[ti, xi]
                noisy_T = T_value + rng.normal(0.0, noise * cfg.delta_T)
                values.append((noisy_T - cfg.T_cool) / cfg.delta_T)
        batches.append(
            SensorBatch(
                x_hat=np.asarray(xs, dtype=np.float32).reshape(-1, 1),
                t_hat=np.asarray(ts, dtype=np.float32).reshape(-1, 1),
                theta=np.asarray(values, dtype=np.float32).reshape(-1, 1),
                noise_level=noise,
                window_index=window_index,
            )
        )
    return batches
