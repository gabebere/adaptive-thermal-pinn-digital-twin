from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for the 1D throat-wall thermal PINN experiment."""

    # Output and reproducibility.
    output_dir: Path = Path("outputs")
    seed: int = 7
    device: str = "cpu"

    # Physical parameters for a copper-like wall.
    L_wall: float = 5.0e-3
    rho: float = 8960.0
    cp: float = 385.0
    k: float = 385.0
    h_cool: float = 2.5e4
    T_cool: float = 300.0
    T0: float = 300.0
    t_final: float = 1.0
    q_base: float = 4.0e6
    q_amp: float = 1.2e6
    q_period: float = 0.75
    delta_T: float = 100.0

    # Reference finite-difference grid.
    nx: int = 81
    nt: int = 301

    # PINN architecture and sampling.
    hidden_layers: int = 3
    hidden_units: int = 48
    activation: str = "tanh"
    learning_rate: float = 2.0e-3
    adaptive_learning_rate: float = 5.0e-4
    baseline_epochs: int = 1200
    update_epochs: int = 160
    n_pde: int = 1536
    n_bc: int = 256
    n_ic: int = 256

    # Loss weights.
    w_pde: float = 1.0
    w_bc: float = 2.0
    w_ic: float = 5.0
    w_sensor: float = 10.0

    # Adaptive sensor setup.
    sensor_x_hat: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    sensor_windows: int = 5
    sensor_stride: int = 5
    sensor_noise: float = 0.01
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.05)

    # Plot and evaluation settings.
    selected_times: tuple[float, ...] = (0.2, 0.5, 1.0)
    main_noise_level: float = 0.01
    eval_chunk_size: int = 8192

    @property
    def alpha(self) -> float:
        return self.k / (self.rho * self.cp)

    @property
    def beta(self) -> float:
        return self.alpha * self.t_final / (self.L_wall**2)

    @property
    def biot(self) -> float:
        return self.h_cool * self.L_wall / self.k

    def q_hot(self, t: np.ndarray | float) -> np.ndarray | float:
        """Smooth imposed hot-gas heat flux in W/m^2."""
        return self.q_base + self.q_amp * np.sin(2.0 * np.pi * np.asarray(t) / self.q_period)


def make_config(mode: str = "smoke", output_dir: str | Path | None = None) -> ExperimentConfig:
    """Return a runnable default configuration.

    ``smoke`` is intended for quick laptop verification. ``full`` increases
    training and sampling for report-quality plots.
    """
    cfg = ExperimentConfig()
    if output_dir is not None:
        cfg.output_dir = Path(output_dir)

    if mode == "smoke":
        cfg.nx = 41
        cfg.nt = 121
        cfg.hidden_layers = 2
        cfg.hidden_units = 32
        cfg.baseline_epochs = 180
        cfg.update_epochs = 35
        cfg.n_pde = 384
        cfg.n_bc = 96
        cfg.n_ic = 96
        cfg.sensor_windows = 3
        cfg.sensor_stride = 8
        cfg.w_sensor = 10.0
        cfg.noise_levels = (0.0, 0.01, 0.05)
    elif mode == "progress":
        cfg.nx = 61
        cfg.nt = 181
        cfg.hidden_layers = 2
        cfg.hidden_units = 40
        cfg.baseline_epochs = 350
        cfg.update_epochs = 90
        cfg.n_pde = 640
        cfg.n_bc = 128
        cfg.n_ic = 128
        cfg.sensor_windows = 4
        cfg.sensor_stride = 7
        cfg.w_sensor = 8.0
        cfg.noise_levels = (0.01,)
    elif mode == "full":
        pass
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'smoke' or 'full'.")
    return cfg
