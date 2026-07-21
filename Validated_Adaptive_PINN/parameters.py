"""All user-adjustable physics, data, PINN, and online-update parameters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


# Edit these functions to change the shared paper spatial boundary profile or
# the initial field. The analytical series coefficients below are derived for
# this parabolic boundary profile.
def boundary_spatial_profile(coordinate):
    coordinate = np.asarray(coordinate)
    return coordinate - coordinate**2


def initial_condition(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return boundary_spatial_profile(x) + boundary_spatial_profile(y)


@dataclass(frozen=True)
class BoundarySet:
    """Four paper boundary functions eta_i(tau)=amplitude_i*exp(-decay_i*tau)."""

    name: str
    decays: tuple[float, float, float, float]
    amplitudes: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


PAPER_BOUNDARY_SETS = {
    "set_1": BoundarySet("set_1", (1.0, 1.0, 1.0, 1.0)),
    "set_2": BoundarySet("set_2", (1.0, 1.0, 2.0, 2.0)),
    "set_3": BoundarySet("set_3", (1.0, 2.0, 3.0, 4.0)),
}


@dataclass
class WorkflowConfig:
    # Paths and reproducibility.
    output_dir: Path = Path(__file__).resolve().parent / "outputs"
    paper_tables_dir: Path = (
        Path(__file__).resolve().parents[1] / "paper_result" / "mdpi_416_tables"
    )
    seed: int = 7

    # Analytical reference solution.
    boundary_set: BoundarySet = field(default_factory=lambda: PAPER_BOUNDARY_SETS["set_1"])
    series_terms: int = 20
    tau_final: float = 100.0
    time_instances: int = 41
    field_nx: int = 17
    field_ny: int = 17

    # Sparse numerical values treated as experimental sensor input.
    sensor_x: tuple[float, ...] = (0.2, 0.5, 0.8)
    sensor_y: tuple[float, ...] = (0.2, 0.5, 0.8)
    batch_size_n: int = 5

    # DeepXDE network and optimization.
    hidden_layers: tuple[int, ...] = (48, 48, 48)
    activation: str = "tanh"
    baseline_iterations: int = 2500
    adaptive_iterations_per_batch: int = 250
    baseline_learning_rate: float = 1.0e-3
    adaptive_learning_rate: float = 3.0e-4
    data_loss_weight: float = 10.0
    num_domain: int = 1800
    num_boundary: int = 400
    num_initial: int = 300

    # Boundary-change experiment. The change occurs halfway through the total
    # time instances, not halfway through one n-point update batch.
    changed_boundary_set: BoundarySet = field(
        default_factory=lambda: PAPER_BOUNDARY_SETS["set_2"]
    )
    switch_fraction: float = 0.5
    reset_boundary_clock_at_switch: bool = True
    reveal_boundary_change_to_pinn: bool = False
    switch_solver_grid: int = 17
    switch_solver_substeps_per_interval: int = 4

    def validate(self) -> None:
        if self.batch_size_n < 1:
            raise ValueError("batch_size_n must be positive")
        if self.time_instances < 3:
            raise ValueError("time_instances must be at least 3")
        if not 0.0 < self.switch_fraction < 1.0:
            raise ValueError("switch_fraction must lie strictly between 0 and 1")
        for locations in (self.sensor_x, self.sensor_y):
            if not locations or any(value <= 0.0 or value >= 1.0 for value in locations):
                raise ValueError("sensor locations must lie strictly inside the unit square")


def make_config(profile: str = "full") -> WorkflowConfig:
    """Create a report-quality or quick verification configuration."""
    cfg = WorkflowConfig()
    if profile == "full":
        return cfg
    if profile == "smoke":
        return replace(
            cfg,
            series_terms=10,
            tau_final=3.0,
            time_instances=11,
            field_nx=9,
            field_ny=9,
            hidden_layers=(24, 24),
            baseline_iterations=80,
            adaptive_iterations_per_batch=15,
            num_domain=180,
            num_boundary=60,
            num_initial=60,
            switch_solver_grid=9,
            switch_solver_substeps_per_interval=2,
        )
    raise ValueError("profile must be 'smoke' or 'full'")
