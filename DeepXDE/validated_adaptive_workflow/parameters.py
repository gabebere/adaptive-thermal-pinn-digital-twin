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
        Path(__file__).resolve().parents[2]
        / "Literature_results"
        / "mdpi_416_tables"
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
    # None retains every observation.  An integer keeps only that many recent
    # batches in the data-loss term, which lets the PINN forget stale pre-event
    # data more quickly after an unknown boundary change.
    observation_window_batches: int | None = None

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
        if (
            self.observation_window_batches is not None
            and self.observation_window_batches < 1
        ):
            raise ValueError("observation_window_batches must be positive or None")
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


@dataclass(frozen=True)
class LatencyExperiment:
    """One editable row in the boundary-change latency study."""

    name: str
    sample_spacing_tau: float
    batch_size_n: int
    sensor_x: tuple[float, ...] = (0.2, 0.5, 0.8)
    sensor_y: tuple[float, ...] = (0.2, 0.5, 0.8)
    observation_window_batches: int | None = None
    data_loss_weight: float = 10.0
    adaptive_iterations_per_batch: int = 100
    reveal_boundary_change_to_pinn: bool = False


# These experiments isolate one latency control at a time, followed by a
# combined low-latency setup.  Add or edit rows here without changing the
# training or plotting code.
LATENCY_EXPERIMENTS = (
    LatencyExperiment("current: n=5, all history", 2.5, 5),
    LatencyExperiment("smaller batch: n=2", 2.5, 2),
    LatencyExperiment("immediate updates: n=1", 2.5, 1),
    LatencyExperiment("denser sampling: dt=1.25", 1.25, 2),
    LatencyExperiment(
        "boundary-aware sensors",
        2.5,
        2,
        sensor_x=(0.05, 0.5, 0.95),
        sensor_y=(0.05, 0.5, 0.95),
    ),
    LatencyExperiment(
        "recent history: 2 batches",
        2.5,
        2,
        observation_window_batches=2,
    ),
    LatencyExperiment(
        "combined low-latency",
        1.25,
        1,
        sensor_x=(0.05, 0.275, 0.5, 0.725, 0.95),
        sensor_y=(0.05, 0.275, 0.5, 0.725, 0.95),
        observation_window_batches=4,
        data_loss_weight=20.0,
    ),
)


LATENCY_RECOVERY_FRACTION = 0.50
LATENCY_RECOVERY_CONSECUTIVE_INSTANCES = 2


def config_for_latency_experiment(
    base: WorkflowConfig, experiment: LatencyExperiment
) -> WorkflowConfig:
    """Apply one latency-study row to a base workflow configuration."""
    intervals = int(round(base.tau_final / experiment.sample_spacing_tau))
    if not np.isclose(intervals * experiment.sample_spacing_tau, base.tau_final):
        raise ValueError(
            f"{experiment.name}: sample spacing must divide tau_final exactly"
        )
    return replace(
        base,
        time_instances=intervals + 1,
        batch_size_n=experiment.batch_size_n,
        sensor_x=experiment.sensor_x,
        sensor_y=experiment.sensor_y,
        observation_window_batches=experiment.observation_window_batches,
        data_loss_weight=experiment.data_loss_weight,
        adaptive_iterations_per_batch=experiment.adaptive_iterations_per_batch,
        reveal_boundary_change_to_pinn=experiment.reveal_boundary_change_to_pinn,
    )
