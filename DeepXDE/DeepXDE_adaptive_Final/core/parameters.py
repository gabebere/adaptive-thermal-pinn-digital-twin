"""All user-adjustable physics, data, PINN, and online-update parameters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import tomllib

import numpy as np


# Edit these functions to change the shared paper spatial boundary profile or
# the initial field. The analytical series coefficients below are derived for
# this parabolic boundary profile.
def boundary_spatial_profile(coordinate):
    coordinate = np.asarray(coordinate)
    return coordinate - coordinate**2


def initial_condition(x, y):
    """Original paper initial field, retained for literature validation."""
    x, y = np.asarray(x), np.asarray(y)
    return boundary_spatial_profile(x) + boundary_spatial_profile(y)


def initial_condition_for_boundaries(x, y, boundaries):
    """Initial field compatible with the selected four Dirichlet edges."""
    x, y = np.asarray(x), np.asarray(y)
    a1, a2, a3, a4 = boundaries.amplitudes
    return (
        ((1.0 - x) * a1 + x * a2) * boundary_spatial_profile(y)
        + ((1.0 - y) * a3 + y * a4) * boundary_spatial_profile(x)
    )


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

CONTINUOUS_BOUNDARY_SETS = {
    "continuous_left": BoundarySet(
        "continuous_left", (0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    ),
    "continuous_left_high": BoundarySet(
        "continuous_left_high", (0.0, 0.0, 0.0, 0.0), (1.5, 0.0, 0.0, 0.0)
    ),
}


@dataclass
class WorkflowConfig:
    # Paths and reproducibility.
    output_dir: Path = Path(__file__).resolve().parent / "outputs"
    paper_tables_dir: Path = (
        Path(__file__).resolve().parents[3]
        / "Literature_results"
        / "mdpi_416_tables"
    )
    seed: int = 7

    # Analytical reference solution.
    boundary_set: BoundarySet = field(
        default_factory=lambda: CONTINUOUS_BOUNDARY_SETS["continuous_left"]
    )
    series_terms: int = 20
    tau_final: float = 100.0
    time_instances: int = 41
    field_nx: int = 17
    field_ny: int = 17
    time_distribution: str = "log"

    # Sparse numerical values treated as experimental sensor input.
    sensor_x: tuple[float, ...] = (0.2, 0.5, 0.8)
    sensor_y: tuple[float, ...] = (0.2, 0.5, 0.8)
    batch_size_n: int = 5
    adaptive_windows: int | None = None
    exclude_initial_sensor_time: bool = False
    sensor_noise_std: float = 0.0
    # None retains every observation.  An integer keeps only that many recent
    # batches in the data-loss term, which lets the PINN forget stale pre-event
    # data more quickly after an unknown boundary change.
    observation_window_batches: int | None = None

    # DeepXDE network and optimization.
    hidden_layers: tuple[int, ...] = (48, 48, 48)
    activation: str = "tanh"
    initializer: str = "Glorot normal"
    baseline_iterations: int = 2500
    adaptive_iterations_per_batch: int = 250
    baseline_learning_rate: float = 1.0e-3
    adaptive_learning_rate: float = 3.0e-4
    pde_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    initial_loss_weight: float = 1.0
    data_loss_weight: float = 10.0
    num_domain: int = 1800
    num_boundary: int = 400
    num_initial: int = 300
    train_distribution: str = "Hammersley"
    resample_period: int | None = None
    resample_pde_points: bool = True
    resample_bc_points: bool = False

    # Boundary-change experiment. The change occurs halfway through the total
    # time instances, not halfway through one n-point update batch.
    changed_boundary_set: BoundarySet = field(
        default_factory=lambda: CONTINUOUS_BOUNDARY_SETS["continuous_left_high"]
    )
    switch_fraction: float = 0.5
    reset_boundary_clock_at_switch: bool = True
    reveal_boundary_change_to_pinn: bool = False

    def validate(self) -> None:
        if self.batch_size_n < 1:
            raise ValueError("batch_size_n must be positive")
        if self.adaptive_windows is not None and self.adaptive_windows < 1:
            raise ValueError("adaptive_windows must be positive or None")
        if self.sensor_noise_std < 0.0:
            raise ValueError("sensor_noise_std must be nonnegative")
        if self.resample_period is not None and self.resample_period < 1:
            raise ValueError("resample_period must be positive or None")
        if self.train_distribution not in {
            "uniform", "pseudo", "LHS", "Halton", "Hammersley", "Sobol"
        }:
            raise ValueError("unsupported train_distribution")
        if (
            self.observation_window_batches is not None
            and self.observation_window_batches < 1
        ):
            raise ValueError("observation_window_batches must be positive or None")
        if self.time_instances < 3:
            raise ValueError("time_instances must be at least 3")
        if self.time_distribution not in {"linear", "log"}:
            raise ValueError("time_distribution must be 'linear' or 'log'")
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
        "balanced",
        2.5,
        2,
        sensor_x=(0.05, 0.5, 0.95),
        sensor_y=(0.05, 0.5, 0.95),
        observation_window_batches=2,
        data_loss_weight=10.0,
    ),
    LatencyExperiment(
        "low_latency",
        1.25,
        1,
        sensor_x=(0.05, 0.275, 0.5, 0.725, 0.95),
        sensor_y=(0.05, 0.275, 0.5, 0.725, 0.95),
        observation_window_batches=4,
        data_loss_weight=20.0,
    ),
)


MAINTAINED_ADAPTIVE_PROFILES = {
    experiment.name: experiment
    for experiment in LATENCY_EXPERIMENTS
    if experiment.name in {"balanced", "low_latency"}
}


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


def config_for_adaptive_profile(
    base: WorkflowConfig, profile_name: str = "balanced"
) -> WorkflowConfig:
    """Apply one of the two maintained online-adaptation configurations."""
    try:
        profile = MAINTAINED_ADAPTIVE_PROFILES[profile_name]
    except KeyError as error:
        choices = ", ".join(sorted(MAINTAINED_ADAPTIVE_PROFILES))
        raise ValueError(
            f"unknown adaptive profile {profile_name!r}; choose one of: {choices}"
        ) from error
    if base.baseline_iterations <= 80:
        # Smoke mode preserves the online semantics on a short ten-interval
        # horizon and limits optimization work; production profiles retain
        # their physical sample spacing and iteration counts.
        profile = replace(
            profile,
            sample_spacing_tau=base.tau_final / (base.time_instances - 1),
            adaptive_iterations_per_batch=10,
        )
    return config_for_latency_experiment(base, profile)


_ARCHITECTURE_FIELDS = {
    "network": {
        "hidden_layers",
        "activation",
        "initializer",
    },
    "training": {
        "baseline_iterations",
        "adaptive_iterations_per_batch",
        "baseline_learning_rate",
        "adaptive_learning_rate",
        "pde_loss_weight",
        "boundary_loss_weight",
        "initial_loss_weight",
        "data_loss_weight",
        "num_domain",
        "num_boundary",
        "num_initial",
        "train_distribution",
        "resample_period",
        "resample_pde_points",
        "resample_bc_points",
    },
    "streaming": {
        "sample_spacing_tau",
        "time_distribution",
        "batch_size_n",
        "adaptive_windows",
        "exclude_initial_sensor_time",
        "sensor_noise_std",
        "sensor_x",
        "sensor_y",
        "observation_window_batches",
    },
}


def load_architecture_file(
    base: WorkflowConfig, path: str | Path, *, smoke: bool = False
) -> WorkflowConfig:
    """Load a strict TOML architecture preset onto a workflow configuration.

    Physics and analytical-reference choices deliberately remain in
    ``parameters.py``.  The TOML file controls the replaceable neural-network,
    optimization, collocation, sensor, and online-update decisions.
    """
    architecture_path = Path(path)
    with architecture_path.open("rb") as stream:
        document = tomllib.load(stream)

    allowed_sections = {"profile", *_ARCHITECTURE_FIELDS}
    unknown_sections = set(document) - allowed_sections
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"unsupported architecture section(s): {names}")

    profile_metadata = document.get("profile", {})
    unknown_metadata = set(profile_metadata) - {"name", "description"}
    if unknown_metadata:
        names = ", ".join(sorted(unknown_metadata))
        raise ValueError(f"unsupported profile metadata field(s): {names}")

    replacements: dict[str, object] = {}
    sample_spacing_tau: float | None = None
    for section_name, allowed_fields in _ARCHITECTURE_FIELDS.items():
        section = document.get(section_name, {})
        unknown_fields = set(section) - allowed_fields
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(
                f"unsupported field(s) in [{section_name}]: {names}"
            )
        for field_name, value in section.items():
            if field_name == "sample_spacing_tau":
                sample_spacing_tau = float(value)
            elif field_name in {"hidden_layers", "sensor_x", "sensor_y"}:
                replacements[field_name] = tuple(value)
            else:
                replacements[field_name] = value

    configured = replace(base, **replacements)
    if smoke:
        # Smoke runs validate wiring and configuration without accidentally
        # launching a report-scale optimization or a long physical timeline.
        configured = replace(
            configured,
            time_instances=base.time_instances,
            baseline_iterations=min(configured.baseline_iterations, 80),
            adaptive_iterations_per_batch=min(
                configured.adaptive_iterations_per_batch, 10
            ),
            num_domain=min(configured.num_domain, base.num_domain),
            num_boundary=min(configured.num_boundary, base.num_boundary),
            num_initial=min(configured.num_initial, base.num_initial),
        )
    elif sample_spacing_tau is not None:
        if sample_spacing_tau <= 0.0:
            raise ValueError("sample_spacing_tau must be positive")
        intervals = int(round(configured.tau_final / sample_spacing_tau))
        if not np.isclose(intervals * sample_spacing_tau, configured.tau_final):
            raise ValueError("sample_spacing_tau must divide tau_final exactly")
        configured = replace(configured, time_instances=intervals + 1)

    configured.validate()
    return configured
