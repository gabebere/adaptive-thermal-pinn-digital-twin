"""Exact 1-D constant-flux/convective-wall reference and CSV corpus.

The transient solution is an eigenfunction expansion about the steady linear
profile. A flux switch is handled by linear superposition of two step
responses, so temperature is continuous through the event.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import tomllib

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class ConstantFluxConfig:
    seed: int = 8128
    wall_thickness_m: float = 5.0e-3
    conductivity_w_mk: float = 385.0
    density_kg_m3: float = 8960.0
    heat_capacity_j_kgk: float = 385.0
    coolant_h_w_m2k: float = 2.5e4
    coolant_temperature_k: float = 300.0
    initial_temperature_k: float = 300.0
    final_time_s: float = 1.0
    q_min_w_m2: float = 2.8e6
    q_max_w_m2: float = 5.2e6
    baseline_flux_w_m2: float = 4.0e6
    switched_flux_w_m2: float = 5.2e6
    switch_time_s: float = 0.5
    field_points: int = 81
    time_instances: int = 41
    series_terms: int = 80
    train_scenarios: int = 600
    validation_scenarios: int = 100
    test_scenarios: int = 200
    decaying_scenario_fraction: float = 0.30
    minimum_switch_fraction: float = 0.10
    maximum_switch_fraction: float = 0.90
    minimum_decay_rate_s: float = 0.5
    maximum_decay_rate_s: float = 8.0
    sensor_x_hat: tuple[float, ...] = (0.05, 0.5, 0.95)
    hidden_layers: tuple[int, ...] = (48, 48, 48)
    activation: str = "tanh"
    initializer: str = "Glorot normal"
    baseline_iterations: int = 2500
    adaptive_iterations_per_batch: int = 100
    baseline_learning_rate: float = 1.0e-3
    adaptive_learning_rate: float = 3.0e-4
    pde_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    initial_loss_weight: float = 1.0
    data_loss_weight: float = 10.0
    num_domain: int = 1800
    num_boundary: int = 400
    num_initial: int = 300
    batch_size_n: int = 2
    observation_window_batches: int | None = 2
    operator_hidden: int = 128
    operator_basis: int = 96
    operator_epochs: int = 500
    operator_batch_size: int = 16
    operator_learning_rate: float = 8.0e-4
    operator_pde_weight: float = 5.0e-3
    operator_boundary_weight: float = 5.0e-2
    operator_transition_weight: float = 3.0

    @property
    def diffusivity_m2_s(self) -> float:
        return self.conductivity_w_mk / (
            self.density_kg_m3 * self.heat_capacity_j_kgk
        )

    @property
    def biot(self) -> float:
        return self.coolant_h_w_m2k * self.wall_thickness_m / self.conductivity_w_mk

    @property
    def beta(self) -> float:
        return self.diffusivity_m2_s * self.final_time_s / self.wall_thickness_m**2

    @property
    def temperature_scale_k(self) -> float:
        return self.q_max_w_m2 * (
            1.0 / self.coolant_h_w_m2k
            + self.wall_thickness_m / self.conductivity_w_mk
        )


def load_balanced_config(path: Path, profile: str) -> ConstantFluxConfig:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    network = document.get("network", {})
    training = document.get("training", {})
    streaming = document.get("streaming", {})
    sample_spacing = float(streaming.get("sample_spacing_tau", 2.5)) / 100.0
    replacements = {
        "hidden_layers": tuple(network.get("hidden_layers", (48, 48, 48))),
        "activation": network.get("activation", "tanh"),
        "initializer": network.get("initializer", "Glorot normal"),
        "baseline_iterations": int(training.get("baseline_iterations", 2500)),
        "adaptive_iterations_per_batch": int(
            training.get("adaptive_iterations_per_batch", 100)
        ),
        "baseline_learning_rate": float(training.get("baseline_learning_rate", 1e-3)),
        "adaptive_learning_rate": float(training.get("adaptive_learning_rate", 3e-4)),
        "pde_loss_weight": float(training.get("pde_loss_weight", 1.0)),
        "boundary_loss_weight": float(training.get("boundary_loss_weight", 1.0)),
        "initial_loss_weight": float(training.get("initial_loss_weight", 1.0)),
        "data_loss_weight": float(training.get("data_loss_weight", 10.0)),
        "num_domain": int(training.get("num_domain", 1800)),
        "num_boundary": int(training.get("num_boundary", 400)),
        "num_initial": int(training.get("num_initial", 300)),
        "time_instances": int(round(1.0 / sample_spacing)) + 1,
        "batch_size_n": int(streaming.get("batch_size_n", 2)),
        "sensor_x_hat": tuple(streaming.get("sensor_x", (0.05, 0.5, 0.95))),
        "observation_window_batches": streaming.get("observation_window_batches", None),
    }
    cfg = replace(ConstantFluxConfig(), **replacements)
    if profile == "smoke":
        cfg = replace(
            cfg,
            field_points=31,
            time_instances=21,
            series_terms=30,
            train_scenarios=8,
            validation_scenarios=3,
            test_scenarios=3,
            hidden_layers=(24, 24),
            baseline_iterations=80,
            adaptive_iterations_per_batch=10,
            num_domain=180,
            num_boundary=60,
            num_initial=60,
            operator_hidden=32,
            operator_basis=24,
            operator_epochs=8,
            operator_batch_size=4,
        )
    return cfg


def robin_eigenvalues(biot: float, terms: int) -> np.ndarray:
    """Positive roots of mu*tan(mu)=Bi for Neumann/Robin eigenfunctions."""
    roots = []
    epsilon = 1.0e-10
    for index in range(terms):
        left = index * np.pi + epsilon
        right = index * np.pi + 0.5 * np.pi - epsilon
        roots.append(brentq(lambda mu: mu * np.tan(mu) - biot, left, right))
    return np.asarray(roots)


def steady_temperature(
    x_m: np.ndarray, flux_w_m2: float, cfg: ConstantFluxConfig
) -> np.ndarray:
    return (
        cfg.coolant_temperature_k
        + flux_w_m2 / cfg.coolant_h_w_m2k
        + flux_w_m2 * (cfg.wall_thickness_m - np.asarray(x_m)) / cfg.conductivity_w_mk
    )


def unit_flux_step_response(
    x_m: np.ndarray,
    time_s: np.ndarray,
    cfg: ConstantFluxConfig,
) -> np.ndarray:
    """Temperature rise per W/m2 for a constant flux applied at t=0."""
    x = np.asarray(x_m, dtype=float)
    time = np.asarray(time_s, dtype=float)
    xi = x / cfg.wall_thickness_m
    steady = 1.0 / cfg.coolant_h_w_m2k + (cfg.wall_thickness_m - x) / cfg.conductivity_w_mk
    roots = robin_eigenvalues(cfg.biot, cfg.series_terms)
    constant = -1.0 / cfg.coolant_h_w_m2k - cfg.wall_thickness_m / cfg.conductivity_w_mk
    slope = cfg.wall_thickness_m / cfg.conductivity_w_mk
    numerator = (
        constant * np.sin(roots) / roots
        + slope
        * (
            np.sin(roots) / roots
            + (np.cos(roots) - 1.0) / roots**2
        )
    )
    denominator = 0.5 + np.sin(2.0 * roots) / (4.0 * roots)
    coefficients = numerator / denominator
    modes = np.cos(roots[:, None] * xi[None, :])
    decay = np.exp(
        -cfg.diffusivity_m2_s
        * roots[None, :] ** 2
        * np.maximum(time[:, None], 0.0)
        / cfg.wall_thickness_m**2
    )
    response = steady[None, :] + (decay * coefficients[None, :]) @ modes
    response[time <= 0.0] = 0.0
    return response


def exponentially_relaxing_flux_temperature(
    x_m: np.ndarray,
    time_s: np.ndarray,
    flux_before_w_m2: float,
    flux_peak_w_m2: float,
    flux_terminal_w_m2: float,
    switch_time_s: float,
    decay_rate_s: float,
    cfg: ConstantFluxConfig,
) -> np.ndarray:
    """Exact response when the post-switch flux relaxes exponentially.

    After the event, ``q(t)=q_terminal+(q_peak-q_terminal) exp(-lambda*u)``.
    Duhamel's integral is evaluated analytically for every Robin eigenmode.
    """
    x = np.asarray(x_m, dtype=float)
    time = np.asarray(time_s, dtype=float)
    delayed = time - switch_time_s
    base = flux_before_w_m2 * unit_flux_step_response(x, time, cfg)
    switched = (
        flux_peak_w_m2 - flux_before_w_m2
    ) * unit_flux_step_response(x, delayed, cfg)

    steady = 1.0 / cfg.coolant_h_w_m2k + (
        cfg.wall_thickness_m - x
    ) / cfg.conductivity_w_mk
    roots = robin_eigenvalues(cfg.biot, cfg.series_terms)
    constant = -1.0 / cfg.coolant_h_w_m2k - cfg.wall_thickness_m / cfg.conductivity_w_mk
    slope = cfg.wall_thickness_m / cfg.conductivity_w_mk
    numerator = (
        constant * np.sin(roots) / roots
        + slope
        * (np.sin(roots) / roots + (np.cos(roots) - 1.0) / roots**2)
    )
    denominator = 0.5 + np.sin(2.0 * roots) / (4.0 * roots)
    coefficients = numerator / denominator
    modal_spatial = coefficients[:, None] * np.cos(
        roots[:, None] * x[None, :] / cfg.wall_thickness_m
    )
    rates = cfg.diffusivity_m2_s * roots**2 / cfg.wall_thickness_m**2
    u = np.maximum(delayed, 0.0)
    decay = float(decay_rate_s)
    difference = rates - decay
    modal_integral = np.empty((len(u), len(rates)))
    ordinary = np.abs(difference) > 1.0e-10
    modal_integral[:, ordinary] = (
        np.exp(-decay * u[:, None])
        - np.exp(-rates[None, ordinary] * u[:, None])
    ) / difference[None, ordinary]
    modal_integral[:, ~ordinary] = (
        u[:, None] * np.exp(-rates[None, ~ordinary] * u[:, None])
    )
    amplitude = flux_peak_w_m2 - flux_terminal_w_m2
    convolution = -amplitude * steady[None, :] * (
        1.0 - np.exp(-decay * u[:, None])
    )
    convolution += (-decay * amplitude * modal_integral) @ modal_spatial
    convolution[delayed < 0.0] = 0.0
    return cfg.initial_temperature_k + base + switched + convolution


def switched_flux_temperature(
    x_m: np.ndarray,
    time_s: np.ndarray,
    flux_before_w_m2: float,
    flux_after_w_m2: float,
    switch_time_s: float,
    cfg: ConstantFluxConfig,
) -> np.ndarray:
    """Exact linear-system response to a piecewise-constant heat flux."""
    time = np.asarray(time_s, dtype=float)
    before = unit_flux_step_response(x_m, time, cfg) * flux_before_w_m2
    delayed_time = time - switch_time_s
    change = unit_flux_step_response(x_m, delayed_time, cfg) * (
        flux_after_w_m2 - flux_before_w_m2
    )
    return cfg.initial_temperature_k + before + np.where(
        delayed_time[:, None] >= 0.0, change, 0.0
    )


def analytical_validation(cfg: ConstantFluxConfig) -> dict[str, float]:
    x = np.linspace(0.0, cfg.wall_thickness_m, cfg.field_points)
    times = np.array([0.0, cfg.final_time_s])
    transient = switched_flux_temperature(
        x,
        times,
        cfg.baseline_flux_w_m2,
        cfg.baseline_flux_w_m2,
        cfg.final_time_s + 1.0,
        cfg,
    )
    steady = steady_temperature(x, cfg.baseline_flux_w_m2, cfg)
    return {
        "initial_max_abs_error_k": float(
            np.max(np.abs(transient[0] - cfg.initial_temperature_k))
        ),
        "one_second_to_steady_max_abs_k": float(np.max(np.abs(transient[-1] - steady))),
        "steady_hot_temperature_k": float(steady[0]),
        "steady_cool_temperature_k": float(steady[-1]),
        "biot": cfg.biot,
        "fourier_beta": cfg.beta,
    }


def _write_gzip_csv(path: Path, header: str, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                np.savetxt(stream, values, delimiter=",", header=header, comments="", fmt="%.12g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_csv_corpus(root: Path, cfg: ConstantFluxConfig) -> None:
    """Generate stratified exact CSV splits with step and decaying fluxes."""
    if (root / "manifest.csv").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(0.0, cfg.wall_thickness_m, cfg.field_points)
    x_hat = x / cfg.wall_thickness_m
    times = np.linspace(0.0, cfg.final_time_s, cfg.time_instances)
    sensor_indices = np.asarray([np.argmin(np.abs(x_hat - value)) for value in cfg.sensor_x_hat])
    specifications: list[dict[str, float | str | bool]] = []
    for split, count in (
        ("train", cfg.train_scenarios),
        ("validation", cfg.validation_scenarios),
        ("test_interpolation", cfg.test_scenarios),
    ):
        # Stratification prevents random clustering in event time, direction,
        # and step magnitude.  The shuffle removes a correlation with row id.
        switch_quantiles = (np.arange(count) + rng.random(count)) / count
        rng.shuffle(switch_quantiles)
        switch_fractions = cfg.minimum_switch_fraction + switch_quantiles * (
            cfg.maximum_switch_fraction - cfg.minimum_switch_fraction
        )
        magnitude_quantiles = (np.arange(count) + rng.random(count)) / count
        rng.shuffle(magnitude_quantiles)
        magnitudes = 0.35e6 + magnitude_quantiles * 2.0e6
        directions = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
        rng.shuffle(directions)
        decaying_count = int(round(cfg.decaying_scenario_fraction * count))
        boundary_modes = np.asarray(
            ["exponential"] * decaying_count + ["step"] * (count - decaying_count)
        )
        rng.shuffle(boundary_modes)
        for index in range(count):
            magnitude = float(magnitudes[index])
            if directions[index] > 0:
                before = rng.uniform(cfg.q_min_w_m2, cfg.q_max_w_m2 - magnitude)
                peak = before + magnitude
            else:
                before = rng.uniform(cfg.q_min_w_m2 + magnitude, cfg.q_max_w_m2)
                peak = before - magnitude
            mode = str(boundary_modes[index])
            if mode == "exponential":
                # A genuine decay after the event.  The event jump itself is
                # still balanced across signs and magnitudes by the strata.
                available_drop = max(peak - cfg.q_min_w_m2, 0.10e6)
                drop = rng.uniform(0.10e6, available_drop)
                terminal = max(cfg.q_min_w_m2, peak - drop)
                decay_rate = float(
                    np.exp(
                        rng.uniform(
                            np.log(cfg.minimum_decay_rate_s),
                            np.log(cfg.maximum_decay_rate_s),
                        )
                    )
                )
            else:
                terminal = peak
                decay_rate = 0.0
            switch = float(
                times[
                    np.argmin(
                        np.abs(times - switch_fractions[index] * cfg.final_time_s)
                    )
                ]
            )
            specifications.append(
                {
                    "scenario_id": f"{split}_{index:04d}",
                    "split": split,
                    "q_before_w_m2": float(before),
                    "q_peak_w_m2": float(peak),
                    "q_terminal_w_m2": float(terminal),
                    "switch_time_s": switch,
                    "decay_rate_s": decay_rate,
                    "boundary_mode": mode,
                    "locked": False,
                }
            )
    specifications.append(
        {
            "scenario_id": "locked_q4p0_to_q5p2_t0p5",
            "split": "test_locked",
            "q_before_w_m2": cfg.baseline_flux_w_m2,
            "q_peak_w_m2": cfg.switched_flux_w_m2,
            "q_terminal_w_m2": cfg.switched_flux_w_m2,
            "switch_time_s": cfg.switch_time_s,
            "decay_rate_s": 0.0,
            "boundary_mode": "step",
            "locked": True,
        }
    )
    manifest_rows, checksum_rows = [], []
    for number, specification in enumerate(specifications, 1):
        scenario_id = str(specification["scenario_id"])
        split = str(specification["split"])
        before = float(specification["q_before_w_m2"])
        peak = float(specification["q_peak_w_m2"])
        terminal = float(specification["q_terminal_w_m2"])
        switch = float(specification["switch_time_s"])
        decay_rate = float(specification["decay_rate_s"])
        mode = str(specification["boundary_mode"])
        locked = bool(specification["locked"])
        if mode == "exponential":
            temperature = exponentially_relaxing_flux_temperature(
                x, times, before, peak, terminal, switch, decay_rate, cfg
            )
            current_flux = np.where(
                times < switch,
                before,
                terminal + (peak - terminal) * np.exp(-decay_rate * (times - switch)),
            )
        else:
            temperature = switched_flux_temperature(x, times, before, peak, switch, cfg)
            current_flux = np.where(times < switch, before, peak)
        tt, xx = np.meshgrid(times, x_hat, indexing="ij")
        field = np.column_stack(
            (
                xx.ravel(),
                tt.ravel(),
                temperature.ravel(),
                np.repeat(current_flux, len(x)),
            )
        )
        field_path = root / split / f"{scenario_id}_field.csv.gz"
        _write_gzip_csv(field_path, "x_hat,time_s,temperature_k,q_hot_w_m2", field)
        sensor_table = np.column_stack(
            (
                np.repeat(times, len(sensor_indices)),
                np.tile(x_hat[sensor_indices], len(times)),
                temperature[:, sensor_indices].ravel(),
                np.repeat(current_flux, len(sensor_indices)),
            )
        )
        sensor_path = root / split / f"{scenario_id}_sensors.csv"
        np.savetxt(
            sensor_path,
            sensor_table,
            delimiter=",",
            header="time_s,x_hat,temperature_k,q_hot_w_m2",
            comments="",
            fmt="%.12g",
        )
        manifest_rows.append(
            {
                "scenario_id": scenario_id,
                "split": split,
                "q_before_w_m2": before,
                "q_peak_w_m2": peak,
                "q_terminal_w_m2": terminal,
                "switch_time_s": switch,
                "decay_rate_s": decay_rate,
                "boundary_mode": mode,
                "locked": int(locked),
                "field_file": field_path.relative_to(root).as_posix(),
                "sensor_file": sensor_path.relative_to(root).as_posix(),
            }
        )
        checksum_rows.extend(
            (
                {"file": field_path.relative_to(root).as_posix(), "sha256": _sha256(field_path)},
                {"file": sensor_path.relative_to(root).as_posix(), "sha256": _sha256(sensor_path)},
            )
        )
        if number % 50 == 0 or number == len(specifications):
            print(f"constant-flux CSV scenarios: {number}/{len(specifications)}", flush=True)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader(); writer.writerows(manifest_rows)
    with (root / "checksums.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=checksum_rows[0].keys())
        writer.writeheader(); writer.writerows(checksum_rows)
    (root / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def read_manifest(root: Path, split: str) -> list[dict[str, str]]:
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["split"] == split]


def load_scenario(root: Path, row: dict[str, str], cfg: ConstantFluxConfig) -> dict[str, np.ndarray]:
    field = np.loadtxt(root / row["field_file"], delimiter=",", skiprows=1, dtype=np.float32)
    sensor = np.loadtxt(root / row["sensor_file"], delimiter=",", skiprows=1, dtype=np.float32)
    nt, nx, ns = cfg.time_instances, cfg.field_points, len(cfg.sensor_x_hat)
    return {
        "x_hat": field[:nx, 0],
        "times": field[:, 1].reshape(nt, nx)[:, 0],
        "field": field[:, 2].reshape(nt, nx),
        "flux": field[:, 3].reshape(nt, nx)[:, 0],
        "sensor_x_hat": sensor[:ns, 1],
        "sensor_values": sensor[:, 2].reshape(nt, ns),
    }
