"""Balanced DeepXDE PINN for the physical constant-flux engine wall."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import numpy as np

from constant_flux_physics import ConstantFluxConfig


def make_network(cfg: ConstantFluxConfig):
    network = dde.nn.FNN(
        [2, *cfg.hidden_layers, 1], cfg.activation, cfg.initializer
    )
    # theta=0 at t=0 is exact; theta is scaled temperature rise.
    network.apply_output_transform(lambda inputs, raw: inputs[:, 1:2] * raw)
    return network


def _flux_interpolator(times: np.ndarray, flux: np.ndarray, reveal_until: float | None):
    times = np.asarray(times, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if reveal_until is not None:
        revealed = times <= reveal_until + 1.0e-12
        known_times = times[revealed]
        known_flux = flux[revealed]
        if not len(known_times):
            known_times, known_flux = times[:1], flux[:1]
    else:
        known_times, known_flux = times, flux

    def profile(time_values: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(time_values).reshape(-1),
            known_times,
            known_flux,
            left=known_flux[0],
            right=known_flux[-1],
        )[:, None]

    return profile


def make_problem(
    cfg: ConstantFluxConfig,
    flux_profile,
    observation_points: np.ndarray | None = None,
    observation_values: np.ndarray | None = None,
):
    def pde(inputs, theta):
        theta_t = dde.grad.jacobian(theta, inputs, i=0, j=1)
        theta_xx = dde.grad.hessian(theta, inputs, component=0, i=0, j=0)
        return theta_t - cfg.beta * theta_xx

    def hot_boundary(inputs, theta, physical_points):
        theta_x = dde.grad.jacobian(theta, inputs, i=0, j=0)
        # DeepXDE supplies the collocation coordinates as a NumPy array in the
        # third argument.  Using it keeps the prescribed (non-trainable) flux
        # schedule off the autograd graph and also works when the network is on
        # CUDA, where a tensor cannot be converted to NumPy directly.
        physical_time = physical_points[:, 1:2] * cfg.final_time_s
        q_hot_numpy = flux_profile(physical_time)
        import torch

        q_hot = torch.as_tensor(
            q_hot_numpy, dtype=theta.dtype, device=theta.device
        )
        scaled_gradient = q_hot * cfg.wall_thickness_m / (
            cfg.conductivity_w_mk * cfg.temperature_scale_k
        )
        return theta_x + scaled_gradient

    def cool_boundary(inputs, theta, _):
        theta_x = dde.grad.jacobian(theta, inputs, i=0, j=0)
        return theta_x + cfg.biot * theta

    left = lambda point, on_boundary: on_boundary and np.isclose(point[0], 0.0)
    right = lambda point, on_boundary: on_boundary and np.isclose(point[0], 1.0)
    geometry = dde.geometry.Interval(0.0, 1.0)
    time_domain = dde.geometry.TimeDomain(0.0, 1.0)
    geometry_time = dde.geometry.GeometryXTime(geometry, time_domain)
    constraints = [
        dde.icbc.OperatorBC(geometry_time, hot_boundary, left),
        dde.icbc.OperatorBC(geometry_time, cool_boundary, right),
        dde.icbc.IC(geometry_time, lambda points: 0.0, lambda _, initial: initial),
    ]
    if observation_points is not None:
        points = np.asarray(observation_points, dtype=float).copy()
        points[:, 1] /= cfg.final_time_s
        values = (
            np.asarray(observation_values, dtype=float) - cfg.coolant_temperature_k
        ) / cfg.temperature_scale_k
        constraints.append(dde.icbc.PointSetBC(points, values, component=0))
    return dde.data.TimePDE(
        geometry_time,
        pde,
        constraints,
        num_domain=cfg.num_domain,
        num_boundary=cfg.num_boundary,
        num_initial=cfg.num_initial,
        train_distribution="Hammersley",
    )


def loss_weights(cfg: ConstantFluxConfig, with_data: bool = False) -> list[float]:
    weights = [
        cfg.pde_loss_weight,
        cfg.boundary_loss_weight,
        cfg.boundary_loss_weight,
        cfg.initial_loss_weight,
    ]
    return weights + ([cfg.data_loss_weight] if with_data else [])


def predict(network, points: np.ndarray, cfg: ConstantFluxConfig) -> np.ndarray:
    normalized = np.asarray(points, dtype=float).copy()
    normalized[:, 1] /= cfg.final_time_s
    network.eval()
    import torch

    device = next(network.parameters()).device
    rows = []
    with torch.inference_mode():
        for start in range(0, len(normalized), 65536):
            values = network(
                torch.as_tensor(normalized[start : start + 65536], dtype=torch.float32, device=device)
            )
            rows.append(values.cpu().numpy())
    theta = np.concatenate(rows)
    return cfg.coolant_temperature_k + cfg.temperature_scale_k * theta


def _rmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(reference) - np.asarray(prediction)) ** 2)))


@dataclass
class PhysicalPINNResult:
    network: object
    prediction: np.ndarray
    time_rmse: np.ndarray
    rmse: float
    network_state: dict
    update_latency_ms: np.ndarray | None = None


def train_baseline(
    cfg: ConstantFluxConfig,
    field_points: np.ndarray,
    field_values: np.ndarray,
    times: np.ndarray,
    verbose: int = 1,
) -> PhysicalPINNResult:
    dde.config.set_random_seed(cfg.seed)
    baseline_flux = np.full_like(times, cfg.baseline_flux_w_m2)
    profile = _flux_interpolator(times, baseline_flux, None)
    network = make_network(cfg)
    model = dde.Model(make_problem(cfg, profile), network)
    model.compile("adam", lr=cfg.baseline_learning_rate, loss_weights=loss_weights(cfg))
    model.train(
        iterations=cfg.baseline_iterations,
        display_every=max(1, cfg.baseline_iterations // 5),
        verbose=verbose,
    )
    prediction = predict(network, field_points, cfg)
    shaped_reference = field_values.reshape(len(times), -1)
    shaped_prediction = prediction.reshape(len(times), -1)
    return PhysicalPINNResult(
        network=network,
        prediction=shaped_prediction,
        time_rmse=np.sqrt(np.mean((shaped_prediction - shaped_reference) ** 2, axis=1)),
        rmse=_rmse(field_values, prediction),
        network_state=copy.deepcopy(network.state_dict()),
    )


def adapt_balanced(
    cfg: ConstantFluxConfig,
    baseline_state: dict,
    sensor_points: np.ndarray,
    sensor_values: np.ndarray,
    flux: np.ndarray,
    field_points: np.ndarray,
    field_values: np.ndarray,
    times: np.ndarray,
    verbose: int = 0,
) -> PhysicalPINNResult:
    network = make_network(cfg)
    network.load_state_dict(copy.deepcopy(baseline_state))
    posterior = np.full_like(field_values.reshape(len(times), -1), np.nan, dtype=float)
    latency = []
    revealed_batches: list[np.ndarray] = []
    for start in range(0, len(times), cfg.batch_size_n):
        batch_times = times[start : start + cfg.batch_size_n]
        revealed_batches.append(batch_times)
        if cfg.observation_window_batches is None:
            training_times = np.concatenate(revealed_batches)
        else:
            training_times = np.concatenate(
                revealed_batches[-cfg.observation_window_batches :]
            )
        observed = np.isin(np.round(sensor_points[:, 1], 12), np.round(training_times, 12))
        reveal_until = float(batch_times[-1])
        profile = _flux_interpolator(times, flux, reveal_until)
        data = make_problem(
            cfg,
            profile,
            sensor_points[observed],
            sensor_values[observed],
        )
        model = dde.Model(data, network)
        model.compile(
            "adam",
            lr=cfg.adaptive_learning_rate,
            loss_weights=loss_weights(cfg, with_data=True),
        )
        tick = time.perf_counter()
        model.train(
            iterations=cfg.adaptive_iterations_per_batch,
            display_every=max(1, cfg.adaptive_iterations_per_batch),
            verbose=verbose,
        )
        latency.append((time.perf_counter() - tick) * 1000.0)
        for tau in batch_times:
            index = int(np.flatnonzero(np.isclose(times, tau))[0])
            mask = np.isclose(field_points[:, 1], tau)
            posterior[index] = predict(network, field_points[mask], cfg).ravel()
    reference = field_values.reshape(len(times), -1)
    return PhysicalPINNResult(
        network=network,
        prediction=posterior,
        time_rmse=np.sqrt(np.mean((posterior - reference) ** 2, axis=1)),
        rmse=_rmse(reference, posterior),
        network_state=copy.deepcopy(network.state_dict()),
        update_latency_ms=np.asarray(latency),
    )
