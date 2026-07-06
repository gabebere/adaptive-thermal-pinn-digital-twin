from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .evaluate import EvaluationResult
from .reference import ReferenceSolution, generate_reference_solution
from .sensors import SensorBatch, simulate_sensor_data


os.environ.setdefault("DDE_BACKEND", "pytorch")


@dataclass
class DeepXDEResult:
    model: object
    loss_history: object
    train_state: object
    runtime_s: float


def _import_deepxde():
    try:
        import deepxde as dde
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DeepXDE is not installed. Install the optional dependency with "
            "`pip install deepxde` or `pip install -e .[deepxde]`."
        ) from exc
    return dde, torch


def _pde_factory(cfg: ExperimentConfig):
    dde, _ = _import_deepxde()

    def pde(inputs, outputs):
        theta_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        theta_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
        return theta_t - cfg.beta * theta_xx

    return pde


def _hot_flux_bc_factory(cfg: ExperimentConfig):
    dde, torch = _import_deepxde()

    def hot_flux(inputs, outputs, _):
        theta_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        t_phys = inputs[:, 1:2] * cfg.t_final
        q = cfg.q_base + cfg.q_amp * torch.sin(2.0 * torch.pi * t_phys / cfg.q_period)
        q_nondim = q * cfg.L_wall / (cfg.k * cfg.delta_T)
        return -theta_x - q_nondim

    return hot_flux


def _coolant_bc_factory(cfg: ExperimentConfig):
    dde, _ = _import_deepxde()

    def coolant_flux(inputs, outputs, _):
        theta_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return -theta_x - cfg.biot * outputs

    return coolant_flux


def _build_data(cfg: ExperimentConfig, sensor_batches: list[SensorBatch] | None = None):
    dde, _ = _import_deepxde()
    geom = dde.geometry.Interval(0.0, 1.0)
    time_domain = dde.geometry.TimeDomain(0.0, 1.0)
    geom_time = dde.geometry.GeometryXTime(geom, time_domain)

    def on_hot(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0.0)

    def on_coolant(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1.0)

    def on_initial(_, on_initial_time):
        return on_initial_time

    theta0 = (cfg.T0 - cfg.T_cool) / cfg.delta_T
    bcs = [
        dde.icbc.OperatorBC(geom_time, _hot_flux_bc_factory(cfg), on_hot),
        dde.icbc.OperatorBC(geom_time, _coolant_bc_factory(cfg), on_coolant),
        dde.icbc.IC(geom_time, lambda _: theta0, on_initial),
    ]

    if sensor_batches:
        for batch in sensor_batches:
            points = np.hstack([batch.x_hat, batch.t_hat]).astype(np.float32)
            values = batch.theta.astype(np.float32)
            bcs.append(dde.icbc.PointSetBC(points, values, component=0))

    return dde.data.TimePDE(
        geom_time,
        _pde_factory(cfg),
        bcs,
        num_domain=cfg.n_pde,
        num_boundary=cfg.n_bc,
        num_initial=cfg.n_ic,
        train_distribution="pseudo",
    )


def _build_net(cfg: ExperimentConfig):
    dde, _ = _import_deepxde()
    return dde.nn.FNN(
        [2] + [cfg.hidden_units] * cfg.hidden_layers + [1],
        cfg.activation,
        "Glorot uniform",
    )


def _loss_weights(cfg: ExperimentConfig, sensor_count: int = 0) -> list[float]:
    return [cfg.w_pde, cfg.w_bc, cfg.w_bc, cfg.w_ic] + [cfg.w_sensor] * sensor_count


def train_deepxde_baseline(cfg: ExperimentConfig) -> DeepXDEResult:
    dde, _ = _import_deepxde()
    dde.config.set_random_seed(cfg.seed)
    data = _build_data(cfg)
    net = _build_net(cfg)
    model = dde.Model(data, net)
    model.compile("adam", lr=cfg.learning_rate, loss_weights=_loss_weights(cfg))
    start = time.perf_counter()
    loss_history, train_state = model.train(
        iterations=cfg.baseline_epochs,
        display_every=max(1, cfg.baseline_epochs // 5),
        verbose=1,
    )
    return DeepXDEResult(model=model, loss_history=loss_history, train_state=train_state, runtime_s=time.perf_counter() - start)


def update_deepxde_adaptive(
    baseline: DeepXDEResult,
    sensor_batches: list[SensorBatch],
    cfg: ExperimentConfig,
) -> DeepXDEResult:
    dde, _ = _import_deepxde()
    seen: list[SensorBatch] = []
    model = baseline.model
    loss_history = None
    train_state = None
    total_runtime = 0.0

    for batch in sensor_batches:
        seen.append(batch)
        data = _build_data(cfg, seen)
        model = dde.Model(data, model.net)
        model.compile("adam", lr=cfg.learning_rate, loss_weights=_loss_weights(cfg, sensor_count=len(seen)))
        start = time.perf_counter()
        loss_history, train_state = model.train(
            iterations=cfg.update_epochs,
            display_every=max(1, cfg.update_epochs // 2),
            verbose=1,
        )
        total_runtime += time.perf_counter() - start

    return DeepXDEResult(model=model, loss_history=loss_history, train_state=train_state, runtime_s=total_runtime)


def evaluate_deepxde_model(model: object, reference: ReferenceSolution, cfg: ExperimentConfig) -> EvaluationResult:
    x_hat = reference.x / cfg.L_wall
    t_hat = reference.t / cfg.t_final
    X, TT = np.meshgrid(x_hat, t_hat)
    points = np.stack([X.reshape(-1), TT.reshape(-1)], axis=1).astype(np.float32)
    theta_pred = model.predict(points).reshape(reference.T.shape)
    T_pred = cfg.T_cool + cfg.delta_T * theta_pred
    error = T_pred - reference.T
    response = reference.T - cfg.T_cool
    relative_l2_time = np.linalg.norm(error, axis=1) / (np.linalg.norm(response, axis=1) + 1.0e-6)
    return EvaluationResult(
        T_pred=T_pred,
        error=error,
        relative_l2_global=float(np.linalg.norm(error) / (np.linalg.norm(response) + 1.0e-12)),
        relative_l2_time=relative_l2_time,
        max_abs_error=float(np.max(np.abs(error))),
        hot_side_max_abs_error=float(np.max(np.abs(error[:, 0]))),
    )


def run_deepxde_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    reference = generate_reference_solution(cfg)
    baseline = train_deepxde_baseline(cfg)
    baseline_eval = evaluate_deepxde_model(baseline.model, reference, cfg)
    batches = simulate_sensor_data(reference, cfg, noise_level=cfg.main_noise_level)
    adaptive = update_deepxde_adaptive(baseline, batches, cfg)
    adaptive_eval = evaluate_deepxde_model(adaptive.model, reference, cfg)
    return {
        "baseline": {
            "relative_l2_global": baseline_eval.relative_l2_global,
            "max_abs_error": baseline_eval.max_abs_error,
            "hot_side_max_abs_error": baseline_eval.hot_side_max_abs_error,
            "training_runtime_s": baseline.runtime_s,
        },
        "adaptive": {
            "noise_level": cfg.main_noise_level,
            "relative_l2_global": adaptive_eval.relative_l2_global,
            "max_abs_error": adaptive_eval.max_abs_error,
            "hot_side_max_abs_error": adaptive_eval.hot_side_max_abs_error,
            "update_runtime_s": adaptive.runtime_s,
        },
    }
