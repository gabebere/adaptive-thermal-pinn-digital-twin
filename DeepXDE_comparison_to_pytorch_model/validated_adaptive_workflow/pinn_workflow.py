"""Stages 4-5: offline DeepXDE PINN and accumulated online adaptation."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import numpy as np

from parameters import (
    BoundarySet,
    WorkflowConfig,
    boundary_spatial_profile,
    initial_condition,
)


def tau_to_network_time(tau: np.ndarray, tau_final: float) -> np.ndarray:
    """Logarithmic time coordinate s in [0,1] for resolving early transients."""
    return np.log1p(tau) / np.log1p(tau_final)


def to_network_points(points: np.ndarray, tau_final: float) -> np.ndarray:
    result = np.asarray(points, dtype=float).copy()
    result[:, 2] = tau_to_network_time(result[:, 2], tau_final)
    return result


def make_network(cfg: WorkflowConfig):
    return dde.nn.FNN(
        [3, *cfg.hidden_layers, 1], cfg.activation, "Glorot uniform"
    )


def resampling_callbacks():
    """Draw fresh interior/initial physics points every Adam iteration.

    Boundary points stay fixed because DeepXDE requires each separately
    weighted 2-D boundary condition to retain its compiled sample count.
    """
    return [
        dde.callbacks.PDEPointResampler(
            period=1,
            pde_points=True,
            bc_points=False,
        )
    ]


def make_problem(
    cfg: WorkflowConfig,
    boundaries: BoundarySet,
    observation_points: np.ndarray | None = None,
    observation_values: np.ndarray | None = None,
):
    log_scale = float(np.log1p(cfg.tau_final))

    def pde(X, theta):
        theta_s = dde.grad.jacobian(theta, X, i=0, j=2)
        theta_xx = dde.grad.hessian(theta, X, component=0, i=0, j=0)
        theta_yy = dde.grad.hessian(theta, X, component=0, i=1, j=1)
        d_tau_d_s = log_scale * dde.backend.exp(log_scale * X[:, 2:3])
        return theta_s - d_tau_d_s * (theta_xx + theta_yy)

    def left(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[0], 0.0)

    def right(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[0], 1.0)

    def bottom(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[1], 0.0)

    def top(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[1], 1.0)

    def side_function(coordinate_index: int, side_index: int):
        decay = boundaries.decays[side_index]
        amplitude = boundaries.amplitudes[side_index]

        def function(X):
            coordinate = X[:, coordinate_index : coordinate_index + 1]
            tau = np.expm1(log_scale * X[:, 2:3])
            value = (
                amplitude
                * boundary_spatial_profile(coordinate)
                * np.exp(-decay * tau)
            )
            if not cfg.reveal_boundary_change_to_pinn:
                return value

            switch_tau = cfg.switch_fraction * cfg.tau_final
            changed_decay = cfg.changed_boundary_set.decays[side_index]
            changed_amplitude = cfg.changed_boundary_set.amplitudes[side_index]
            changed_tau = (
                np.maximum(tau - switch_tau, 0.0)
                if cfg.reset_boundary_clock_at_switch
                else tau
            )
            changed_value = (
                changed_amplitude
                * boundary_spatial_profile(coordinate)
                * np.exp(-changed_decay * changed_tau)
            )
            return np.where(tau >= switch_tau, changed_value, value)

        return function

    geometry = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    time_domain = dde.geometry.TimeDomain(0.0, 1.0)
    geometry_time = dde.geometry.GeometryXTime(geometry, time_domain)
    constraints = [
        dde.icbc.DirichletBC(geometry_time, side_function(1, 0), left),
        dde.icbc.DirichletBC(geometry_time, side_function(1, 1), right),
        dde.icbc.DirichletBC(geometry_time, side_function(0, 2), bottom),
        dde.icbc.DirichletBC(geometry_time, side_function(0, 3), top),
        dde.icbc.IC(
            geometry_time,
            lambda X: initial_condition(X[:, 0:1], X[:, 1:2]),
            lambda _, initial: initial,
        ),
    ]
    if observation_points is not None:
        constraints.append(
            dde.icbc.PointSetBC(
                to_network_points(observation_points, cfg.tau_final),
                observation_values,
                component=0,
            )
        )
    return dde.data.TimePDE(
        geometry_time,
        pde,
        constraints,
        num_domain=cfg.num_domain,
        num_boundary=cfg.num_boundary,
        num_initial=cfg.num_initial,
        train_distribution="pseudo",
    )


def predict(model, points: np.ndarray, cfg: WorkflowConfig) -> np.ndarray:
    return model.predict(to_network_points(points, cfg.tau_final))


def rmse(reference, prediction) -> float:
    return float(np.sqrt(np.mean((np.asarray(reference) - np.asarray(prediction)) ** 2)))


def rmse_by_time(points, reference, prediction, times):
    result = []
    for tau in times:
        mask = np.isclose(points[:, 2], tau)
        result.append(rmse(reference[mask], prediction[mask]))
    return np.asarray(result)


@dataclass
class BaselineResult:
    model: object
    network_state: dict
    field_prediction: np.ndarray
    field_rmse: float
    time_rmse: np.ndarray


@dataclass
class AdaptiveResult:
    model: object
    field_prediction: np.ndarray
    field_rmse: float
    time_rmse: np.ndarray
    causal_prior_time_rmse: np.ndarray
    causal_posterior_time_rmse: np.ndarray
    causal_prior_field_prediction: np.ndarray
    causal_posterior_field_prediction: np.ndarray
    history: list[dict]


def train_baseline(
    cfg: WorkflowConfig,
    field_points: np.ndarray,
    field_values: np.ndarray,
    times: np.ndarray,
    verbose: int = 1,
) -> BaselineResult:
    dde.config.set_random_seed(cfg.seed)
    network = make_network(cfg)
    model = dde.Model(make_problem(cfg, cfg.boundary_set), network)
    model.compile(
        "adam",
        lr=cfg.baseline_learning_rate,
        loss_weights=[1.0, 2.0, 2.0, 2.0, 2.0, 5.0],
    )
    model.train(
        iterations=cfg.baseline_iterations,
        display_every=max(1, cfg.baseline_iterations // 5),
        verbose=verbose,
        callbacks=resampling_callbacks(),
    )
    field_prediction = predict(model, field_points, cfg)
    return BaselineResult(
        model=model,
        network_state=copy.deepcopy(network.state_dict()),
        field_prediction=field_prediction,
        field_rmse=rmse(field_values, field_prediction),
        time_rmse=rmse_by_time(field_points, field_values, field_prediction, times),
    )


def adapt_online(
    cfg: WorkflowConfig,
    baseline_state: dict,
    sensor_points: np.ndarray,
    sensor_values: np.ndarray,
    field_points: np.ndarray,
    field_values: np.ndarray,
    times: np.ndarray,
    verbose: int = 1,
) -> AdaptiveResult:
    """Reveal n instances, predict, then update using all or recent observations.

    The causal arrays save the error produced by the network state that actually
    existed at each online step.  They avoid retrospectively evaluating the
    final adapted network over times that occurred before it was trained.
    """
    network = make_network(cfg)
    network.load_state_dict(copy.deepcopy(baseline_state))
    model = dde.Model(make_problem(cfg, cfg.boundary_set), network)
    model.compile(
        "adam",
        lr=cfg.adaptive_learning_rate,
        loss_weights=[1.0, 2.0, 2.0, 2.0, 2.0, 5.0],
    )

    time_batches = [
        times[start : start + cfg.batch_size_n]
        for start in range(0, len(times), cfg.batch_size_n)
    ]
    history = []
    revealed_batches: list[np.ndarray] = []
    causal_prior = np.full(len(times), np.nan)
    causal_posterior = np.full(len(times), np.nan)
    causal_prior_field = np.full_like(field_values, np.nan, dtype=float)
    causal_posterior_field = np.full_like(field_values, np.nan, dtype=float)
    for batch_index, batch_times in enumerate(time_batches, start=1):
        revealed_batches.append(batch_times)
        if cfg.observation_window_batches is None:
            training_batches = revealed_batches
        else:
            training_batches = revealed_batches[-cfg.observation_window_batches :]
        training_times = np.concatenate(training_batches)
        new_mask = np.isin(np.round(sensor_points[:, 2], 12), np.round(batch_times, 12))
        observed_mask = np.isin(
            np.round(sensor_points[:, 2], 12), np.round(training_times, 12)
        )
        new_field_mask = np.isin(
            np.round(field_points[:, 2], 12), np.round(batch_times, 12)
        )
        before = predict(model, sensor_points[new_mask], cfg)
        before_rmse = rmse(sensor_values[new_mask], before)
        before_field = predict(model, field_points[new_field_mask], cfg)
        before_field_by_time = rmse_by_time(
            field_points[new_field_mask],
            field_values[new_field_mask],
            before_field,
            batch_times,
        )
        batch_time_indices = [
            int(np.flatnonzero(np.isclose(times, tau))[0]) for tau in batch_times
        ]
        causal_prior[batch_time_indices] = before_field_by_time
        causal_prior_field[new_field_mask] = before_field

        data = make_problem(
            cfg,
            cfg.boundary_set,
            sensor_points[observed_mask],
            sensor_values[observed_mask],
        )
        model = dde.Model(data, network)  # same network: weights are retained
        model.compile(
            "adam",
            lr=cfg.adaptive_learning_rate,
            loss_weights=[1.0, 2.0, 2.0, 2.0, 2.0, 5.0, cfg.data_loss_weight],
        )
        model.train(
            iterations=cfg.adaptive_iterations_per_batch,
            display_every=max(1, cfg.adaptive_iterations_per_batch),
            verbose=verbose,
            callbacks=resampling_callbacks(),
        )
        after = predict(model, sensor_points[new_mask], cfg)
        after_field = predict(model, field_points[new_field_mask], cfg)
        after_field_by_time = rmse_by_time(
            field_points[new_field_mask],
            field_values[new_field_mask],
            after_field,
            batch_times,
        )
        causal_posterior[batch_time_indices] = after_field_by_time
        causal_posterior_field[new_field_mask] = after_field
        history.append(
            {
                "batch": batch_index,
                "time_start": float(batch_times[0]),
                "time_end": float(batch_times[-1]),
                "instances": int(len(batch_times)),
                "new_sensor_points": int(np.count_nonzero(new_mask)),
                "total_revealed_time_instances": int(sum(map(len, revealed_batches))),
                "training_time_instances": int(len(training_times)),
                "training_sensor_points": int(np.count_nonzero(observed_mask)),
                # Backward-compatible name used by earlier result readers.
                "accumulated_sensor_points": int(np.count_nonzero(observed_mask)),
                "before_batch_rmse": before_rmse,
                "after_batch_rmse": rmse(sensor_values[new_mask], after),
                "before_batch_field_rmse": rmse(
                    field_values[new_field_mask], before_field
                ),
                "after_batch_field_rmse": rmse(
                    field_values[new_field_mask], after_field
                ),
            }
        )

    final_prediction = predict(model, field_points, cfg)
    return AdaptiveResult(
        model=model,
        field_prediction=final_prediction,
        field_rmse=rmse(field_values, final_prediction),
        time_rmse=rmse_by_time(field_points, field_values, final_prediction, times),
        causal_prior_time_rmse=causal_prior,
        causal_posterior_time_rmse=causal_posterior,
        causal_prior_field_prediction=causal_prior_field,
        causal_posterior_field_prediction=causal_posterior_field,
        history=history,
    )
