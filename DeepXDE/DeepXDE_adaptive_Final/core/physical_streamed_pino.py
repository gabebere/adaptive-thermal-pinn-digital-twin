"""Causal physics-informed operator for constant-flux wall trajectories."""

from __future__ import annotations

import copy
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from constant_flux_physics import (
    ConstantFluxConfig,
    load_scenario,
    read_manifest,
)


class ConstantFluxStreamedPINO(nn.Module):
    def __init__(self, sensors: int, hidden: int = 96, basis: int = 64):
        super().__init__()
        self.hidden = hidden
        # Temperatures plus q, delta-q, time, and dt.  Giving the observer the
        # explicit flux increment makes an abrupt event distinguishable from
        # a slowly relaxing boundary without waiting for several GRU steps.
        self.observer = nn.GRUCell(sensors + 4, hidden)
        self.coefficients = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, basis)
        )
        self.spatial = nn.Sequential(
            nn.Linear(5, hidden), nn.Tanh(), nn.Linear(hidden, basis)
        )

    def initial_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden, device=device)

    @staticmethod
    def spatial_features(x_hat: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                x_hat,
                torch.sin(math.pi * x_hat),
                torch.cos(math.pi * x_hat),
                torch.sin(2.0 * math.pi * x_hat),
                torch.cos(2.0 * math.pi * x_hat),
            ),
            dim=-1,
        )

    def observe(
        self,
        state: torch.Tensor,
        measurements_k: torch.Tensor,
        flux_k: torch.Tensor,
        delta_flux_k: torch.Tensor,
        time_k: torch.Tensor,
        dt_k: torch.Tensor,
        cfg: ConstantFluxConfig,
    ) -> torch.Tensor:
        normalized_measurements = (
            measurements_k - cfg.coolant_temperature_k
        ) / cfg.temperature_scale_k
        features = torch.cat(
            (
                normalized_measurements,
                flux_k[:, None] / cfg.q_max_w_m2,
                delta_flux_k[:, None] / (cfg.q_max_w_m2 - cfg.q_min_w_m2),
                time_k[:, None] / cfg.final_time_s,
                dt_k[:, None] / cfg.final_time_s,
            ),
            dim=-1,
        )
        return self.observer(features, state)

    def decode(
        self,
        state: torch.Tensor,
        x_hat: torch.Tensor,
        time_k: torch.Tensor,
        cfg: ConstantFluxConfig,
    ) -> torch.Tensor:
        coefficients = self.coefficients(state)
        modes = self.spatial(self.spatial_features(x_hat))
        raw = (modes * coefficients[:, None]).sum(-1) / math.sqrt(coefficients.shape[-1])
        time_gate = 1.0 - torch.exp(-8.0 * time_k[:, None] / cfg.final_time_s)
        theta = time_gate * raw
        return cfg.coolant_temperature_k + cfg.temperature_scale_k * theta

    def step(
        self,
        state: torch.Tensor,
        measurements_k: torch.Tensor,
        flux_k: torch.Tensor,
        delta_flux_k: torch.Tensor,
        time_k: torch.Tensor,
        dt_k: torch.Tensor,
        x_hat: torch.Tensor,
        cfg: ConstantFluxConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.observe(
            state, measurements_k, flux_k, delta_flux_k, time_k, dt_k, cfg
        )
        return self.decode(state, x_hat, time_k, cfg), state


def load_split(root: Path, split: str, cfg: ConstantFluxConfig):
    rows = read_manifest(root, split)
    scenarios = [load_scenario(root, row, cfg) for row in rows]
    return rows, {
        "field": np.stack([scenario["field"] for scenario in scenarios]),
        "sensor_values": np.stack([scenario["sensor_values"] for scenario in scenarios]),
        "flux": np.stack([scenario["flux"] for scenario in scenarios]),
        "x_hat": scenarios[0]["x_hat"],
        "sensor_x_hat": scenarios[0]["sensor_x_hat"],
        "times": scenarios[0]["times"],
    }


def _sequence_prediction(
    model: ConstantFluxStreamedPINO,
    measurements: torch.Tensor,
    flux: torch.Tensor,
    times: torch.Tensor,
    x_hat: torch.Tensor,
    cfg: ConstantFluxConfig,
) -> torch.Tensor:
    batch = len(measurements)
    state = model.initial_state(batch, measurements.device)
    coordinates = x_hat[None].expand(batch, -1, -1)
    rows = []
    previous = torch.zeros(batch, device=measurements.device)
    previous_flux = flux[:, 0]
    for index, current in enumerate(times):
        dt = current.expand(batch) - previous
        delta_flux = (
            torch.zeros_like(previous_flux)
            if index == 0
            else flux[:, index] - previous_flux
        )
        prediction, state = model.step(
            state,
            measurements[:, index],
            flux[:, index],
            delta_flux,
            current.expand(batch),
            dt,
            coordinates,
            cfg,
        )
        rows.append(prediction)
        previous = current.expand(batch)
        previous_flux = flux[:, index]
    return torch.stack(rows, dim=1)


def _physics_loss(
    prediction: torch.Tensor,
    flux: torch.Tensor,
    times: torch.Tensor,
    x_hat: torch.Tensor,
    cfg: ConstantFluxConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = (prediction - cfg.coolant_temperature_k) / cfg.temperature_scale_k
    dt = (times[1:] - times[:-1])[None, :, None] / cfg.final_time_s
    theta_t = (theta[:, 1:] - theta[:, :-1]) / dt
    dx = x_hat[1, 0] - x_hat[0, 0]
    theta_xx = (
        theta[:, 1:, :-2] - 2.0 * theta[:, 1:, 1:-1] + theta[:, 1:, 2:]
    ) / dx**2
    pde = (theta_t[:, :, 1:-1] - cfg.beta * theta_xx).square().mean()
    hot_gradient = (theta[:, :, 1] - theta[:, :, 0]) / dx
    hot_target = -flux * cfg.wall_thickness_m / (
        cfg.conductivity_w_mk * cfg.temperature_scale_k
    )
    cool_gradient = (theta[:, :, -1] - theta[:, :, -2]) / dx
    boundary = (hot_gradient - hot_target).square().mean()
    boundary = boundary + (cool_gradient + cfg.biot * theta[:, :, -1]).square().mean()
    return pde, boundary


def predict(
    model: ConstantFluxStreamedPINO,
    arrays: dict[str, np.ndarray],
    cfg: ConstantFluxConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return _sequence_prediction(
            model,
            torch.as_tensor(arrays["sensor_values"], device=device),
            torch.as_tensor(arrays["flux"], device=device),
            torch.as_tensor(arrays["times"], device=device),
            torch.as_tensor(arrays["x_hat"][:, None], device=device),
            cfg,
        ).cpu().numpy()


def train_streamed_pino(
    root: Path,
    cfg: ConstantFluxConfig,
    output_dir: Path,
    device: torch.device,
) -> tuple[ConstantFluxStreamedPINO, dict[str, float], dict[str, np.ndarray]]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    train_rows, train_arrays = load_split(root, "train", cfg)
    validation_rows, validation_arrays = load_split(root, "validation", cfg)
    test_rows, test_arrays = load_split(root, "test_interpolation", cfg)
    _, locked_arrays = load_split(root, "test_locked", cfg)
    model = ConstantFluxStreamedPINO(
        sensors=len(cfg.sensor_x_hat), hidden=cfg.operator_hidden, basis=cfg.operator_basis
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.operator_learning_rate, weight_decay=1.0e-6
    )
    rng = np.random.default_rng(cfg.seed)
    order = np.arange(len(train_rows))
    times = torch.as_tensor(train_arrays["times"], device=device)
    x_hat = torch.as_tensor(train_arrays["x_hat"][:, None], device=device)
    best = float("inf")
    best_state = None
    best_epoch = 0
    validation_history: list[tuple[int, float, float]] = []
    for epoch in range(1, cfg.operator_epochs + 1):
        rng.shuffle(order)
        model.train()
        train_squared = 0.0
        train_count = 0
        for start in range(0, len(order), cfg.operator_batch_size):
            selected = order[start : start + cfg.operator_batch_size]
            measurements = torch.as_tensor(train_arrays["sensor_values"][selected], device=device)
            flux = torch.as_tensor(train_arrays["flux"][selected], device=device)
            target = torch.as_tensor(train_arrays["field"][selected], device=device)
            prediction = _sequence_prediction(model, measurements, flux, times, x_hat, cfg)
            normalized_error = (prediction - target) / cfg.temperature_scale_k
            # Put extra weight on samples at and immediately around a large
            # boundary change while retaining every ordinary time sample.
            flux_activity = torch.zeros_like(flux)
            flux_activity[:, 1:] = torch.abs(flux[:, 1:] - flux[:, :-1]) / (
                cfg.q_max_w_m2 - cfg.q_min_w_m2
            )
            neighborhood = flux_activity.clone()
            neighborhood[:, :-1] = torch.maximum(
                neighborhood[:, :-1], 0.5 * flux_activity[:, 1:]
            )
            neighborhood[:, 1:] = torch.maximum(
                neighborhood[:, 1:], 0.5 * flux_activity[:, :-1]
            )
            sample_weights = 1.0 + cfg.operator_transition_weight * neighborhood
            data_loss = (
                normalized_error.square() * sample_weights[:, :, None]
            ).sum() / (sample_weights.sum() * prediction.shape[-1])
            pde_loss, boundary_loss = _physics_loss(prediction, flux, times, x_hat, cfg)
            loss = (
                data_loss
                + cfg.operator_pde_weight * pde_loss
                + cfg.operator_boundary_weight * boundary_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_squared += float((prediction - target).square().sum())
            train_count += prediction.numel()
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.operator_epochs:
            validation_prediction = predict(model, validation_arrays, cfg, device)
            validation_rmse = float(
                np.sqrt(np.mean((validation_prediction - validation_arrays["field"]) ** 2))
            )
            validation_history.append(
                (epoch, float(np.sqrt(train_squared / train_count)), validation_rmse)
            )
            print(
                f"operator epoch={epoch:03d} train_rmse={np.sqrt(train_squared/train_count):.5f} "
                f"validation_rmse={validation_rmse:.5f}",
                flush=True,
            )
            if validation_rmse < best:
                best = validation_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("operator training produced no checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "validation_rmse_k": best,
            "best_epoch": best_epoch,
            "training_scenarios": [row["scenario_id"] for row in train_rows],
        },
        output_dir / "streamed_pino.pt",
    )
    with (output_dir / "streamed_pino_training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "training_rmse_k", "validation_rmse_k"))
        writer.writerows(validation_history)
    test_prediction = predict(model, test_arrays, cfg, device)
    scenario_rmse = np.sqrt(np.mean((test_prediction - test_arrays["field"]) ** 2, axis=(1, 2)))
    modes = np.asarray([row.get("boundary_mode", "step") for row in test_rows])
    with (output_dir / "streamed_pino_interpolation_test.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream); writer.writerow(("scenario_id", "rmse_k"))
        writer.writerows((row["scenario_id"], value) for row, value in zip(test_rows, scenario_rmse))
    locked_prediction = predict(model, locked_arrays, cfg, device)[0]
    # Measure a genuine one-step live update after warm-up.
    state = model.initial_state(1, device)
    coords = torch.as_tensor(locked_arrays["x_hat"][:, None], device=device)[None]
    latencies = []
    previous = 0.0
    previous_flux = float(locked_arrays["flux"][0, 0])
    model.eval()
    with torch.inference_mode():
        for index, tau in enumerate(locked_arrays["times"]):
            if device.type == "cuda": torch.cuda.synchronize()
            tick = time.perf_counter()
            _, state = model.step(
                state,
                torch.as_tensor(locked_arrays["sensor_values"][0, index:index+1], device=device),
                torch.as_tensor(locked_arrays["flux"][0, index:index+1], device=device),
                torch.tensor(
                    [
                        0.0
                        if index == 0
                        else locked_arrays["flux"][0, index] - previous_flux
                    ],
                    device=device,
                ),
                torch.tensor([tau], device=device),
                torch.tensor([tau - previous], device=device),
                coords,
                cfg,
            )
            if device.type == "cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter() - tick) * 1000.0)
            previous = float(tau)
            previous_flux = float(locked_arrays["flux"][0, index])
    metrics = {
        "locked_rmse_k": float(np.sqrt(np.mean((locked_prediction - locked_arrays["field"][0]) ** 2))),
        "interpolation_mean_rmse_k": float(np.mean(scenario_rmse)),
        "interpolation_std_rmse_k": float(np.std(scenario_rmse)),
        "median_latency_ms": float(np.median(latencies)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "validation_rmse_k": best,
        "best_epoch": best_epoch,
        "nondecaying_interpolation_mean_rmse_k": float(
            np.mean(scenario_rmse[modes == "step"])
        ),
        "decaying_interpolation_mean_rmse_k": float(
            np.mean(scenario_rmse[modes == "exponential"])
        ),
    }
    return model, metrics, {
        "prediction": locked_prediction,
        "truth": locked_arrays["field"][0],
        "times": locked_arrays["times"],
        "x_hat": locked_arrays["x_hat"],
    }
