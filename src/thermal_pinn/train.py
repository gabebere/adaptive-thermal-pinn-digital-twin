from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .losses import boundary_losses, initial_condition_loss, pde_residual
from .model import pinn_model
from .sensors import SensorBatch


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict[str, float]]
    runtime_s: float


def _rand(n: int, cfg: ExperimentConfig) -> torch.Tensor:
    return torch.rand((n, 1), device=cfg.device)


def _sensor_tensors(batch: SensorBatch, cfg: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(batch.x_hat, dtype=torch.float32, device=cfg.device),
        torch.as_tensor(batch.t_hat, dtype=torch.float32, device=cfg.device),
        torch.as_tensor(batch.theta, dtype=torch.float32, device=cfg.device),
    )


def _loss_components(
    model: nn.Module,
    cfg: ExperimentConfig,
    sensor_batches: list[SensorBatch] | None = None,
) -> dict[str, torch.Tensor]:
    x_pde = _rand(cfg.n_pde, cfg)
    t_pde = _rand(cfg.n_pde, cfg)
    residual = pde_residual(model, x_pde, t_pde, cfg)
    loss_pde = torch.mean(residual**2)

    t_bc = _rand(cfg.n_bc, cfg)
    loss_hot, loss_cool = boundary_losses(model, t_bc, cfg)
    loss_bc = loss_hot + loss_cool

    x_ic = _rand(cfg.n_ic, cfg)
    loss_ic = initial_condition_loss(model, x_ic, cfg)

    loss_sensor = torch.zeros((), device=cfg.device)
    if sensor_batches:
        sensor_losses = []
        for batch in sensor_batches:
            x_s, t_s, y_s = _sensor_tensors(batch, cfg)
            sensor_losses.append(torch.mean((model(x_s, t_s) - y_s) ** 2))
        loss_sensor = torch.mean(torch.stack(sensor_losses))

    total = (
        cfg.w_pde * loss_pde
        + cfg.w_bc * loss_bc
        + cfg.w_ic * loss_ic
        + cfg.w_sensor * loss_sensor
    )
    return {
        "total": total,
        "pde": loss_pde,
        "bc": loss_bc,
        "ic": loss_ic,
        "sensor": loss_sensor,
    }


def _train(
    model: nn.Module,
    cfg: ExperimentConfig,
    epochs: int,
    sensor_batches: list[SensorBatch] | None = None,
    phase: str = "baseline",
) -> TrainResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = _loss_components(model, cfg, sensor_batches)
        losses["total"].backward()
        optimizer.step()

        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 10) == 0:
            row = {name: float(value.detach().cpu()) for name, value in losses.items()}
            row["epoch"] = float(epoch + 1)
            row["phase"] = phase
            history.append(row)

    return TrainResult(model=model, history=history, runtime_s=time.perf_counter() - start)


def train_baseline_pinn(cfg: ExperimentConfig) -> TrainResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = pinn_model(cfg)
    return _train(model, cfg, cfg.baseline_epochs, phase="baseline")


def update_adaptive_pinn(
    model: nn.Module,
    sensor_batches: list[SensorBatch],
    cfg: ExperimentConfig,
) -> TrainResult:
    """Sequentially update a baseline model as sensor windows arrive."""
    adaptive = deepcopy(model).to(cfg.device)
    full_history: list[dict[str, float]] = []
    total_runtime = 0.0
    seen_batches: list[SensorBatch] = []
    epoch_offset = 0

    for batch in sensor_batches:
        seen_batches.append(batch)
        result = _train(
            adaptive,
            cfg,
            cfg.update_epochs,
            sensor_batches=seen_batches,
            phase=f"adaptive_window_{batch.window_index}",
        )
        total_runtime += result.runtime_s
        for row in result.history:
            adjusted = dict(row)
            adjusted["epoch"] = float(adjusted["epoch"] + epoch_offset)
            full_history.append(adjusted)
        epoch_offset += cfg.update_epochs

    return TrainResult(model=adaptive, history=full_history, runtime_s=total_runtime)
