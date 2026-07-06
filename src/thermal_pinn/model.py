from __future__ import annotations

import torch
from torch import nn

from .config import ExperimentConfig


class PINN(nn.Module):
    """Small MLP mapping nondimensional (x_hat, t_hat) to theta."""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        if cfg.activation != "tanh":
            raise ValueError("Only tanh activation is currently supported.")

        layers: list[nn.Module] = []
        in_features = 2
        for _ in range(cfg.hidden_layers):
            layers.append(nn.Linear(in_features, cfg.hidden_units))
            layers.append(nn.Tanh())
            in_features = cfg.hidden_units
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_hat, t_hat], dim=1))


def pinn_model(cfg: ExperimentConfig) -> PINN:
    return PINN(cfg).to(cfg.device)
