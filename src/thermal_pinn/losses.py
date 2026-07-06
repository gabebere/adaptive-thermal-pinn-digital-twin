from __future__ import annotations

import torch
from torch import nn

from .config import ExperimentConfig


def q_hat(cfg: ExperimentConfig, t_hat: torch.Tensor) -> torch.Tensor:
    t_phys = t_hat * cfg.t_final
    q = cfg.q_base + cfg.q_amp * torch.sin(2.0 * torch.pi * t_phys / cfg.q_period)
    return q * cfg.L_wall / (cfg.k * cfg.delta_T)


def pde_residual(model: nn.Module, x_hat: torch.Tensor, t_hat: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    """Return nondimensional heat-equation residual."""
    x_hat = x_hat.detach().clone().requires_grad_(True)
    t_hat = t_hat.detach().clone().requires_grad_(True)
    theta = model(x_hat, t_hat)
    theta_x = torch.autograd.grad(theta, x_hat, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    theta_t = torch.autograd.grad(theta, t_hat, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    theta_xx = torch.autograd.grad(theta_x, x_hat, grad_outputs=torch.ones_like(theta_x), create_graph=True)[0]
    return theta_t - cfg.beta * theta_xx


def boundary_losses(model: nn.Module, t_hat: torch.Tensor, cfg: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = torch.zeros_like(t_hat, requires_grad=True)
    x1 = torch.ones_like(t_hat, requires_grad=True)

    theta_hot = model(x0, t_hat)
    theta_x_hot = torch.autograd.grad(
        theta_hot, x0, grad_outputs=torch.ones_like(theta_hot), create_graph=True
    )[0]
    hot_residual = -theta_x_hot - q_hat(cfg, t_hat)

    theta_cool = model(x1, t_hat)
    theta_x_cool = torch.autograd.grad(
        theta_cool, x1, grad_outputs=torch.ones_like(theta_cool), create_graph=True
    )[0]
    cool_residual = -theta_x_cool - cfg.biot * theta_cool
    return torch.mean(hot_residual**2), torch.mean(cool_residual**2)


def initial_condition_loss(model: nn.Module, x_hat: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    t0 = torch.zeros_like(x_hat)
    target = torch.full_like(x_hat, (cfg.T0 - cfg.T_cool) / cfg.delta_T)
    return torch.mean((model(x_hat, t0) - target) ** 2)
