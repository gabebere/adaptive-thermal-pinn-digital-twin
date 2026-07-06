from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig
from .reference import ReferenceSolution


@dataclass
class EvaluationResult:
    T_pred: np.ndarray
    error: np.ndarray
    relative_l2_global: float
    relative_l2_time: np.ndarray
    max_abs_error: float
    hot_side_max_abs_error: float


def evaluate_model(model: nn.Module, reference: ReferenceSolution, cfg: ExperimentConfig) -> EvaluationResult:
    model.eval()
    x_hat = reference.x / cfg.L_wall
    t_hat = reference.t / cfg.t_final
    X, TT = np.meshgrid(x_hat, t_hat)
    points = np.stack([X.reshape(-1), TT.reshape(-1)], axis=1).astype(np.float32)

    theta_pred_parts = []
    with torch.no_grad():
        for start in range(0, len(points), cfg.eval_chunk_size):
            chunk = torch.as_tensor(points[start : start + cfg.eval_chunk_size], device=cfg.device)
            theta = model(chunk[:, 0:1], chunk[:, 1:2]).detach().cpu().numpy()
            theta_pred_parts.append(theta)

    theta_pred = np.vstack(theta_pred_parts).reshape(reference.T.shape)
    T_pred = cfg.T_cool + cfg.delta_T * theta_pred
    error = T_pred - reference.T

    response = reference.T - cfg.T_cool
    denom = np.linalg.norm(response) + 1.0e-12
    relative_l2_global = float(np.linalg.norm(error) / denom)
    relative_l2_time = np.linalg.norm(error, axis=1) / (np.linalg.norm(response, axis=1) + 1.0e-6)
    return EvaluationResult(
        T_pred=T_pred,
        error=error,
        relative_l2_global=relative_l2_global,
        relative_l2_time=relative_l2_time,
        max_abs_error=float(np.max(np.abs(error))),
        hot_side_max_abs_error=float(np.max(np.abs(error[:, 0]))),
    )
