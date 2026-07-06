from __future__ import annotations

import numpy as np
import torch

from thermal_pinn import evaluate_model, generate_reference_solution, make_config, pinn_model, simulate_sensor_data
from thermal_pinn.losses import pde_residual


def test_reference_solution_shapes_and_finite_values(tmp_path):
    cfg = make_config("smoke", tmp_path)
    cfg.nx = 17
    cfg.nt = 21
    reference = generate_reference_solution(cfg)
    assert reference.T.shape == (cfg.nt, cfg.nx)
    assert reference.theta.shape == (cfg.nt, cfg.nx)
    assert np.isfinite(reference.T).all()
    assert reference.T[-1, 0] > reference.T[0, 0]


def test_pinn_forward_and_residual_are_finite(tmp_path):
    cfg = make_config("smoke", tmp_path)
    model = pinn_model(cfg)
    x = torch.rand((8, 1))
    t = torch.rand((8, 1))
    y = model(x, t)
    residual = pde_residual(model, x, t, cfg)
    assert y.shape == (8, 1)
    assert residual.shape == (8, 1)
    assert torch.isfinite(residual).all()


def test_sensor_sampler_and_evaluation(tmp_path):
    cfg = make_config("smoke", tmp_path)
    cfg.nx = 17
    cfg.nt = 25
    reference = generate_reference_solution(cfg)
    batches = simulate_sensor_data(reference, cfg, noise_level=0.0)
    assert len(batches) == cfg.sensor_windows
    assert batches[0].x_hat.shape[1] == 1
    assert batches[0].theta.shape == batches[0].x_hat.shape

    model = pinn_model(cfg)
    result = evaluate_model(model, reference, cfg)
    assert np.isfinite(result.relative_l2_global)
    assert np.isfinite(result.relative_l2_time).all()
