import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constant_flux_physics import (
    ConstantFluxConfig,
    exponentially_relaxing_flux_temperature,
    steady_temperature,
    switched_flux_temperature,
    unit_flux_step_response,
)


def test_exact_response_has_correct_initial_and_steady_limits():
    cfg = ConstantFluxConfig(series_terms=100)
    x = np.linspace(0.0, cfg.wall_thickness_m, 81)
    response = unit_flux_step_response(x, np.array([0.0, 20.0]), cfg)
    expected = (steady_temperature(x, 1.0, cfg) - cfg.coolant_temperature_k)
    assert np.max(abs(response[0])) == 0.0
    assert np.max(abs(response[1] - expected)) < 1.0e-10


def test_flux_switch_preserves_temperature_continuity():
    cfg = ConstantFluxConfig(series_terms=100)
    x = np.linspace(0.0, cfg.wall_thickness_m, 81)
    values = switched_flux_temperature(
        x, np.array([0.5 - 1e-9, 0.5, 0.5]), 4e6, 5.2e6, 0.5, cfg
    )
    assert np.max(abs(values[1] - values[0])) < 1.0e-4
    assert np.max(abs(values[2] - values[1])) < 1.0e-4


def test_exponential_flux_response_is_continuous_and_reaches_terminal_steady():
    cfg = ConstantFluxConfig(series_terms=100)
    x = np.linspace(0.0, cfg.wall_thickness_m, 81)
    switch = 0.5
    values = exponentially_relaxing_flux_temperature(
        x,
        np.array([switch - 1.0e-9, switch, switch + 1.0e-9, 10.0]),
        4.0e6,
        5.0e6,
        3.2e6,
        switch,
        2.0,
        cfg,
    )
    assert np.max(np.abs(values[1] - values[0])) < 1.0e-4
    # A finite 100-mode truncation has a small boundary Gibbs error immediately
    # after a discontinuous flux jump; it remains continuous to far below 0.1 K.
    assert np.max(np.abs(values[2] - values[1])) < 5.0e-2
    terminal = steady_temperature(x, 3.2e6, cfg)
    assert np.max(np.abs(values[-1] - terminal)) < 1.0e-3
