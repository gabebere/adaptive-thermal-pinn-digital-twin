import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constant_flux_physics import ConstantFluxConfig
from physical_pinn_workflow import make_problem


def test_sensor_data_constraint_is_added_only_to_adaptive_problem():
    cfg = ConstantFluxConfig(num_domain=8, num_boundary=4, num_initial=4)
    flux = lambda time: np.full((len(time), 1), cfg.baseline_flux_w_m2)
    offline = make_problem(cfg, flux)
    adaptive = make_problem(
        cfg,
        flux,
        observation_points=np.array([[0.5, 0.1], [0.5, 0.2]]),
        observation_values=np.array([[325.0], [335.0]]),
    )

    assert len(offline.bcs) == 3
    assert len(adaptive.bcs) == 4
