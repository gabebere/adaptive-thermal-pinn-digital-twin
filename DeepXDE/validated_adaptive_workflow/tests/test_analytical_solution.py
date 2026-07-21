import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytical_solution import boundary_values, initial_condition, temperature
from parameters import PAPER_BOUNDARY_SETS


def test_initial_condition_inside_domain():
    xy = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.4]])
    points = np.column_stack((xy, np.zeros(len(xy))))
    actual = temperature(points, PAPER_BOUNDARY_SETS["set_1"], terms=12)[:, 0]
    expected = initial_condition(xy[:, 0], xy[:, 1])
    np.testing.assert_allclose(actual, expected, atol=2e-12)


def test_all_four_boundaries():
    tau = np.array([0.0, 0.15, 0.8])
    interior = np.array([0.2, 0.5, 0.8])
    boundaries = PAPER_BOUNDARY_SETS["set_3"]
    cases = {
        "left": np.column_stack((np.zeros(3), interior, tau)),
        "right": np.column_stack((np.ones(3), interior, tau)),
        "bottom": np.column_stack((interior, np.zeros(3), tau)),
        "top": np.column_stack((interior, np.ones(3), tau)),
    }
    for name, points in cases.items():
        expected = boundary_values(points, boundaries)[name]
        actual = temperature(points, boundaries, terms=12)[:, 0]
        np.testing.assert_allclose(actual, expected, atol=2e-12)
