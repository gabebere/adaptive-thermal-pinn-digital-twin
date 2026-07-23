import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytical_solution import boundary_values, initial_condition, temperature
from parameters import CONTINUOUS_BOUNDARY_SETS, PAPER_BOUNDARY_SETS
from reference_data import ReferenceDataset, save_reference_csv


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


def test_continuous_left_heating_reaches_nonzero_long_time_field():
    points = np.array([[0.0, 0.5, 100.0], [0.5, 0.5, 100.0]])
    actual = temperature(
        points, CONTINUOUS_BOUNDARY_SETS["continuous_left"], terms=20
    )[:, 0]
    np.testing.assert_allclose(actual[0], 0.25, atol=2e-12)
    assert actual[1] > 0.0


def test_reference_csv_export(tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 100.0]])
    values = np.array([[0.0], [0.125]])
    dataset = ReferenceDataset(
        points,
        values,
        np.array([0.0, 100.0]),
        np.array([0.0, 0.5]),
        np.array([0.0, 0.5]),
        points,
        values,
    )
    destination = tmp_path / "long_horizon.csv"
    save_reference_csv(dataset, destination)

    exported = np.genfromtxt(destination, delimiter=",", names=True)
    assert exported.dtype.names == ("x", "y", "tau", "temperature")
    np.testing.assert_allclose(exported["tau"], [0.0, 100.0])
    np.testing.assert_allclose(exported["temperature"], values[:, 0])
