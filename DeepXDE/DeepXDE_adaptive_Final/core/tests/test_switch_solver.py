import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytical_solution import boundary_values, temperature
from parameters import make_config
from switch_solver import generate_switch_dataset, save_switch_csv


def test_switch_dataset_shapes_and_finiteness():
    cfg = make_config("smoke")
    data = generate_switch_dataset(cfg)
    assert data.field_points.shape[1] == 3
    assert data.field_values.shape[0] == data.field_points.shape[0]
    assert data.sensor_values.shape[0] == data.sensor_points.shape[0]
    assert np.isfinite(data.field_values).all()
    assert np.isclose(data.switch_tau, cfg.tau_final * cfg.switch_fraction)


def test_pre_event_field_uses_verified_analytical_solution():
    cfg = make_config("smoke")
    data = generate_switch_dataset(cfg)
    mask = data.field_points[:, 2] < data.switch_tau
    expected = temperature(data.field_points[mask], cfg.boundary_set, cfg.series_terms)
    np.testing.assert_allclose(data.field_values[mask], expected, atol=2e-12)


def test_event_preserves_interior_and_applies_new_boundaries():
    cfg = make_config("smoke")
    data = generate_switch_dataset(cfg)
    event = np.isclose(data.field_points[:, 2], data.switch_tau)
    points = data.field_points[event]
    values = data.field_values[event, 0]

    center = np.isclose(points[:, 0], 0.5) & np.isclose(points[:, 1], 0.5)
    expected_center = temperature(
        np.array([[0.5, 0.5, data.switch_tau]]),
        cfg.boundary_set,
        cfg.series_terms,
    )[0, 0]
    np.testing.assert_allclose(values[center], expected_center, atol=2e-12)

    left = np.isclose(points[:, 0], 0.0)
    local_points = points[left].copy()
    local_points[:, 2] = 0.0
    expected_left = boundary_values(local_points, cfg.changed_boundary_set)["left"]
    np.testing.assert_allclose(values[left], expected_left, atol=2e-12)


def test_switch_csv_export(tmp_path):
    data = generate_switch_dataset(make_config("smoke"))
    destination = tmp_path / "boundary_change.csv"
    save_switch_csv(data, destination)

    exported = np.genfromtxt(destination, delimiter=",", names=True)
    assert exported.dtype.names == (
        "x",
        "y",
        "tau",
        "temperature",
        "is_post_switch",
    )
    assert len(exported) == len(data.field_points)
    assert np.all(exported["is_post_switch"] == (exported["tau"] >= data.switch_tau))
