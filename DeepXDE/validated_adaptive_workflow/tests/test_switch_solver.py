import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
