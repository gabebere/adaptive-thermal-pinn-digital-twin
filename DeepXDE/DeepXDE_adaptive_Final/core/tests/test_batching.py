import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parameters import make_config
from pinn_workflow import make_time_batches


def test_n_instance_batches_cover_every_time_once():
    cfg = make_config("smoke")
    times = np.arange(cfg.time_instances)
    batches = [times[i : i + cfg.batch_size_n] for i in range(0, len(times), cfg.batch_size_n)]
    assert all(len(batch) == cfg.batch_size_n for batch in batches[:-1])
    np.testing.assert_array_equal(np.concatenate(batches), times)


def test_explicit_windows_can_exclude_initial_sensor_time():
    cfg = make_config("full")
    cfg.adaptive_windows = 5
    cfg.exclude_initial_sensor_time = True
    times = np.linspace(0.0, 100.0, 41)
    batches = make_time_batches(times, cfg)
    assert [len(batch) for batch in batches] == [8, 8, 8, 8, 8]
    assert all(not np.any(np.isclose(batch, 0.0)) for batch in batches)
    np.testing.assert_array_equal(np.concatenate(batches), times[1:])
