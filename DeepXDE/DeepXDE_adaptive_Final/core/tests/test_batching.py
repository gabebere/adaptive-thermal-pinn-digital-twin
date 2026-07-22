import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parameters import make_config


def test_n_instance_batches_cover_every_time_once():
    cfg = make_config("smoke")
    times = np.arange(cfg.time_instances)
    batches = [times[i : i + cfg.batch_size_n] for i in range(0, len(times), cfg.batch_size_n)]
    assert all(len(batch) == cfg.batch_size_n for batch in batches[:-1])
    np.testing.assert_array_equal(np.concatenate(batches), times)
