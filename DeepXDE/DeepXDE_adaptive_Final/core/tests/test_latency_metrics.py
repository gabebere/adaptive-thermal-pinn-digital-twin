import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latency_study import _first_event_update, _settling_metrics
from parameters import (
    LATENCY_EXPERIMENTS,
    config_for_adaptive_profile,
    config_for_latency_experiment,
    make_config,
)


def test_first_event_update_counts_changed_samples_in_batch():
    times = np.arange(0.0, 12.5, 2.5)
    history = [{"time_end": 10.0}]
    update_tau, samples = _first_event_update(history, times, switch_tau=5.0)
    assert update_tau == 10.0
    assert samples == 3


def test_settling_metric_uses_causal_error_drop():
    times = np.arange(8.0)
    errors = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 3.0, 2.0, 1.5])
    metrics = _settling_metrics(times, errors, 4.0, fraction=0.5, consecutive=2)
    assert metrics["recovery_target_error"] == 3.0
    assert metrics["recovery_tau"] == 5.0
    assert metrics["recovery_delay_intervals"] == 1


def test_settling_cannot_precede_first_effective_update():
    times = np.arange(9.0)
    errors = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 2.0, 2.0, 1.5, 1.4])
    metrics = _settling_metrics(
        times,
        errors,
        4.0,
        fraction=0.5,
        consecutive=2,
        earliest_update_tau=6.0,
    )
    assert metrics["recovery_tau"] == 7.0


def test_low_latency_configuration_is_reproducible():
    base = make_config("full")
    combined = next(row for row in LATENCY_EXPERIMENTS if row.name == "low_latency")
    cfg = config_for_latency_experiment(base, combined)
    assert cfg.time_instances == 81
    assert cfg.batch_size_n == 1
    assert cfg.observation_window_batches == 4
    assert cfg.sensor_x == (0.05, 0.275, 0.5, 0.725, 0.95)
    assert len(cfg.sensor_x) * len(cfg.sensor_y) == 25


def test_balanced_profile_is_the_maintained_default_tradeoff():
    cfg = config_for_adaptive_profile(make_config("full"), "balanced")
    assert cfg.time_instances == 41
    assert cfg.batch_size_n == 2
    assert cfg.sensor_x == (0.05, 0.5, 0.95)
    assert cfg.sensor_y == (0.05, 0.5, 0.95)
    assert cfg.observation_window_batches == 2
    assert cfg.data_loss_weight == 10.0
    assert cfg.adaptive_iterations_per_batch == 100


def test_maintained_profiles_scale_to_smoke_horizon():
    for profile_name in ("balanced", "low_latency"):
        cfg = config_for_adaptive_profile(make_config("smoke"), profile_name)
        assert cfg.time_instances == 11
        assert cfg.adaptive_iterations_per_batch == 10
