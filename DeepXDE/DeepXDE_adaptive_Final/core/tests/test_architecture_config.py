from pathlib import Path

import pytest

from parameters import load_architecture_file, make_config


ARCHITECTURES = (
    Path(__file__).resolve().parents[2]
    / "architectures"
)


def test_balanced_architecture_document_matches_maintained_configuration():
    cfg = load_architecture_file(make_config("full"), ARCHITECTURES / "balanced.toml")

    assert cfg.time_instances == 41
    assert cfg.batch_size_n == 2
    assert cfg.sensor_x == (0.05, 0.5, 0.95)
    assert cfg.observation_window_batches == 2
    assert cfg.adaptive_iterations_per_batch == 100
    assert cfg.initializer == "Glorot normal"


def test_pytorch_comparison_document_matches_source_model_choices():
    cfg = load_architecture_file(
        make_config("full"), ARCHITECTURES / "pytorch_comparison.toml"
    )

    assert cfg.hidden_layers == (48, 48, 48)
    assert cfg.activation == "tanh"
    assert cfg.initializer == "Glorot uniform"
    assert cfg.baseline_iterations == 1200
    assert cfg.adaptive_iterations_per_batch == 160
    assert cfg.baseline_learning_rate == 2.0e-3
    assert cfg.adaptive_learning_rate == 5.0e-4
    assert cfg.pde_loss_weight == 1.0
    assert cfg.boundary_loss_weight == 2.0
    assert cfg.initial_loss_weight == 5.0
    assert cfg.sensor_x == (0.05, 0.25, 0.5, 0.75)
    assert cfg.sensor_y == (0.5,)
    assert cfg.observation_window_batches is None


def test_unknown_architecture_setting_is_rejected(tmp_path):
    path = tmp_path / "typo.toml"
    path.write_text("[network]\nactivaton = 'tanh'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="activaton"):
        load_architecture_file(make_config("full"), path)


def test_smoke_profile_caps_compute_but_keeps_architecture():
    base = make_config("smoke")
    cfg = load_architecture_file(
        base, ARCHITECTURES / "low_latency.toml", smoke=True
    )

    assert cfg.hidden_layers == (48, 48, 48)
    assert cfg.time_instances == 11
    assert cfg.baseline_iterations == 80
    assert cfg.adaptive_iterations_per_batch == 10
    assert cfg.num_domain == 180
