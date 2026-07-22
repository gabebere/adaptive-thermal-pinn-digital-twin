from pathlib import Path

from constant_flux_physics import load_balanced_config


ARCHITECTURE = (
    Path(__file__).resolve().parents[2]
    / "architectures"
    / "constant_flux_balanced.toml"
)


def test_constant_flux_preset_retains_validated_training_budget():
    cfg = load_balanced_config(ARCHITECTURE, "full")

    assert cfg.hidden_layers == (48, 48, 48)
    assert cfg.baseline_iterations == 2500
    assert cfg.adaptive_iterations_per_batch == 381
    assert cfg.sensor_x_hat == (0.05, 0.5, 0.95)
    assert cfg.operator_epochs == 500
    assert cfg.operator_hidden == 128
    assert cfg.operator_transition_weight == 3.0
