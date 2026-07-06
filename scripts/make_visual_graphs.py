#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from thermal_pinn import generate_reference_solution, make_config, pinn_model, simulate_sensor_data
from thermal_pinn.evaluate import evaluate_model
from thermal_pinn.visuals import make_visual_explanation_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create explanatory plots from saved smoke/full model outputs.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_config(args.mode, args.output_dir)
    cfg.device = args.device

    config_path = args.output_dir / "results" / "config.json"
    if config_path.exists():
        data = json.loads(config_path.read_text())
        cfg.main_noise_level = float(data.get("main_noise_level", cfg.main_noise_level))

    reference = generate_reference_solution(cfg)
    baseline = pinn_model(cfg)
    adaptive = pinn_model(cfg)
    baseline.load_state_dict(torch.load(args.output_dir / "results" / "baseline_model.pt", map_location=args.device))
    adaptive.load_state_dict(torch.load(args.output_dir / "results" / "adaptive_model.pt", map_location=args.device))

    baseline_eval = evaluate_model(baseline, reference, cfg)
    adaptive_eval = evaluate_model(adaptive, reference, cfg)
    batches = simulate_sensor_data(reference, cfg, noise_level=cfg.main_noise_level)
    paths = make_visual_explanation_plots(reference, baseline_eval, adaptive_eval, batches, cfg)
    print("Created visual explainer graphs:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
