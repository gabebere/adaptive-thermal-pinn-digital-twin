from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ExperimentConfig
from .evaluate import evaluate_model
from .plots import make_plots
from .reference import generate_reference_solution
from .sensors import simulate_sensor_data
from .train import train_baseline_pinn, update_adaptive_pinn
from .visuals import make_visual_explanation_plots


def _serializable_config(cfg: ExperimentConfig) -> dict[str, object]:
    data = asdict(cfg)
    data["output_dir"] = str(cfg.output_dir)
    return data


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    """Run reference generation, baseline training, adaptive updates, and plotting."""
    start = time.perf_counter()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "results").mkdir(parents=True, exist_ok=True)
    _write_json(cfg.output_dir / "results" / "config.json", _serializable_config(cfg))

    reference = generate_reference_solution(cfg)
    baseline = train_baseline_pinn(cfg)
    baseline_eval = evaluate_model(baseline.model, reference, cfg)

    noise_summary: list[dict[str, float]] = []
    adaptive_results = {}
    main_adaptive = None
    main_model = None
    main_history = None

    for noise_level in cfg.noise_levels:
        batches = simulate_sensor_data(reference, cfg, noise_level=noise_level)
        adaptive = update_adaptive_pinn(baseline.model, batches, cfg)
        adaptive_eval = evaluate_model(adaptive.model, reference, cfg)
        row = {
            "noise_level": float(noise_level),
            "relative_l2_global": adaptive_eval.relative_l2_global,
            "max_abs_error": adaptive_eval.max_abs_error,
            "hot_side_max_abs_error": adaptive_eval.hot_side_max_abs_error,
            "adaptive_update_runtime_s": adaptive.runtime_s,
        }
        noise_summary.append(row)
        adaptive_results[float(noise_level)] = (adaptive, adaptive_eval)
        if abs(noise_level - cfg.main_noise_level) < 1.0e-12:
            main_adaptive = adaptive_eval
            main_model = adaptive.model
            main_history = adaptive.history

    if main_adaptive is None:
        chosen_noise = float(cfg.noise_levels[0])
        main_adaptive = adaptive_results[chosen_noise][1]
        main_model = adaptive_results[chosen_noise][0].model
        main_history = adaptive_results[chosen_noise][0].history

    histories = {"baseline": baseline.history, "adaptive": main_history or []}
    figure_paths = make_plots(reference, baseline_eval, main_adaptive, histories, noise_summary, cfg)
    main_batches = simulate_sensor_data(reference, cfg, noise_level=cfg.main_noise_level)
    figure_paths.extend(make_visual_explanation_plots(reference, baseline_eval, main_adaptive, main_batches, cfg))

    metrics = {
        "baseline": {
            "relative_l2_global": baseline_eval.relative_l2_global,
            "max_abs_error": baseline_eval.max_abs_error,
            "hot_side_max_abs_error": baseline_eval.hot_side_max_abs_error,
            "training_runtime_s": baseline.runtime_s,
        },
        "adaptive_main_noise": {
            "noise_level": cfg.main_noise_level,
            "relative_l2_global": main_adaptive.relative_l2_global,
            "max_abs_error": main_adaptive.max_abs_error,
            "hot_side_max_abs_error": main_adaptive.hot_side_max_abs_error,
        },
        "noise_summary": noise_summary,
        "total_runtime_s": time.perf_counter() - start,
        "figures": [str(path) for path in figure_paths],
    }
    _write_json(cfg.output_dir / "results" / "metrics.json", metrics)
    _write_history(cfg.output_dir / "results" / "baseline_history.csv", baseline.history)
    _write_history(cfg.output_dir / "results" / "adaptive_history.csv", main_history or [])
    _write_history(cfg.output_dir / "results" / "noise_summary.csv", noise_summary)
    torch.save(baseline.model.state_dict(), cfg.output_dir / "results" / "baseline_model.pt")
    if main_model is not None:
        torch.save(main_model.state_dict(), cfg.output_dir / "results" / "adaptive_model.pt")
    return metrics
