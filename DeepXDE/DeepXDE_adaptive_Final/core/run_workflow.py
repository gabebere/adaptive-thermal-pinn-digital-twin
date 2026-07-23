"""Run the complete validated analytical -> offline PINN -> adaptive workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from literature_validation import validate_against_literature
from parameters import (
    MAINTAINED_ADAPTIVE_PROFILES,
    config_for_adaptive_profile,
    load_architecture_file,
    make_config,
)
from pinn_workflow import adapt_online, predict, rmse, train_baseline
from plots import (
    plot_adaptive_comparison,
    plot_offline_comparison,
    plot_streaming_map,
    plot_switch_test,
    save_metrics,
)
from reference_data import generate_reference_dataset
from switch_solver import generate_switch_dataset, save_switch_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    configuration = parser.add_mutually_exclusive_group()
    configuration.add_argument(
        "--adaptive-profile",
        choices=tuple(MAINTAINED_ADAPTIVE_PROFILES),
        help="built-in maintained online configuration",
    )
    configuration.add_argument(
        "--architecture-file",
        type=Path,
        help="TOML file containing network, training, and streaming choices",
    )
    parser.add_argument("--paper-tables-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    base = make_config(args.profile)
    if args.architecture_file is not None:
        cfg = load_architecture_file(
            base, args.architecture_file, smoke=args.profile == "smoke"
        )
        configuration_name = args.architecture_file.stem
    else:
        configuration_name = args.adaptive_profile or "balanced"
        cfg = config_for_adaptive_profile(base, configuration_name)
    if args.paper_tables_dir is not None:
        cfg.paper_tables_dir = args.paper_tables_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    cfg.validate()
    output_dir = (
        cfg.output_dir / args.profile
        if configuration_name == "balanced"
        else cfg.output_dir / configuration_name / args.profile
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.architecture_file is not None:
        shutil.copyfile(args.architecture_file, output_dir / "architecture_used.toml")

    print("[1/6] Validating the general analytical solution against the paper...")
    literature_metrics = validate_against_literature(cfg, output_dir)

    print("[2/6] Sampling the validated analytical field...")
    reference = generate_reference_dataset(cfg, output_dir)

    print("[3/6] Training and evaluating the offline physics-only PINN...")
    baseline = train_baseline(
        cfg, reference.field_points, reference.field_values, reference.times
    )
    plot_offline_comparison(cfg, reference, baseline, output_dir)

    streaming_label = (
        f"in {cfg.adaptive_windows} explicit windows"
        if cfg.adaptive_windows is not None
        else f"every n={cfg.batch_size_n} instances"
    )
    print(f"[4/6] Assimilating analytical sensor data {streaming_label}...")
    adaptive = adapt_online(
        cfg,
        baseline.network_state,
        reference.sensor_points,
        reference.sensor_values,
        reference.field_points,
        reference.field_values,
        reference.times,
    )
    plot_streaming_map(cfg, reference, output_dir)
    plot_adaptive_comparison(cfg, reference, baseline, adaptive, output_dir)

    print("[5/6] Generating the analytical boundary-change reference...")
    switch_data = generate_switch_dataset(cfg)
    np.savez_compressed(
        output_dir / "06_boundary_change_reference.npz",
        field_points=switch_data.field_points,
        field_values=switch_data.field_values,
        sensor_points=switch_data.sensor_points,
        sensor_values=switch_data.sensor_values,
        times=switch_data.times,
        switch_tau=switch_data.switch_tau,
    )
    save_switch_csv(switch_data, output_dir / "06_boundary_change_reference.csv")
    switch_baseline_prediction = predict(baseline.model, switch_data.field_points, cfg)
    switch_adaptive = adapt_online(
        cfg,
        baseline.network_state,
        switch_data.sensor_points,
        switch_data.sensor_values,
        switch_data.field_points,
        switch_data.field_values,
        switch_data.times,
    )
    plot_switch_test(
        cfg, switch_data, switch_baseline_prediction, switch_adaptive, output_dir
    )

    print("[6/6] Saving metrics...")
    save_metrics(
        cfg,
        literature_metrics,
        baseline,
        adaptive,
        rmse(switch_data.field_values, switch_baseline_prediction),
        switch_adaptive,
        output_dir,
    )
    print(f"Offline global RMSE: {baseline.field_rmse:.6g}")
    print(f"Adaptive global RMSE: {adaptive.field_rmse:.6g}")
    print(f"Boundary-switch adaptive RMSE: {switch_adaptive.field_rmse:.6g}")
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

