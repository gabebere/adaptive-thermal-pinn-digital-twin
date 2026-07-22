"""Run the paper check and constant-flux PINN/PINO comparison end to end."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from constant_flux_physics import (
    analytical_validation,
    generate_csv_corpus,
    load_balanced_config,
    load_scenario,
    read_manifest,
    steady_temperature,
    switched_flux_temperature,
)
from literature_validation import validate_against_literature
from parameters import make_config
from physical_pinn_workflow import adapt_balanced, train_baseline
from physical_streamed_pino import train_streamed_pino


HERE = Path(__file__).resolve().parent
FINAL_ROOT = HERE.parent
REPOSITORY_ROOT = FINAL_ROOT.parents[1]


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _save_analytical_plot(cfg, graph_dir: Path) -> None:
    x_hat = np.linspace(0.0, 1.0, cfg.field_points)
    x_m = x_hat * cfg.wall_thickness_m
    selected = np.asarray([0.0, 0.25, 0.5, 0.525, 0.75, 1.0])
    exact = switched_flux_temperature(
        x_m,
        selected,
        cfg.baseline_flux_w_m2,
        cfg.switched_flux_w_m2,
        cfg.switch_time_s,
        cfg,
    )
    steady_before = steady_temperature(x_m, cfg.baseline_flux_w_m2, cfg)
    steady_after = steady_temperature(x_m, cfg.switched_flux_w_m2, cfg)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for index, tau in enumerate(selected):
        ax.plot(x_hat, exact[index], label=f"exact t={tau:g} s")
    ax.plot(x_hat, steady_before, "k--", alpha=0.6, label="4.0 MW/m² steady")
    ax.plot(x_hat, steady_after, "k:", alpha=0.8, label="5.2 MW/m² steady")
    ax.axvspan(0.0, 0.03, color="tab:red", alpha=0.08, label="hot-flux face")
    ax.set(
        title="Exact constant-flux / convective-wall analytical solution",
        xlabel="Normalized wall coordinate, x/L",
        ylabel="Temperature (K)",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(graph_dir / "00b_physical_analytical_solution.png", dpi=200)
    plt.close(fig)


def _plot_corpus_coverage(corpus: Path, graph_dir: Path) -> dict:
    with (corpus / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    switch = np.asarray([float(row["switch_time_s"]) for row in rows])
    before = np.asarray([float(row["q_before_w_m2"]) for row in rows]) / 1.0e6
    peak = np.asarray([float(row["q_peak_w_m2"]) for row in rows]) / 1.0e6
    terminal = np.asarray([float(row["q_terminal_w_m2"]) for row in rows]) / 1.0e6
    decay_rate = np.asarray([float(row["decay_rate_s"]) for row in rows])
    exponential = np.asarray([row["boundary_mode"] == "exponential" for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), layout="constrained")
    for mask, label, color in (
        (~exponential, "non-decaying step", "tab:blue"),
        (exponential, "exponentially decaying", "tab:red"),
    ):
        axes[0].scatter(switch[mask], (peak - before)[mask], s=15, alpha=0.65, label=label, color=color)
        axes[1].scatter(before[mask], peak[mask], s=15, alpha=0.65, label=label, color=color)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(title="Event time and signed jump", xlabel="switch time (s)", ylabel="q_peak - q_before (MW/m²)")
    axes[1].plot([2.7, 5.3], [2.7, 5.3], "k--", linewidth=0.8)
    axes[1].set(title="Boundary levels", xlabel="q_before (MW/m²)", ylabel="q_peak (MW/m²)")
    scatter = axes[2].scatter(
        peak[exponential],
        terminal[exponential],
        c=decay_rate[exponential],
        cmap="viridis",
        s=18,
        alpha=0.75,
    )
    axes[2].plot([2.7, 5.3], [2.7, 5.3], "k--", linewidth=0.8)
    axes[2].set(title="Decaying cases", xlabel="q_peak (MW/m²)", ylabel="q_terminal (MW/m²)")
    colorbar = fig.colorbar(scatter, ax=axes[2])
    colorbar.set_label("decay rate (1/s)")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Stratified PINO training-corpus coverage")
    fig.savefig(graph_dir / "00c_training_corpus_coverage.png", dpi=200)
    plt.close(fig)
    return {
        "training_scenarios": len(rows),
        "nondecaying_scenarios": int((~exponential).sum()),
        "decaying_scenarios": int(exponential.sum()),
        "switch_time_min_s": float(switch.min()),
        "switch_time_max_s": float(switch.max()),
        "signed_jump_min_mw_m2": float((peak - before).min()),
        "signed_jump_max_mw_m2": float((peak - before).max()),
        "decay_rate_min_s": float(decay_rate[exponential].min()),
        "decay_rate_max_s": float(decay_rate[exponential].max()),
    }


def _plot_baseline_adaptive(
    x_hat, times, truth, baseline, adaptive, graph_dir: Path
) -> None:
    baseline_error = np.abs(baseline - truth)
    adaptive_error = np.abs(adaptive - truth)
    positive = np.concatenate((baseline_error.ravel(), adaptive_error.ravel()))
    positive = positive[positive > 0]
    common_min = max(float(np.percentile(positive, 0.5)), 1.0e-4)
    common_max = max(float(np.percentile(positive, 99.9)), common_min * 10.0)
    norm = LogNorm(vmin=common_min, vmax=common_max)
    baseline_rmse = np.sqrt(np.mean((baseline - truth) ** 2, axis=1))
    adaptive_rmse = np.sqrt(np.mean((adaptive - truth) ** 2, axis=1))
    fig = plt.figure(figsize=(16, 4.7), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 0.045, 1.0))
    axes = (fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), fig.add_subplot(grid[3]))
    color_axis = fig.add_subplot(grid[2])
    extent = [x_hat.min(), x_hat.max(), times.min(), times.max()]
    images = []
    for ax, values, title in (
        (axes[0], baseline_error, "Offline baseline PINN |error|"),
        (axes[1], adaptive_error, "Adaptive balanced PINN |error|"),
    ):
        images.append(
            ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
                norm=norm,
            )
        )
        ax.axhline(0.5, color="cyan", linestyle="--", linewidth=1.5)
        ax.set(title=title, xlabel="x/L", ylabel="time (s)")
    colorbar = fig.colorbar(images[-1], cax=color_axis)
    colorbar.set_label("Absolute error (K), shared log scale")
    axes[2].semilogy(times[1:], baseline_rmse[1:], label="offline baseline PINN")
    axes[2].semilogy(times[1:], adaptive_rmse[1:], label="adaptive balanced PINN")
    axes[2].axvline(0.5, color="black", linestyle="--", label="flux switch")
    axes[2].set(
        title="Whole-wall RMSE at each time",
        xlabel="time (s)",
        ylabel="RMSE (K)",
    )
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(fontsize=8)
    fig.suptitle("PINNs compared with the exact constant-flux analytical solution")
    fig.savefig(graph_dir / "01_balanced_adaptive_vs_baseline_error.png", dpi=200)
    plt.close(fig)


def _plot_temperature_profiles(
    x_hat, times, truth, baseline, adaptive, pino, graph_dir: Path
) -> None:
    requested_times = (0.25, 0.475, 0.5, 0.525, 0.75, 1.0)
    indices = [int(np.argmin(np.abs(times - value))) for value in requested_times]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), sharex=True, sharey=True)
    for ax, index in zip(axes.flat, indices):
        ax.plot(x_hat, truth[index], "k-", linewidth=2.4, label="analytical")
        ax.plot(x_hat, baseline[index], "--", color="tab:orange", label="offline PINN")
        ax.plot(x_hat, adaptive[index], color="tab:blue", label="adaptive balanced PINN")
        ax.plot(x_hat, pino[index], color="tab:green", label="streamed PINO")
        ax.set_title(f"t = {times[index]:.3f} s")
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("x/L")
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature (K)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("Temperature point-by-point: exact solution and all three learned models")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(graph_dir / "02_all_models_temperature_values.png", dpi=200)
    plt.close(fig)


def _plot_adaptive_pino_errors(
    x_hat, times, truth, adaptive, pino, graph_dir: Path
) -> None:
    adaptive_error = np.abs(adaptive - truth)
    pino_error = np.abs(pino - truth)
    positive = np.concatenate((adaptive_error.ravel(), pino_error.ravel()))
    positive = positive[positive > 0]
    vmin = max(float(np.percentile(positive, 1.0)), 1.0e-5)
    vmax = max(float(np.percentile(positive, 99.5)), vmin * 10.0)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    extent = [x_hat.min(), x_hat.max(), times.min(), times.max()]
    fig = plt.figure(figsize=(16, 4.7), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 0.045, 1.0))
    axes = (fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), fig.add_subplot(grid[3]))
    color_axis = fig.add_subplot(grid[2])
    for ax, values, title in (
        (axes[0], adaptive_error, "Adaptive balanced PINN |error|"),
        (axes[1], pino_error, "Streamed PINO |error|"),
    ):
        image = ax.imshow(
            np.maximum(values, vmin),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            norm=norm,
        )
        ax.axhline(0.5, color="magenta", linestyle="--", linewidth=1.5)
        ax.set(title=title, xlabel="x/L", ylabel="time (s)")
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Absolute error (K), shared log scale")
    adaptive_rmse = np.sqrt(np.mean((adaptive - truth) ** 2, axis=1))
    pino_rmse = np.sqrt(np.mean((pino - truth) ** 2, axis=1))
    axes[2].semilogy(times[1:], adaptive_rmse[1:], label="adaptive balanced PINN")
    axes[2].semilogy(times[1:], pino_rmse[1:], label="streamed PINO")
    axes[2].axvline(0.5, color="black", linestyle="--", label="flux switch")
    axes[2].set(
        title="Whole-wall causal RMSE",
        xlabel="time (s)",
        ylabel="RMSE (K)",
    )
    axes[2].grid(alpha=0.25, which="both")
    axes[2].legend(fontsize=8)
    fig.suptitle("Online models compared with the exact analytical solution")
    fig.savefig(graph_dir / "03_adaptive_pinn_vs_streamed_pino_error.png", dpi=200)
    plt.close(fig)


def _plot_pino_improvement(
    previous_npz: Path, current_npz: Path, graph_dir: Path
) -> None:
    if not previous_npz.is_file():
        return
    old = np.load(previous_npz)
    new = np.load(current_npz)
    old_error = np.sqrt(
        np.mean(
            (old["streamed_pino_temperature_k"] - old["analytical_temperature_k"]) ** 2,
            axis=1,
        )
    )
    new_error = np.sqrt(
        np.mean(
            (new["streamed_pino_temperature_k"] - new["analytical_temperature_k"]) ** 2,
            axis=1,
        )
    )
    adaptive_error = np.sqrt(
        np.mean(
            (new["adaptive_pinn_temperature_k"] - new["analytical_temperature_k"]) ** 2,
            axis=1,
        )
    )
    time = new["time_s"]
    overall = (
        float(np.sqrt(np.mean(old_error**2))),
        float(np.sqrt(np.mean(new_error**2))),
        float(np.sqrt(np.mean(adaptive_error**2))),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    axes[0].semilogy(time[1:], old_error[1:], label="previous PINO")
    axes[0].semilogy(time[1:], new_error[1:], label="improved mixed-sample PINO")
    axes[0].semilogy(time[1:], adaptive_error[1:], label="8,001-step adaptive PINN", alpha=0.75)
    axes[0].axvline(0.5, color="black", linestyle="--", label="flux switch")
    axes[0].set(title="Whole-wall RMSE at each time", xlabel="time (s)", ylabel="RMSE (K)")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)
    bars = axes[1].bar(
        ("previous\nPINO", "improved\nPINO", "adaptive\nPINN"),
        overall,
        color=("tab:gray", "tab:green", "tab:blue"),
    )
    axes[1].bar_label(bars, labels=[f"{value:.3f} K" for value in overall], padding=3)
    axes[1].set(title="Locked-case overall RMSE", ylabel="RMSE (K)")
    axes[1].grid(alpha=0.2, axis="y")
    fig.suptitle("Effect of longer training and stratified mixed boundary samples")
    fig.savefig(graph_dir / "04_pino_before_after_improvement.png", dpi=200)
    plt.close(fig)


def _metric_block(reference, prediction, times, switch_time):
    difference = np.asarray(prediction) - np.asarray(reference)
    time_rmse = np.sqrt(np.mean(difference**2, axis=1))
    pre = times < switch_time
    post = ~pre
    return {
        "overall_rmse_k": float(np.sqrt(np.mean(difference**2))),
        "pre_switch_rmse_k": float(np.sqrt(np.mean(difference[pre] ** 2))),
        "post_switch_rmse_k": float(np.sqrt(np.mean(difference[post] ** 2))),
        "switch_time_rmse_k": float(time_rmse[np.argmin(np.abs(times - switch_time))]),
        "maximum_time_rmse_k": float(time_rmse.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--architecture-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-tables-dir", type=Path)
    args = parser.parse_args()

    cfg = load_balanced_config(args.architecture_file, args.profile)
    run_dir = args.output_dir / "mixed_flux_balanced" / args.profile
    graph_dir = run_dir / "graphs"
    model_dir = run_dir / "models"
    paper_dir = run_dir / "paper_validation"
    for path in (graph_dir, model_dir, paper_dir):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.architecture_file, run_dir / "architecture_used.toml")

    # This remains a distinct dimensionless 2-D check: it verifies the legacy
    # analytical implementation against the paper's published table values.
    paper_cfg = make_config("full")
    paper_cfg.paper_tables_dir = (
        args.paper_tables_dir.resolve()
        if args.paper_tables_dir
        else REPOSITORY_ROOT / "paper_result" / "mdpi_416_tables"
    )
    print("Stage 0/4: validating the original paper series", flush=True)
    paper_metrics = validate_against_literature(paper_cfg, paper_dir)
    shutil.copy2(
        paper_dir / "01_literature_validation.png",
        graph_dir / "00a_paper_analytical_validation.png",
    )
    physical_metrics = analytical_validation(cfg)
    _save_analytical_plot(cfg, graph_dir)

    corpus = FINAL_ROOT / "data" / (
        "mixed_flux_v2" if args.profile == "full" else "mixed_flux_v2_smoke"
    )
    print(f"Stage 1/4: preparing exact CSV corpus at {corpus}", flush=True)
    generate_csv_corpus(corpus, cfg)
    corpus_metrics = _plot_corpus_coverage(corpus, graph_dir)
    locked_row = read_manifest(corpus, "test_locked")[0]
    locked = load_scenario(corpus, locked_row, cfg)
    times, x_hat, truth = locked["times"], locked["x_hat"], locked["field"]
    tt, xx = np.meshgrid(times, x_hat, indexing="ij")
    field_points = np.column_stack((xx.ravel(), tt.ravel()))
    sensor_tt, sensor_xx = np.meshgrid(times, locked["sensor_x_hat"], indexing="ij")
    sensor_points = np.column_stack((sensor_xx.ravel(), sensor_tt.ravel()))
    sensor_values = locked["sensor_values"].ravel()[:, None]

    print("Stage 2/4: training offline and adaptive balanced DeepXDE PINNs", flush=True)
    baseline = train_baseline(cfg, field_points, truth.ravel()[:, None], times)
    torch.save(baseline.network_state, model_dir / "offline_baseline_pinn.pt")
    adaptive = adapt_balanced(
        cfg,
        baseline.network_state,
        sensor_points,
        sensor_values,
        locked["flux"],
        field_points,
        truth.ravel()[:, None],
        times,
    )
    torch.save(adaptive.network_state, model_dir / "adaptive_balanced_pinn.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 3/4: training streamed PINO on {device}", flush=True)
    _, pino_metrics, pino_arrays = train_streamed_pino(corpus, cfg, model_dir, device)
    pino = pino_arrays["prediction"]

    print("Stage 4/4: writing comparable metrics and graphs", flush=True)
    metrics = {
        "problem": {
            "description": "1-D wall, constant hot-side flux, convective coolant face",
            "locked_test": "4.0 to 5.2 MW/m2 at t=0.5 s",
            "analytical_reference": "Robin eigenfunction expansion with switched-step superposition",
            "paper_reference_scope": "separate 2-D dimensionless literature implementation validation",
            "sensor_data_loss_scope": "adaptive retraining only; absent from offline PINN training",
        },
        "device": str(device),
        "configuration": asdict(cfg),
        "paper_validation": paper_metrics,
        "physical_analytical_checks": physical_metrics,
        "training_corpus": corpus_metrics,
        "offline_baseline_pinn": _metric_block(truth, baseline.prediction, times, cfg.switch_time_s),
        "adaptive_balanced_pinn": {
            **_metric_block(truth, adaptive.prediction, times, cfg.switch_time_s),
            "median_update_latency_ms": float(np.median(adaptive.update_latency_ms)),
            "p99_update_latency_ms": float(np.percentile(adaptive.update_latency_ms, 99)),
        },
        "streamed_pino": {
            **_metric_block(truth, pino, times, cfg.switch_time_s),
            **pino_metrics,
        },
        "paths": {
            "csv_corpus": str(corpus.resolve()),
            "architecture": str(args.architecture_file.resolve()),
            "graphs": str(graph_dir.resolve()),
        },
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(_json_ready(metrics), indent=2), encoding="utf-8"
    )
    with (run_dir / "rmse_by_time.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "offline_pinn_rmse_k", "adaptive_pinn_rmse_k", "streamed_pino_rmse_k"))
        for index, tau in enumerate(times):
            writer.writerow(
                (
                    tau,
                    np.sqrt(np.mean((baseline.prediction[index] - truth[index]) ** 2)),
                    np.sqrt(np.mean((adaptive.prediction[index] - truth[index]) ** 2)),
                    np.sqrt(np.mean((pino[index] - truth[index]) ** 2)),
                )
            )
    np.savez_compressed(
        run_dir / "locked_test_predictions.npz",
        x_hat=x_hat,
        time_s=times,
        analytical_temperature_k=truth,
        offline_pinn_temperature_k=baseline.prediction,
        adaptive_pinn_temperature_k=adaptive.prediction,
        streamed_pino_temperature_k=pino,
    )
    _plot_baseline_adaptive(x_hat, times, truth, baseline.prediction, adaptive.prediction, graph_dir)
    _plot_temperature_profiles(x_hat, times, truth, baseline.prediction, adaptive.prediction, pino, graph_dir)
    _plot_adaptive_pino_errors(x_hat, times, truth, adaptive.prediction, pino, graph_dir)
    _plot_pino_improvement(
        args.output_dir
        / "constant_flux_balanced"
        / "full"
        / "locked_test_predictions.npz",
        run_dir / "locked_test_predictions.npz",
        graph_dir,
    )
    print(json.dumps(_json_ready(metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
