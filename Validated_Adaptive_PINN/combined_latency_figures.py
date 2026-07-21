"""Generate paper-style figures for the combined low-latency adaptive PINN."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm, TwoSlopeNorm

from parameters import LATENCY_EXPERIMENTS, config_for_latency_experiment, make_config
from pinn_workflow import adapt_online, predict, rmse, train_baseline
from reference_data import generate_reference_dataset
from switch_solver import generate_switch_dataset


def _slice_matrix(points, values, times, x_values, y_value=0.5):
    matrix = np.empty((len(times), len(x_values)))
    for row, tau in enumerate(times):
        for column, x_value in enumerate(x_values):
            mask = (
                np.isclose(points[:, 0], x_value)
                & np.isclose(points[:, 1], y_value)
                & np.isclose(points[:, 2], tau)
            )
            matrix[row, column] = values[mask][0, 0]
    return matrix


def _caption(fig, text):
    fig.text(
        0.03,
        0.035,
        textwrap.fill(text, width=90),
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
    )


def plot_streaming_sensors(cfg, data, output_dir: Path, reference_slice=None):
    if reference_slice is None:
        reference_slice = _slice_matrix(
            data.field_points, data.field_values, data.times, data.x
        )
    displayed_reference = np.clip(reference_slice, 0.0, None)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.24, top=0.90)
    image = ax.pcolormesh(
        data.x,
        data.times,
        displayed_reference,
        shading="auto",
        cmap="magma",
        norm=PowerNorm(
            gamma=0.30,
            vmin=0.0,
            vmax=float(np.max(displayed_reference)),
        ),
    )
    fig.colorbar(
        image,
        ax=ax,
        label="Numerical reference θ at Y=0.5 (power-scaled colors)",
    )

    center_mask = np.isclose(data.sensor_points[:, 1], 0.5)
    center_points = data.sensor_points[center_mask]
    edges = np.linspace(0.0, cfg.tau_final, 5)
    segment_labels = tuple(
        f"segment {index + 1}: τ={start:g}–{stop:g}"
        for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:]))
    )
    colors = plt.get_cmap("tab10")([0, 1, 2, 3])
    for index, (start, stop, label, color) in enumerate(
        zip(edges[:-1], edges[1:], segment_labels, colors)
    ):
        if index == len(segment_labels) - 1:
            mask = (center_points[:, 2] >= start) & (center_points[:, 2] <= stop)
        else:
            mask = (center_points[:, 2] >= start) & (center_points[:, 2] < stop)
        ax.scatter(
            center_points[mask, 0],
            center_points[mask, 2],
            s=20,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
            zorder=3,
        )
    ax.axhline(
        data.switch_tau,
        color="white",
        linestyle="--",
        linewidth=1.4,
        label=f"boundary change at τ={data.switch_tau:g}",
    )
    ax.set(
        xlabel="Dimensionless wall coordinate X at Y=0.5",
        ylabel="Dimensionless time τ",
        title="Combined low-latency streaming sensors over the numerical reference field",
        xlim=(0.0, 1.0),
        ylim=(0.0, cfg.tau_final),
    )
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
    informative_ticks = np.asarray((0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0))
    informative_ticks = informative_ticks[informative_ticks <= cfg.tau_final]
    ax.set_yticks(informative_ticks)
    ax.set_yticklabels([f"{value:g}" for value in informative_ticks])
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    _caption(
        fig,
        "Fig. 8. Sparse boundary-aware observations used by the combined low-latency "
        "adaptive PINN. Colors group the run into four display segments; they are not "
        "training batches. The model updates after every time instance (n=1) and retains "
        "only the four most recent time instances in its data-loss window. The symlog time "
        "axis and power-scaled colors reveal the early transient and lower-amplitude field.",
    )
    path = output_dir / "08_combined_streaming_sensors.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return reference_slice


def plot_error_fields(cfg, data, baseline_prediction, adaptive, output_dir: Path):
    truth = _slice_matrix(data.field_points, data.field_values, data.times, data.x)
    baseline = _slice_matrix(
        data.field_points, baseline_prediction, data.times, data.x
    )
    posterior = _slice_matrix(
        data.field_points,
        adaptive.causal_posterior_field_prediction,
        data.times,
        data.x,
    )
    offline_error = baseline - truth
    adaptive_error = posterior - truth
    improvement = np.abs(offline_error) - np.abs(adaptive_error)

    error_limit = max(np.max(np.abs(offline_error)), np.max(np.abs(adaptive_error)))
    improvement_limit = np.max(np.abs(improvement))
    error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    improvement_norm = TwoSlopeNorm(
        vmin=-improvement_limit, vcenter=0.0, vmax=improvement_limit
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8))
    fig.subplots_adjust(left=0.055, right=0.94, bottom=0.25, top=0.86, wspace=0.38)
    first = axes[0].pcolormesh(
        data.x,
        data.times,
        offline_error,
        shading="auto",
        cmap="coolwarm",
        norm=error_norm,
    )
    axes[1].pcolormesh(
        data.x,
        data.times,
        adaptive_error,
        shading="auto",
        cmap="coolwarm",
        norm=error_norm,
    )
    third = axes[2].pcolormesh(
        data.x,
        data.times,
        improvement,
        shading="auto",
        cmap="BrBG",
        norm=improvement_norm,
    )
    axes[0].set_title("Offline PINN error")
    axes[1].set_title("Combined adaptive PINN error after update")
    axes[2].set_title("|offline error| − |adaptive error|")
    for index, ax in enumerate(axes):
        ax.axhline(
            data.switch_tau,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="boundary change" if index == 0 else None,
        )
        ax.set(
            xlabel="Dimensionless wall coordinate X at Y=0.5",
            ylabel="Dimensionless time τ",
            xlim=(0.0, 1.0),
            ylim=(0.0, cfg.tau_final),
        )
    axes[0].legend(fontsize=8, loc="upper right")
    fig.colorbar(
        first,
        ax=axes[:2],
        fraction=0.035,
        pad=0.025,
        label="Signed prediction error in θ",
    )
    fig.colorbar(
        third,
        ax=axes[2],
        fraction=0.05,
        pad=0.035,
        label="Positive means adaptation helped",
    )
    _caption(
        fig,
        "Fig. 9. Centerline error fields for the offline and combined low-latency "
        "adaptive PINNs. Each adaptive column uses the model immediately after that "
        "time instance was assimilated. Positive values in the right panel indicate "
        "where online adaptation reduced absolute error relative to the offline PINN.",
    )
    path = output_dir / "09_combined_error_fields.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return truth, baseline, posterior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--replot-existing",
        action="store_true",
        help="regenerate the sensor figure from the saved figure-data NPZ",
    )
    args = parser.parse_args()

    base = make_config(args.profile)
    combined = next(
        experiment
        for experiment in LATENCY_EXPERIMENTS
        if experiment.name == "combined low-latency"
    )
    if args.profile == "smoke":
        # Preserve the combined method while scaling the sampling interval to
        # the shorter smoke domain and limiting optimization work.
        combined = type(combined)(
            name=combined.name,
            sample_spacing_tau=base.tau_final / 10.0,
            batch_size_n=1,
            sensor_x=combined.sensor_x,
            sensor_y=combined.sensor_y,
            observation_window_batches=combined.observation_window_batches,
            data_loss_weight=combined.data_loss_weight,
            adaptive_iterations_per_batch=10,
        )
    cfg = config_for_latency_experiment(base, combined)
    cfg.validate()
    output_dir = args.output_dir or base.output_dir / "combined_latency"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_existing:
        stored = np.load(output_dir / "combined_latency_figure_data.npz")
        saved_data = SimpleNamespace(
            x=stored["x"],
            times=stored["times"],
            switch_tau=float(stored["switch_tau"]),
            sensor_points=stored["sensor_points"],
        )
        plot_streaming_sensors(
            cfg,
            saved_data,
            output_dir,
            reference_slice=stored["reference_slice"],
        )
        print(f"Rescaled figure: {(output_dir / '08_combined_streaming_sensors.png').resolve()}")
        return

    print("[1/3] Training the shared offline PINN...")
    reference = generate_reference_dataset(base, output_dir)
    baseline = train_baseline(
        base,
        reference.field_points,
        reference.field_values,
        reference.times,
        verbose=0,
    )
    print("[2/3] Running the combined low-latency boundary-change experiment...")
    switch_data = generate_switch_dataset(cfg)
    baseline_prediction = predict(baseline.model, switch_data.field_points, base)
    adaptive = adapt_online(
        cfg,
        baseline.network_state,
        switch_data.sensor_points,
        switch_data.sensor_values,
        switch_data.field_points,
        switch_data.field_values,
        switch_data.times,
        verbose=0,
    )
    print("[3/3] Creating the two paper-style figures...")
    reference_slice = plot_streaming_sensors(cfg, switch_data, output_dir)
    truth, baseline_slice, posterior_slice = plot_error_fields(
        cfg, switch_data, baseline_prediction, adaptive, output_dir
    )
    np.savez_compressed(
        output_dir / "combined_latency_figure_data.npz",
        x=switch_data.x,
        times=switch_data.times,
        switch_tau=switch_data.switch_tau,
        reference_slice=reference_slice,
        offline_prediction_slice=baseline_slice,
        adaptive_posterior_prediction_slice=posterior_slice,
        sensor_points=switch_data.sensor_points,
        sensor_values=switch_data.sensor_values,
    )
    metrics = {
        "configuration": {
            "sample_spacing_tau": combined.sample_spacing_tau,
            "batch_size_n": combined.batch_size_n,
            "observation_window_batches": combined.observation_window_batches,
            "sensor_x": combined.sensor_x,
            "sensor_y": combined.sensor_y,
            "data_loss_weight": combined.data_loss_weight,
            "adaptive_iterations_per_batch": combined.adaptive_iterations_per_batch,
        },
        "offline_full_field_rmse": rmse(switch_data.field_values, baseline_prediction),
        "causal_prior_full_field_rmse": rmse(
            switch_data.field_values, adaptive.causal_prior_field_prediction
        ),
        "causal_posterior_full_field_rmse": rmse(
            switch_data.field_values, adaptive.causal_posterior_field_prediction
        ),
    }
    (output_dir / "combined_latency_figure_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
