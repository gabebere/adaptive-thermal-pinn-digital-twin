"""Evaluate retained offline/adaptive/PINO models on two unseen flux cases."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from constant_flux_physics import (
    load_balanced_config,
    switched_flux_temperature,
)
from physical_pinn_workflow import adapt_balanced, make_network, predict as pinn_predict
from physical_streamed_pino import (
    ConstantFluxStreamedPINO,
    predict as pino_predict,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load_checkpoints(cfg, model_dir: Path, device: torch.device):
    offline_state = torch.load(
        model_dir / "offline_baseline_pinn.pt", map_location="cpu", weights_only=True
    )
    offline = make_network(cfg)
    offline.load_state_dict(copy.deepcopy(offline_state))
    offline.to(device)

    checkpoint = torch.load(
        model_dir / "streamed_pino.pt", map_location=device, weights_only=True
    )
    pino = ConstantFluxStreamedPINO(
        sensors=len(cfg.sensor_x_hat),
        hidden=cfg.operator_hidden,
        basis=cfg.operator_basis,
    ).to(device)
    pino.load_state_dict(checkpoint["model"])
    pino.eval()
    return offline, offline_state, pino


def _case_arrays(cfg, q_before: float, q_after: float, switch_time: float | None):
    # Match the float32 tensors used to train the retained GRU/PINO checkpoint.
    x_hat = np.linspace(0.0, 1.0, cfg.field_points, dtype=np.float32)
    x_m = x_hat * cfg.wall_thickness_m
    times = np.linspace(
        0.0, cfg.final_time_s, cfg.time_instances, dtype=np.float32
    )
    effective_switch = (
        cfg.final_time_s + 1.0 if switch_time is None else float(switch_time)
    )
    effective_after = q_before if switch_time is None else q_after
    truth = switched_flux_temperature(
        x_m,
        times,
        q_before,
        effective_after,
        effective_switch,
        cfg,
    ).astype(np.float32)
    flux = np.where(times < effective_switch, q_before, effective_after).astype(
        np.float32
    )
    sensor_indices = np.asarray(
        [np.argmin(np.abs(x_hat - value)) for value in cfg.sensor_x_hat]
    )
    sensor_x_hat = x_hat[sensor_indices]
    sensor_values = truth[:, sensor_indices]
    tt, xx = np.meshgrid(times, x_hat, indexing="ij")
    field_points = np.column_stack((xx.ravel(), tt.ravel()))
    sensor_tt, sensor_xx = np.meshgrid(times, sensor_x_hat, indexing="ij")
    sensor_points = np.column_stack((sensor_xx.ravel(), sensor_tt.ravel()))
    return {
        "x_hat": x_hat,
        "times": times,
        "truth": truth,
        "flux": flux,
        "sensor_x_hat": sensor_x_hat,
        "sensor_values": sensor_values,
        "field_points": field_points,
        "sensor_points": sensor_points,
    }


def _rmse_by_time(reference, prediction):
    return np.sqrt(np.mean((np.asarray(prediction) - np.asarray(reference)) ** 2, axis=1))


def _metric_block(reference, prediction, times, switch_time):
    difference = np.asarray(prediction) - np.asarray(reference)
    metrics = {
        "overall_rmse_k": float(np.sqrt(np.mean(difference**2))),
        "maximum_time_rmse_k": float(_rmse_by_time(reference, prediction).max()),
    }
    if switch_time is not None:
        pre = times < switch_time
        post = ~pre
        metrics.update(
            {
                "pre_switch_rmse_k": float(np.sqrt(np.mean(difference[pre] ** 2))),
                "post_switch_rmse_k": float(np.sqrt(np.mean(difference[post] ** 2))),
            }
        )
    return metrics


def _profile_indices(times, switch_time):
    tau = times * 100.0
    if switch_time is None:
        requested = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
    else:
        switch_tau = switch_time * 100.0
        requested = (
            0.0,
            25.0,
            max(0.0, switch_tau - 2.5),
            min(100.0, switch_tau + 2.5),
            75.0,
            100.0,
        )
    return [int(np.argmin(np.abs(tau - value))) for value in requested]


def _temperature_plot(
    path: Path,
    title: str,
    arrays: dict,
    predictions: dict[str, np.ndarray],
    switch_time: float | None,
):
    indices = _profile_indices(arrays["times"], switch_time)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), sharex=True, sharey=True)
    styles = {
        "analytical": {"color": "black", "linewidth": 2.4},
        "offline PINN": {"color": "tab:orange", "linestyle": "--"},
        "adaptive PINN": {"color": "tab:blue"},
        "streamed PINO": {"color": "tab:green"},
    }
    for ax, index in zip(axes.flat, indices):
        for label, field in predictions.items():
            ax.plot(arrays["x_hat"], field[index], label=label, **styles[label])
        ax.set_title(f"tau = {arrays['times'][index] * 100.0:.1f}")
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("x/L")
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature (K)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _rmse_plot(
    path: Path,
    title: str,
    arrays: dict,
    predictions: dict[str, np.ndarray],
    switch_time: float | None,
):
    fig, ax = plt.subplots(figsize=(9.2, 5.4), layout="constrained")
    colors = {
        "offline PINN": "tab:orange",
        "adaptive PINN": "tab:blue",
        "streamed PINO": "tab:green",
    }
    tau = arrays["times"] * 100.0
    for label, field in predictions.items():
        error = _rmse_by_time(arrays["truth"], field)
        ax.semilogy(tau[1:], error[1:], label=label, color=colors[label], linewidth=2)
    if switch_time is not None:
        ax.axvline(
            switch_time * 100.0,
            color="black",
            linestyle="--",
            label=f"boundary break at tau={switch_time * 100.0:.1f}",
        )
    ax.set(title=title, xlabel="tau", ylabel="Whole-wall RMSE vs analytical (K)")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_case(
    name: str,
    specification: dict,
    cfg,
    offline,
    offline_state,
    pino,
    device,
    output_root: Path,
):
    case_dir = output_root / name
    case_dir.mkdir(parents=True, exist_ok=True)
    arrays = _case_arrays(
        cfg,
        specification["q_before_w_m2"],
        specification["q_after_w_m2"],
        specification["switch_time_s"],
    )
    offline_prediction = pinn_predict(offline, arrays["field_points"], cfg).reshape(
        cfg.time_instances, cfg.field_points
    )
    adaptive = adapt_balanced(
        cfg,
        copy.deepcopy(offline_state),
        arrays["sensor_points"],
        arrays["sensor_values"].ravel()[:, None],
        arrays["flux"],
        arrays["field_points"],
        arrays["truth"].ravel()[:, None],
        arrays["times"],
        verbose=0,
    )
    pino_arrays = {
        "sensor_values": arrays["sensor_values"][None],
        "flux": arrays["flux"][None],
        "times": arrays["times"],
        "x_hat": arrays["x_hat"],
    }
    pino_prediction = pino_predict(pino, pino_arrays, cfg, device)[0]
    switch_time = specification["switch_time_s"]
    condition = (
        f"q={specification['q_before_w_m2']/1e6:.3f} MW/m², no break"
        if switch_time is None
        else (
            f"q={specification['q_before_w_m2']/1e6:.3f}→"
            f"{specification['q_after_w_m2']/1e6:.3f} MW/m² at "
            f"tau={switch_time*100.0:.1f}"
        )
    )
    analytical = arrays["truth"]
    _temperature_plot(
        case_dir / "01_adaptive_offline_analytical_temperature.png",
        f"Adaptive vs offline vs analytical: {condition}",
        arrays,
        {
            "analytical": analytical,
            "offline PINN": offline_prediction,
            "adaptive PINN": adaptive.prediction,
        },
        switch_time,
    )
    _rmse_plot(
        case_dir / "02_adaptive_offline_rmse.png",
        f"Adaptive and offline errors: {condition}",
        arrays,
        {"offline PINN": offline_prediction, "adaptive PINN": adaptive.prediction},
        switch_time,
    )
    _temperature_plot(
        case_dir / "03_all_models_temperature.png",
        f"All models vs analytical: {condition}",
        arrays,
        {
            "analytical": analytical,
            "offline PINN": offline_prediction,
            "adaptive PINN": adaptive.prediction,
            "streamed PINO": pino_prediction,
        },
        switch_time,
    )
    _rmse_plot(
        case_dir / "04_all_models_rmse.png",
        f"All model errors: {condition}",
        arrays,
        {
            "offline PINN": offline_prediction,
            "adaptive PINN": adaptive.prediction,
            "streamed PINO": pino_prediction,
        },
        switch_time,
    )
    np.savez_compressed(
        case_dir / "predictions.npz",
        tau=arrays["times"] * 100.0,
        x_hat=arrays["x_hat"],
        analytical_temperature_k=analytical,
        offline_pinn_temperature_k=offline_prediction,
        adaptive_pinn_temperature_k=adaptive.prediction,
        streamed_pino_temperature_k=pino_prediction,
        flux_w_m2=arrays["flux"],
    )
    return {
        "specification": specification,
        "offline_pinn": _metric_block(
            analytical, offline_prediction, arrays["times"], switch_time
        ),
        "adaptive_pinn": {
            **_metric_block(analytical, adaptive.prediction, arrays["times"], switch_time),
            "median_update_latency_ms": float(np.median(adaptive.update_latency_ms)),
        },
        "streamed_pino": _metric_block(
            analytical, pino_prediction, arrays["times"], switch_time
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "new_boundary_cases"
    )
    args = parser.parse_args()
    architecture = ROOT / "architectures" / "balanced.toml"
    cfg = load_balanced_config(architecture, "full")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Curated, Git-tracked checkpoints make evaluation independent of ignored
    # output folders and never trigger base-model retraining.
    model_dir = ROOT / "checkpoints"
    offline, offline_state, pino = _load_checkpoints(cfg, model_dir, device)

    rng = np.random.default_rng(args.seed)
    no_break_flux = float(rng.uniform(cfg.q_min_w_m2, cfg.q_max_w_m2))
    switch_tau = int(rng.integers(30, 61))
    q_before, q_after = rng.uniform(cfg.q_min_w_m2, cfg.q_max_w_m2, 2)
    while abs(q_after - q_before) < 0.6e6:
        q_after = rng.uniform(cfg.q_min_w_m2, cfg.q_max_w_m2)
    cases = {
        "no_break": {
            "q_before_w_m2": no_break_flux,
            "q_after_w_m2": no_break_flux,
            "switch_time_s": None,
        },
        "random_break": {
            "q_before_w_m2": float(q_before),
            "q_after_w_m2": float(q_after),
            "switch_time_s": switch_tau / 100.0,
            "switch_tau": switch_tau,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "seed": args.seed,
        "device": str(device),
        "checkpoint_directory": str(model_dir.resolve()),
        "tau_mapping": "tau=100*t/final_time; physical model interval t=[0,1] s",
        "cases": {},
    }
    for name, specification in cases.items():
        print(f"Evaluating {name}: {specification}", flush=True)
        results["cases"][name] = evaluate_case(
            name,
            specification,
            cfg,
            offline,
            offline_state,
            pino,
            device,
            args.output_dir,
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
