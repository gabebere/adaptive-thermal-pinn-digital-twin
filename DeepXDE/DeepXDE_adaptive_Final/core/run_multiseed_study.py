"""Train and compare the constant-flux PINN/PINO workflow across random seeds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from constant_flux_physics import load_balanced_config, load_scenario, read_manifest
from physical_pinn_workflow import adapt_balanced, train_baseline
from physical_streamed_pino import train_streamed_pino


HERE = Path(__file__).resolve().parent
FINAL_ROOT = HERE.parent
DEFAULT_SEEDS = (7, 11, 19, 23, 31)
MODEL_LABELS = {
    "offline_pinn": "Offline PINN",
    "adaptive_pinn": "Sensor-adaptive PINN",
    "streamed_pino": "Streamed PINO",
}
COLORS = {
    "offline_pinn": "#D55E00",
    "adaptive_pinn": "#0072B2",
    "streamed_pino": "#009E73",
}


def _rmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(reference)) ** 2)))


def _metric_block(reference, prediction, times, switch_time):
    difference = np.asarray(prediction) - np.asarray(reference)
    before = times < switch_time
    after = ~before
    by_time = np.sqrt(np.mean(difference**2, axis=1))
    return {
        "overall_rmse_k": _rmse(reference, prediction),
        "pre_switch_rmse_k": float(np.sqrt(np.mean(difference[before] ** 2))),
        "post_switch_rmse_k": float(np.sqrt(np.mean(difference[after] ** 2))),
        "maximum_time_rmse_k": float(by_time.max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locked_problem(corpus: Path, cfg):
    row = read_manifest(corpus, "test_locked")[0]
    locked = load_scenario(corpus, row, cfg)
    times, x_hat, truth = locked["times"], locked["x_hat"], locked["field"]
    tt, xx = np.meshgrid(times, x_hat, indexing="ij")
    field_points = np.column_stack((xx.ravel(), tt.ravel()))
    sensor_tt, sensor_xx = np.meshgrid(times, locked["sensor_x_hat"], indexing="ij")
    sensor_points = np.column_stack((sensor_xx.ravel(), sensor_tt.ravel()))
    return locked, times, x_hat, truth, field_points, sensor_points


def run_seed(seed: int, base_cfg, corpus: Path, output: Path, device: torch.device):
    seed_dir = output / f"seed_{seed}"
    model_dir = seed_dir / "models"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(base_cfg, seed=seed)
    locked, times, x_hat, truth, field_points, sensor_points = _locked_problem(corpus, cfg)

    print(f"seed {seed}: training offline PINN", flush=True)
    baseline = train_baseline(cfg, field_points, truth.ravel()[:, None], times, verbose=0)
    torch.save(baseline.network_state, model_dir / "offline_baseline_pinn.pt")

    print(f"seed {seed}: adapting PINN from streamed sensors", flush=True)
    adaptive = adapt_balanced(
        cfg,
        baseline.network_state,
        sensor_points,
        locked["sensor_values"].ravel()[:, None],
        locked["flux"],
        field_points,
        truth.ravel()[:, None],
        times,
        verbose=0,
    )
    torch.save(adaptive.network_state, model_dir / "adaptive_balanced_pinn.pt")

    print(f"seed {seed}: training streamed PINO on {device}", flush=True)
    _, pino_training, pino_arrays = train_streamed_pino(corpus, cfg, model_dir, device)
    pino = pino_arrays["prediction"]

    metrics = {
        "seed": seed,
        "device": str(device),
        "configuration": asdict(cfg),
        "offline_pinn": _metric_block(truth, baseline.prediction, times, cfg.switch_time_s),
        "adaptive_pinn": {
            **_metric_block(truth, adaptive.prediction, times, cfg.switch_time_s),
            "median_update_latency_ms": float(np.median(adaptive.update_latency_ms)),
        },
        "streamed_pino": {
            **_metric_block(truth, pino, times, cfg.switch_time_s),
            **pino_training,
        },
    }
    (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        seed_dir / "locked_predictions.npz",
        seed=seed,
        time_s=times,
        x_hat=x_hat,
        analytical_temperature_k=truth,
        offline_pinn_temperature_k=baseline.prediction,
        adaptive_pinn_temperature_k=adaptive.prediction,
        streamed_pino_temperature_k=pino,
    )
    return metrics


def _load_completed(output: Path, requested: list[int]):
    completed = []
    for seed in requested:
        path = output / f"seed_{seed}" / "metrics.json"
        if path.is_file():
            completed.append(json.loads(path.read_text(encoding="utf-8")))
    return completed


def _summary_statistics(values: np.ndarray):
    mean = float(np.mean(values))
    return {
        "mean": mean,
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "coefficient_of_variation": float(np.std(values, ddof=1) / mean)
        if len(values) > 1 and mean != 0.0
        else 0.0,
    }


def summarize(metrics: list[dict], output: Path, corpus: Path, architecture: Path):
    metrics = sorted(metrics, key=lambda item: item["seed"])
    seeds = [item["seed"] for item in metrics]
    fields = ("overall_rmse_k", "pre_switch_rmse_k", "post_switch_rmse_k", "maximum_time_rmse_k")
    rows = []
    summary = {}
    for model in MODEL_LABELS:
        summary[model] = {}
        for field in fields:
            values = np.asarray([item[model][field] for item in metrics], dtype=float)
            summary[model][field] = _summary_statistics(values)
        for item in metrics:
            rows.append({
                "seed": item["seed"],
                "model": model,
                **{field: item[model][field] for field in fields},
            })
    with (output / "seedwise_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "model", *fields))
        writer.writeheader()
        writer.writerows(rows)
    document = {
        "study": "five-seed training-stability benchmark",
        "seeds": seeds,
        "controlled_data_seed": 8128,
        "corpus_manifest_sha256": _sha256(corpus / "manifest.csv"),
        "architecture": str(architecture.resolve()),
        "statistics": summary,
    }
    pino_fields = (
        "locked_rmse_k",
        "interpolation_mean_rmse_k",
        "validation_rmse_k",
        "median_latency_ms",
        "best_epoch",
    )
    document["pino_generalization_statistics"] = {
        field: _summary_statistics(
            np.asarray([item["streamed_pino"][field] for item in metrics], dtype=float)
        )
        for field in pino_fields
    }
    document["adaptive_latency_statistics_ms"] = _summary_statistics(
        np.asarray(
            [item["adaptive_pinn"]["median_update_latency_ms"] for item in metrics],
            dtype=float,
        )
    )
    means = {
        model: summary[model]["overall_rmse_k"]["mean"] for model in MODEL_LABELS
    }
    document["mean_rmse_improvement_factors"] = {
        "adaptive_over_offline": means["offline_pinn"] / means["adaptive_pinn"],
        "pino_over_adaptive": means["adaptive_pinn"] / means["streamed_pino"],
        "pino_over_offline": means["offline_pinn"] / means["streamed_pino"],
    }
    (output / "summary.json").write_text(json.dumps(document, indent=2), encoding="utf-8")

    with (output / "pino_seedwise_generalization.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", *pino_fields))
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {"seed": item["seed"], **{field: item["streamed_pino"][field] for field in pino_fields}}
            )

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(8.6, 5.4), layout="constrained")
    positions = np.arange(len(seeds), dtype=float)
    width = 0.23
    for offset, model in zip((-width, 0.0, width), MODEL_LABELS):
        values = [item[model]["overall_rmse_k"] for item in metrics]
        ax.bar(positions + offset, values, width=width, label=MODEL_LABELS[model],
               color=COLORS[model])
    ax.set_yscale("log")
    ax.set_xticks(positions, [str(seed) for seed in seeds])
    ax.set(title="Training-seed sensitivity on the locked heat-flux-step case",
           xlabel="Random seed", ylabel="Overall RMSE relative to analytical solution (K)")
    ax.grid(alpha=0.25, which="both", axis="y")
    ax.legend()
    fig.savefig(output / "01_seedwise_locked_case_rmse.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.4), layout="constrained")
    data = [np.asarray([item[model]["overall_rmse_k"] for item in metrics]) for model in MODEL_LABELS]
    box = ax.boxplot(data, tick_labels=[MODEL_LABELS[model] for model in MODEL_LABELS],
                     patch_artist=True, showmeans=True)
    for patch, model in zip(box["boxes"], MODEL_LABELS):
        patch.set_facecolor(COLORS[model]); patch.set_alpha(0.65)
    for index, values in enumerate(data, start=1):
        ax.scatter(np.full_like(values, index, dtype=float), values, color="black", s=22, zorder=3)
    ax.set_yscale("log")
    ax.set(title="Distribution of locked-case error across training seeds",
           ylabel="Overall RMSE relative to analytical solution (K)")
    ax.grid(alpha=0.25, which="both", axis="y")
    fig.savefig(output / "02_locked_case_rmse_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.4), layout="constrained")
    for field, label, marker in (
        ("locked_rmse_k", "Locked step case", "o"),
        ("interpolation_mean_rmse_k", "200 held-out interpolation cases", "s"),
        ("validation_rmse_k", "Validation set", "^"),
    ):
        ax.plot(seeds, [item["streamed_pino"][field] for item in metrics],
                marker=marker, linewidth=1.8, label=label)
    ax.set(title="Streamed PINO generalization stability across training seeds",
           xlabel="Random seed", ylabel="RMSE relative to analytical solution (K)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "03_pino_generalization_by_seed.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-file", type=Path,
                        default=FINAL_ROOT / "architectures" / "constant_flux_balanced.toml")
    parser.add_argument("--corpus", type=Path, default=FINAL_ROOT / "data" / "mixed_flux_v2")
    parser.add_argument("--output-dir", type=Path,
                        default=FINAL_ROOT / "outputs" / "multiseed_study")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_balanced_config(args.architecture_file, "full")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.summarize_only:
        for seed in args.seeds:
            completed = args.output_dir / f"seed_{seed}" / "metrics.json"
            if args.skip_existing and completed.is_file():
                print(f"seed {seed}: already complete", flush=True)
                continue
            run_seed(seed, cfg, args.corpus, args.output_dir, device)
    results = _load_completed(args.output_dir, args.seeds)
    if len(results) != len(args.seeds):
        missing = sorted(set(args.seeds) - {item["seed"] for item in results})
        raise RuntimeError(f"missing completed seed results: {missing}")
    summarize(results, args.output_dir, args.corpus, args.architecture_file)
    print(json.dumps({"completed_seeds": args.seeds, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
