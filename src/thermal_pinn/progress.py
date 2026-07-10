from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, make_config
from .experiment import run_experiment


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except Exception:
        return default
    return result.stdout.strip() or default


def _mean_std(rows: list[dict[str, object]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def run_progress_study(
    output_dir: str | Path = "outputs/progress",
    seeds: tuple[int, ...] = (7, 11, 19),
    device: str = "cpu",
) -> dict[str, object]:
    """Run the small multi-seed study used by the preliminary report."""
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    last_metrics: dict[str, object] | None = None

    for seed in seeds:
        cfg = make_config("progress", output_dir / f"seed_{seed}")
        cfg.seed = seed
        cfg.device = device
        metrics = run_experiment(cfg)
        last_metrics = metrics
        baseline = metrics["baseline"]
        adaptive = metrics["adaptive_main_noise"]
        row = {
            "seed": seed,
            "baseline_relative_l2": baseline["relative_l2_global"],
            "adaptive_relative_l2": adaptive["relative_l2_global"],
            "baseline_hot_side_max_error_K": baseline["hot_side_max_abs_error"],
            "adaptive_hot_side_max_error_K": adaptive["hot_side_max_abs_error"],
            "baseline_max_error_K": baseline["max_abs_error"],
            "adaptive_max_error_K": adaptive["max_abs_error"],
            "baseline_runtime_s": baseline["training_runtime_s"],
            "total_runtime_s": metrics["total_runtime_s"],
            "relative_l2_improvement_percent": 100.0
            * (baseline["relative_l2_global"] - adaptive["relative_l2_global"])
            / baseline["relative_l2_global"],
        }
        rows.append(row)

    summary_rows = [
        {
            "metric": "baseline_relative_l2",
            "mean": _mean_std(rows, "baseline_relative_l2")[0],
            "std": _mean_std(rows, "baseline_relative_l2")[1],
        },
        {
            "metric": "adaptive_relative_l2",
            "mean": _mean_std(rows, "adaptive_relative_l2")[0],
            "std": _mean_std(rows, "adaptive_relative_l2")[1],
        },
        {
            "metric": "baseline_hot_side_max_error_K",
            "mean": _mean_std(rows, "baseline_hot_side_max_error_K")[0],
            "std": _mean_std(rows, "baseline_hot_side_max_error_K")[1],
        },
        {
            "metric": "adaptive_hot_side_max_error_K",
            "mean": _mean_std(rows, "adaptive_hot_side_max_error_K")[0],
            "std": _mean_std(rows, "adaptive_hot_side_max_error_K")[1],
        },
        {
            "metric": "relative_l2_improvement_percent",
            "mean": _mean_std(rows, "relative_l2_improvement_percent")[0],
            "std": _mean_std(rows, "relative_l2_improvement_percent")[1],
        },
    ]

    _write_csv(output_dir / "progress_metrics_by_seed.csv", rows)
    _write_csv(output_dir / "progress_metric_summary.csv", summary_rows)
    summary = {
        "seeds": list(seeds),
        "rows": rows,
        "summary": summary_rows,
        "representative_figures": (last_metrics or {}).get("figures", []),
    }
    (output_dir / "progress_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_progress_report(output_dir, summary)
    return summary


def write_progress_report(output_dir: Path, summary: dict[str, object]) -> Path:
    rows = summary["rows"]
    summary_rows = {row["metric"]: row for row in summary["summary"]}
    repo_url = _git_value(["git", "config", "--get", "remote.origin.url"])

    def fmt(metric: str, digits: int = 4) -> str:
        row = summary_rows[metric]
        return f"{row['mean']:.{digits}f} +/- {row['std']:.{digits}f}"

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "preliminary_progress_report.md"
    representative = Path(output_dir) / f"seed_{rows[-1]['seed']}" / "figures"
    visual = representative / "visual_explainer"

    content = f"""# Preliminary Progress Report Draft

## 1. Problem and Physical System

This project studies real-time thermal monitoring of a regeneratively cooled rocket-engine throat wall using a reduced 1D transient heat-conduction model. The goal is not to model combustion or full nozzle flow, but to preserve the engineering question: can sparse wall-temperature measurements improve a physics-informed thermal prediction near the hot-gas side?

The wall temperature obeys

```text
rho cp dT/dt = k d2T/dx2,     0 <= x <= L_wall, 0 <= t <= t_final
```

with hot-gas heat-flux and coolant-side convection boundary conditions:

```text
x = 0:      -k dT/dx = q_hot(t)
x = Lwall: -k dT/dx = h_cool (T - T_cool)
T(x,0) = T0
```

The reference solution is generated by a Crank-Nicolson finite-difference solver. The PINN is trained on nondimensional inputs `(x_hat, t_hat)` and predicts nondimensional temperature rise `theta = (T - T_cool) / delta_T`.

## 2. Baseline and Controlled Modification

The baseline is a standard offline PINN trained once with PDE residual, boundary-condition, and initial-condition losses. This follows the usual PINN formulation introduced by Raissi, Perdikaris, and Karniadakis and the heat-transfer PINN framing reviewed by Cai et al.

The controlled modification is an adaptive PINN. It starts from the offline PINN weights and periodically fine-tunes with streaming sparse thermocouple-like measurements:

```text
loss = PDE + boundary + initial condition + sensor data
```

Thermocouples are treated as sparse observation constraints, not moving physical boundaries. The optional DeepXDE implementation in the repository reproduces the same PDE setup using library-provided `TimePDE`, `OperatorBC`, and `PointSetBC` objects.

## 3. Software Pipeline and Preliminary Results

Repository: {repo_url}  
Main command:

```bash
PYTHONPATH=src python scripts/run_progress_study.py --output-dir outputs/progress
```

Multi-seed preliminary metrics:

| Metric | Mean +/- std |
|---|---:|
| Baseline relative L2 | {fmt('baseline_relative_l2')} |
| Adaptive relative L2 | {fmt('adaptive_relative_l2')} |
| Baseline hot-side max error (K) | {fmt('baseline_hot_side_max_error_K', 2)} |
| Adaptive hot-side max error (K) | {fmt('adaptive_hot_side_max_error_K', 2)} |
| Relative L2 improvement (%) | {fmt('relative_l2_improvement_percent', 2)} |

Representative figures from the latest seed:

- Physical setup: `{visual / '01_physical_setup.png'}`
- Boundary forcing and wall response: `{visual / '02_boundary_forcing_and_surfaces.png'}`
- Reference field: `{representative / 'reference_temperature_field.png'}`
- Baseline profiles: `{representative / 'baseline_profiles.png'}`
- Adaptive profiles: `{representative / 'adaptive_profiles.png'}`
- Error over time: `{representative / 'relative_l2_error_over_time.png'}`
- Error heatmaps: `{visual / '05_error_heatmaps.png'}`

## 4. Remaining Timeline and Fallback

Next steps:

1. Use the progress-study results to select the final sensor-loss weight and adaptive learning rate.
2. Run the full experiment once for publication-quality plots.
3. Add a short discussion of when adaptation helps and where sparse/noisy sensors can locally hurt prediction quality.
4. Convert this draft into the required two-page AIAA-style progress report.

Fallback: if the adaptive model is not consistently better across seeds, the final comparison will be framed as a robustness/limitations study of sensor-assimilating PINNs rather than as a guaranteed improvement.

Author contribution placeholder: both partners should be listed under implementation, physical formulation, figure interpretation, and manuscript writing.
"""
    path.write_text(content, encoding="utf-8")
    return path
