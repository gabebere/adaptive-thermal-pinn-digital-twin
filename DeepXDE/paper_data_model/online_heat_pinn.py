"""Adaptive DeepXDE PINN using the MDPI Axioms 12 (2023), 416 data.

The paper problem is 2-D. Inputs are nondimensional (X, Y, tau), and the output
is nondimensional temperature theta. No reference or experimental data is
generated here; all reference values are loaded from Literature_results CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REFERENCE_DIR = (
    Path(__file__).resolve().parents[2]
    / "Literature_results"
    / "mdpi_416_tables"
)

# Paper cases: eta_i(tau) = exp(-d_i tau).
PAPER_CASES = {
    "table_1": (1.0, 1.0, 1.0, 1.0),
    "table_2": (1.0, 1.0, 2.0, 2.0),
    "table_3": (1.0, 2.0, 3.0, 4.0),
}


@dataclass
class Config:
    case: str = "table_1"
    hidden_layers: tuple[int, ...] = (40, 40)
    baseline_iterations: int = 3000
    update_iterations: int = 500
    baseline_learning_rate: float = 1.0e-3
    adaptive_learning_rate: float = 5.0e-4
    data_weight: float = 10.0
    points_per_update: int = 5
    num_domain: int = 2500
    num_boundary: int = 500
    num_initial: int = 300
    seed: int = 7

    @property
    def decays(self) -> tuple[float, float, float, float]:
        return PAPER_CASES[self.case]


def initial_condition(X: np.ndarray) -> np.ndarray:
    """Paper Eq. (93): theta(X,Y,0)=(X-X^2)+(Y-Y^2)."""
    x = X[:, 0:1]
    y = X[:, 1:2]
    return (x - x**2) + (y - y**2)


def make_problem(
    cfg: Config,
    observation_points: np.ndarray | None = None,
    observation_values: np.ndarray | None = None,
) -> dde.data.TimePDE:
    """Build the paper's square-domain transient heat problem."""

    def heat_equation(X, theta):
        theta_t = dde.grad.jacobian(theta, X, i=0, j=2)
        theta_xx = dde.grad.hessian(theta, X, component=0, i=0, j=0)
        theta_yy = dde.grad.hessian(theta, X, component=0, i=1, j=1)
        return theta_t - theta_xx - theta_yy

    def left(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[0], 0.0)

    def right(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[0], 1.0)

    def bottom(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[1], 0.0)

    def top(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[1], 1.0)

    d1, d2, d3, d4 = cfg.decays

    def vertical_boundary(decay: float):
        return lambda X: (X[:, 1:2] - X[:, 1:2] ** 2) * np.exp(
            -decay * X[:, 2:3]
        )

    def horizontal_boundary(decay: float):
        return lambda X: (X[:, 0:1] - X[:, 0:1] ** 2) * np.exp(
            -decay * X[:, 2:3]
        )

    geometry = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
    time_domain = dde.geometry.TimeDomain(0.0, 1.2)
    geometry_time = dde.geometry.GeometryXTime(geometry, time_domain)
    constraints = [
        dde.icbc.DirichletBC(geometry_time, vertical_boundary(d1), left),
        dde.icbc.DirichletBC(geometry_time, vertical_boundary(d2), right),
        dde.icbc.DirichletBC(geometry_time, horizontal_boundary(d3), bottom),
        dde.icbc.DirichletBC(geometry_time, horizontal_boundary(d4), top),
        dde.icbc.IC(geometry_time, initial_condition, lambda _, initial: initial),
    ]
    if observation_points is not None:
        constraints.append(
            dde.icbc.PointSetBC(observation_points, observation_values, component=0)
        )

    return dde.data.TimePDE(
        geometry_time,
        heat_equation,
        constraints,
        num_domain=cfg.num_domain,
        num_boundary=cfg.num_boundary,
        num_initial=cfg.num_initial,
        train_distribution="Hammersley",
    )


def make_network(cfg: Config):
    # Three inputs (X,Y,tau), two 40-neuron tanh layers, one theta output.
    return dde.nn.FNN([3, *cfg.hidden_layers, 1], "tanh", "Glorot normal")


def read_numeric_columns(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Reference file not found: {path}")
    values = {column: [] for column in columns}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        for row in reader:
            for column in columns:
                values[column].append(float(row[column]))
    return {name: np.asarray(data, dtype=float) for name, data in values.items()}


def center_points(tau: np.ndarray) -> np.ndarray:
    """Return paper midpoint coordinates (X,Y)=(0.5,0.5)."""
    return np.column_stack((np.full(len(tau), 0.5), np.full(len(tau), 0.5), tau))


def load_paper_data(reference_dir: Path, case: str):
    """Load 40 assimilation values and eight printed literature values."""
    dense = read_numeric_columns(
        reference_dir / f"{case}_40_times.csv", ("tau", "calculated_terms_20")
    )
    study = read_numeric_columns(
        reference_dir / f"{case}_study_times.csv",
        ("tau", "published_terms_20", "calculated_terms_20"),
    )
    assimilation_points = center_points(dense["tau"])
    assimilation_values = dense["calculated_terms_20"][:, None]
    literature_points = center_points(study["tau"])
    literature_values = study["published_terms_20"][:, None]
    return assimilation_points, assimilation_values, literature_points, literature_values


def rmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - reference) ** 2)))


def train_adaptively(cfg: Config, assimilation_points, assimilation_values):
    """Baseline physics training followed by five-time-instance updates."""
    dde.config.set_random_seed(cfg.seed)
    network = make_network(cfg)
    model = dde.Model(make_problem(cfg), network)
    model.compile("adam", lr=cfg.baseline_learning_rate, loss_weights=[1.0] * 6)
    model.train(
        iterations=cfg.baseline_iterations,
        display_every=max(1, cfg.baseline_iterations // 5),
    )
    baseline_prediction = model.predict(assimilation_points)

    history = []
    batches = np.array_split(
        np.arange(len(assimilation_points)),
        int(np.ceil(len(assimilation_points) / cfg.points_per_update)),
    )
    for update, batch in enumerate(batches, start=1):
        new_points = assimilation_points[batch]
        new_values = assimilation_values[batch]
        before_rmse = rmse(new_values, model.predict(new_points))

        end = int(batch[-1]) + 1
        data = make_problem(cfg, assimilation_points[:end], assimilation_values[:end])
        model = dde.Model(data, network)  # warm start: retain network weights
        model.compile(
            "adam",
            lr=cfg.adaptive_learning_rate,
            loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, cfg.data_weight],
        )
        model.train(
            iterations=cfg.update_iterations,
            display_every=max(1, cfg.update_iterations // 5),
        )
        after_rmse = rmse(new_values, model.predict(new_points))
        history.append(
            {
                "update": update,
                "tau_end": float(new_points[-1, 2]),
                "before_rmse": before_rmse,
                "after_rmse": after_rmse,
            }
        )
        print(
            f"update {update:02d}, tau={new_points[-1, 2]:.4f}: "
            f"RMSE {before_rmse:.6g} -> {after_rmse:.6g}"
        )
    return model, history, baseline_prediction


def save_results(
    model,
    cfg: Config,
    history,
    assimilation_points,
    assimilation_values,
    baseline_prediction,
    literature_points,
    literature_values,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    assimilation_prediction = model.predict(assimilation_points)
    literature_prediction = model.predict(literature_points)
    literature_rmse = rmse(literature_values, literature_prediction)

    np.savetxt(
        output_dir / "baseline_vs_adaptive_center.csv",
        np.column_stack(
            (
                assimilation_points[:, 2],
                assimilation_values[:, 0],
                baseline_prediction[:, 0],
                assimilation_prediction[:, 0],
            )
        ),
        delimiter=",",
        header="tau,reference_theta,baseline_pinn_theta,adaptive_pinn_theta",
        comments="",
    )
    np.savetxt(
        output_dir / "literature_vs_pinn.csv",
        np.column_stack(
            (literature_points[:, 2], literature_values[:, 0], literature_prediction[:, 0])
        ),
        delimiter=",",
        header="tau,literature_theta,pinn_theta",
        comments="",
    )
    metrics = {
        "paper": "Hsu, Tu & Chang, Axioms 12 (2023), 416",
        "case": cfg.case,
        "literature_column": "published_terms_20",
        "literature_points": int(len(literature_values)),
        "literature_vs_pinn_rmse": literature_rmse,
        "assimilation_vs_pinn_rmse": rmse(
            assimilation_values, assimilation_prediction
        ),
        "update_history": history,
    }
    (output_dir / "literature_pinn_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        assimilation_points[:, 2], assimilation_values[:, 0], "-", label="20-term reference"
    )
    axes[0].plot(
        assimilation_points[:, 2], assimilation_prediction[:, 0], "--", label="PINN"
    )
    axes[0].scatter(
        literature_points[:, 2], literature_values[:, 0], label="printed literature", zorder=3
    )
    axes[0].set(xlabel="tau", ylabel="theta(0.5,0.5,tau)", title=cfg.case)
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        [row["tau_end"] for row in history],
        [row["before_rmse"] for row in history],
        "o-",
        label="before update",
    )
    axes[1].plot(
        [row["tau_end"] for row in history],
        [row["after_rmse"] for row in history],
        "o-",
        label="after update",
    )
    axes[1].set(xlabel="tau", ylabel="batch RMSE", yscale="log", title="Adaptive updates")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_reference_comparison.png", dpi=180)
    plt.close(fig)

    # Figure analogous to the streaming-sensor map. The background is the
    # adaptive PINN slice at Y=0.5; reference observations exist only at X=0.5.
    x_grid = np.linspace(0.0, 1.0, 161)
    tau_grid = np.linspace(0.0, 1.2, 161)
    xx, tt = np.meshgrid(x_grid, tau_grid)
    slice_points = np.column_stack(
        (xx.ravel(), np.full(xx.size, 0.5), tt.ravel())
    )
    theta_slice = model.predict(slice_points).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    image = ax.contourf(xx, tt, theta_slice, levels=40, cmap="magma")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Adaptive PINN theta at Y=0.5")
    batches = np.array_split(
        np.arange(len(assimilation_points)),
        int(np.ceil(len(assimilation_points) / cfg.points_per_update)),
    )
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(batches)))
    for index, batch in enumerate(batches, start=1):
        ax.scatter(
            assimilation_points[batch, 0],
            assimilation_points[batch, 2],
            s=28,
            color=colors[index - 1],
            edgecolor="white",
            linewidth=0.35,
            label=f"window {index}",
            zorder=3,
        )
    ax.set(
        xlabel="Nondimensional coordinate X",
        ylabel="Nondimensional time tau",
        title="Streaming center-sensor windows over the adaptive PINN slice",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.2),
    )
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "streaming_sensor_windows.png", dpi=180)
    plt.close(fig)

    # The paper supplies reference truth only at its center point. Plot honest
    # center-point error histories instead of implying a full error field.
    baseline_error = np.abs(baseline_prediction[:, 0] - assimilation_values[:, 0])
    adaptive_error = np.abs(assimilation_prediction[:, 0] - assimilation_values[:, 0])
    improvement = baseline_error - adaptive_error
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    panels = (
        (baseline_error, "Offline PINN absolute error", "tab:red"),
        (adaptive_error, "Adaptive PINN absolute error", "tab:blue"),
        (improvement, "|baseline error| - |adaptive error|", "tab:green"),
    )
    for ax, (values, title, color) in zip(axes, panels):
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.plot(assimilation_points[:, 2], values, "o-", markersize=3, color=color)
        ax.set(xlabel="tau", title=title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Absolute theta error")
    axes[2].set_ylabel("Positive means adaptation helped")
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_adaptive_error_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Literature vs PINN RMSE ({cfg.case}): {literature_rmse:.8g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR,
        help="Directory containing the literature-reference CSV files.",
    )
    parser.add_argument("--case", choices=tuple(PAPER_CASES), default="table_1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Output directory (default: results folder beside this script).",
    )
    args = parser.parse_args()

    cfg = Config(case=args.case)
    data = load_paper_data(args.reference_dir, cfg.case)
    model, history, baseline_prediction = train_adaptively(cfg, data[0], data[1])
    save_results(
        model,
        cfg,
        history,
        data[0],
        data[1],
        baseline_prediction,
        data[2],
        data[3],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
