"""Stage 1-2: validate the general series solution at the paper midpoint."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analytical_solution import temperature
from parameters import PAPER_BOUNDARY_SETS, WorkflowConfig


TABLE_TO_SET = {"table_1": "set_1", "table_2": "set_2", "table_3": "set_3"}


def _read_table(path: Path) -> dict[str, np.ndarray]:
    columns = ["tau", "published_terms_20", "calculated_terms_20"]
    values = {name: [] for name in columns}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        for row in reader:
            for name in columns:
                values[name].append(float(row[name]))
    return {name: np.asarray(data) for name, data in values.items()}


def validate_against_literature(cfg: WorkflowConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)

    for index, (table, set_name) in enumerate(TABLE_TO_SET.items()):
        data = _read_table(cfg.paper_tables_dir / f"{table}_study_times.csv")
        tau = data["tau"]
        points = np.column_stack((np.full(len(tau), 0.5), np.full(len(tau), 0.5), tau))
        general = temperature(points, PAPER_BOUNDARY_SETS[set_name], cfg.series_terms)[:, 0]
        published = data["published_terms_20"]
        equation_csv = data["calculated_terms_20"]
        published_rmse = float(np.sqrt(np.mean((general - published) ** 2)))
        equation_rmse = float(np.sqrt(np.mean((general - equation_csv) ** 2)))
        metrics[table] = {
            "boundary_set": set_name,
            "published_rmse": published_rmse,
            "paper_equation_csv_rmse": equation_rmse,
            "max_abs_published_error": float(np.max(np.abs(general - published))),
        }
        for t, pub, eq, value in zip(tau, published, equation_csv, general):
            rows.append((table, t, pub, eq, value, value - pub))

        ax = axes[index]
        ax.plot(tau, general, "-", label="general analytical solution")
        ax.scatter(tau, published, marker="o", label="published table", zorder=3)
        case_number = table.removeprefix("table_")
        decays = PAPER_BOUNDARY_SETS[set_name].decays
        ax.set(
            title=f"Table {case_number}: edge decays {decays}\nRMSE vs printed values={published_rmse:.3g}",
            xlabel=r"Dimensionless time, $\tau$",
        )
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Dimensionless center temperature")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Validation against the paper: Table 1 agrees; printed Tables 2–3 conflict with their stated equations"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "01_literature_validation.png", dpi=180)
    plt.close(fig)

    with (output_dir / "01_literature_validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "table",
                "tau",
                "published_theta",
                "paper_equation_csv_theta",
                "general_analytical_theta",
                "general_minus_published",
            ]
        )
        writer.writerows(rows)
    (output_dir / "01_literature_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
