# Adaptive PINN Thermal Digital Twin

This project compares a standard offline physics-informed neural network (PINN)
with an adaptive PINN for a simplified rocket-engine throat-wall thermal
monitoring problem.

The physical model is 1D transient heat conduction through the wall thickness:

```text
rho * cp * dT/dt = k * d2T/dx2
```

Boundary conditions:

```text
x = 0:      -k dT/dx = q_hot(t)
x = Lwall: -k dT/dx = h_cool * (T - T_cool)
T(x, 0) = T0
```

The rocket-engine relevance comes from the hot-gas wall heat flux and
regenerative coolant-side convection. The implementation intentionally avoids
combustion modeling, CFD, and full nozzle geometry.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you do not want an editable install, run commands with:

```bash
PYTHONPATH=src python scripts/run_experiment.py --mode smoke
```

## Run

Quick verification:

```bash
PYTHONPATH=src python scripts/run_experiment.py --mode smoke --output-dir outputs/smoke
```

Longer report-quality run:

```bash
PYTHONPATH=src python scripts/run_experiment.py --mode full --output-dir outputs/full
```

Preliminary progress-report study with three random seeds:

```bash
PYTHONPATH=src python scripts/run_progress_study.py --output-dir outputs/progress
```

This writes `outputs/progress/progress_metric_summary.csv` and a draft report
at `reports/preliminary_progress_report.md`.

Additional visual-explainer graphs can be regenerated from saved model outputs:

```bash
PYTHONPATH=src python scripts/make_visual_graphs.py --mode smoke --output-dir outputs/smoke
```

On Apple Silicon or a CUDA machine you can try:

```bash
PYTHONPATH=src python scripts/run_experiment.py --mode smoke --device mps
PYTHONPATH=src python scripts/run_experiment.py --mode smoke --device cuda
```

## Outputs

The experiment writes:

- `outputs/.../results/config.json`
- `outputs/.../results/metrics.json`
- `outputs/.../results/baseline_history.csv`
- `outputs/.../results/adaptive_history.csv`
- `outputs/.../results/noise_summary.csv`
- `outputs/.../figures/reference_temperature_field.png`
- `outputs/.../figures/baseline_profiles.png`
- `outputs/.../figures/adaptive_profiles.png`
- `outputs/.../figures/relative_l2_error_over_time.png`
- `outputs/.../figures/loss_curves.png`
- `outputs/.../figures/sensor_noise_comparison.png`

## Method Summary

The finite-difference reference solution uses a Crank-Nicolson discretization.
The PINN is trained in nondimensional variables:

```text
x_hat = x / L_wall
t_hat = t / t_final
theta = (T - T_cool) / delta_T
```

The loss combines:

```text
PDE residual + hot boundary + coolant boundary + initial condition
```

The adaptive model starts from the offline baseline weights. Sparse synthetic
sensors are revealed in streaming time windows, and the model is periodically
fine-tuned with an added sensor-data loss.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The tests check the finite-difference solver, PINN residual, sensor sampler,
and metric evaluation.
