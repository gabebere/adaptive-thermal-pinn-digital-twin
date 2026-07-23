# Validated analytical and adaptive PINN workflow

This is the repository's primary full-field DeepXDE workflow. Independent
paper tables and expanded analytical reference artifacts are organized under
`../../Literature_results/`; PINN outputs remain in this workflow's `outputs/`
directory.

This folder implements the workflow in five separate scientific stages:

1. Evaluate the general analytical series solution for all four space-time
   Dirichlet boundaries of the paper's 2D problem.
2. Validate that solution at `(X,Y)=(0.5,0.5)` against Tables 1-3.
3. Sample the validated solution over `(X,Y,tau)` up to configurable
   `tau_final` (100 by default) and train a physics-only DeepXDE PINN.
4. Compare the offline PINN against the complete analytical reference field.
5. Reveal sparse analytical sensor values in batches of `n` time instances,
   continue training the same network with an added data loss, and compare the
   adaptive PINN against the original reference.

A second experiment uses the same reproduced analytical series for a
boundary-change problem. Boundary set 1 is evaluated until half the run, then
boundary set 2 begins with a reset event clock. A homogeneous sine-series
correction preserves the pre-event interior temperature while the edge values
take the new Dirichlet conditions. The PINN is not told about the change by
default; it must respond through the streamed analytical observations.

## Notation and literature cases

`theta` (written mathematically as $\theta$) is the paper's dimensionless
temperature, scaled using its reference temperature. `tau` ($\tau$) is
dimensionless time, and `X` and `Y` are dimensionless positions on the unit
square. Values of `theta` are therefore not temperatures in degrees Celsius.

The analytical validation checks all three cases printed in the source paper.
They differ only in the exponential decay rates imposed on the four edges:

- Table 1: `(1, 1, 1, 1)`;
- Table 2: `(1, 1, 2, 2)`;
- Table 3: `(1, 2, 3, 4)`.

Table 1 is the verified benchmark. Direct evaluation of the paper's equations
does not reproduce the values printed in Tables 2 and 3, although it does
reproduce the equation-derived CSV values. The primary long-horizon workflow
therefore uses the verified Table 1 boundary case. Tables 2 and 3 are retained
to document the publication-level discrepancy, not as trusted training data.

Plots use linear physical time. The PINN still receives an internal
log-transformed time coordinate to resolve the rapid early transient; that
training transform is not used as a display axis.

## Parameters

Edit `parameters.py`. It contains:

- all four boundary decay/amplitude functions;
- the shared `boundary_spatial_profile` and `initial_condition` functions;
- final time and number of time instances;
- sensor locations;
- `batch_size_n`;
- `observation_window_batches` (`None` keeps all history; an integer uses a
  rolling recent-history window);
- network layers, learning rates, loss weight, and iterations;
- boundary-switch time, boundary set, and solver resolution.

## Run

Quick end-to-end verification:

```powershell
python run_workflow.py --profile smoke
```

Report-quality run to `tau=100`:

```powershell
python run_workflow.py --profile full
```

Both commands use the maintained `balanced` online configuration by default:
9 boundary-aware sensors, two time instances per update, two recent batches of
observation history, data-loss weight 10, and 100 adaptive iterations per
update.

For the more expensive latency-critical configuration:

```powershell
python run_workflow.py --profile full --adaptive-profile low_latency
```

`low_latency` samples twice as often, updates after every time instance, uses
25 sensors, retains four recent instances, and uses data-loss weight 20. It is
maintained as an opt-in profile rather than the default because it requires
substantially more online optimization.

Run the boundary-change latency parameter study:

```powershell
python latency_study.py --profile full
```

Generate the two paper-style figures for the combined low-latency case:

```powershell
python combined_latency_figures.py --profile full
```

The study cases are the editable `LATENCY_EXPERIMENTS` rows in
`parameters.py`. They compare sampling interval, update batch size, sensor
placement, recent-history length, and data-loss weight while keeping the same
physical boundary event. The combined low-latency case uses a 5-by-5 grid (25
sensors) at normalized coordinates `0.05, 0.275, 0.5, 0.725, 0.95` in both
directions.

## Online update semantics

For each group of `n` new time instances, the current PINN predicts the new
sensor values before seeing them. Their RMSE is recorded. All analytical sensor
observations revealed so far are then added as a DeepXDE `PointSetBC` data-loss
term. The same neural network weights are updated with Adam; the network is not
reinitialized. The physical PDE coefficients remain fixed.

Two full-field error sequences are saved. `causal_prior_time_rmse` is the true
online prediction error before each new batch is assimilated.
`causal_posterior_time_rmse` is the fit after assimilating that batch. These are
not produced by evaluating the final network retrospectively over the past.

## Outputs

The profile-specific output directory contains:

- `01_literature_validation.png` and CSV/JSON evidence;
- `02_reference_field.png` and the analytical reference NPZ;
- `03_offline_pinn_validation.png`;
- `04_streaming_numerical_inputs.png`;
- `05_adaptive_pinn_error.png`;
- `06_boundary_change_response.png` and switch-reference NPZ;
- `latency_study/07_latency_parameter_sweep.png`, CSV, and JSON;
- `combined_latency/08_combined_streaming_sensors.png`;
- `combined_latency/09_combined_error_fields.png` and their source-data NPZ;
- `workflow_metrics.json`.
