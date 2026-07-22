# Reproducibility snapshot — 2026-07-22

This folder preserves the report-quality results produced during the
constant-flux adaptive PINN / streamed PINO study. Smoke-run outputs and caches
are intentionally excluded.

## Source and trained models

- Maintained implementation: `../../core/`
- Architecture configuration: `../../architectures/balanced.toml`
- Reusable trained weights and SHA-256 manifest: `../../checkpoints/`
- Source revision before this results snapshot: `19483e8`

The offline PINN and streamed PINO are loaded directly from their checkpoints;
they are not retrained for the new-boundary tests. The adaptive PINN starts from
the offline checkpoint and performs its defining sensor-driven online update for
each case.

## Graphs

The `graphs/` folder contains 22 combined full-resolution figures plus 34
standalone panels:

- `01_original_constant_flux/`: the first constant-flux analytical, offline,
  adaptive, and streamed-PINO comparison;
- `02_improved_mixed_flux/`: the improved PINO, stratified training coverage,
  paper/physical analytical checks, error fields, and before/after comparison;
- `03_new_cases/no_break/`: five temperature/RMSE/error-field plots for the unseen constant
  4.596 MW/m² case;
- `03_new_cases/random_break/`: five temperature/RMSE/error-field plots for the unseen
  4.302→5.189 MW/m² case with a random break at tau=53.

For both new cases, Graph 01 shows analytical, offline-PINN, and adaptive-PINN
space-time temperature fields on one shared scale. Graphs 02 and 04 place a
linear overview beside a logarithmic detail view in the same figure. Graph 03
uses four model panels and a shared tau color gradient to show the full
temperature evolution. Graph 05 reports absolute error fields on a shared
logarithmic scale and pairwise error reductions on a shared diverging scale.
Each new-case directory also contains `individual_figures/`, where every panel
from Graphs 01–05 is exported as its own figure. The `01a`–`05f` filename
prefixes preserve the relationship to the corresponding combined figure while
allowing individual selection for papers and presentations.

## Headline RMSE

| Case | Offline PINN | Adaptive PINN | Streamed PINO |
|---|---:|---:|---:|
| Locked 4.0→5.2 MW/m² at t=0.5 | 18.655 K | 2.038 K | **0.246 K** |
| Unseen no-break 4.596 MW/m² | 22.208 K | 2.286 K | **0.226 K** |
| Unseen 4.302→5.189 MW/m² at tau=53 | 21.960 K | 2.180 K | **0.226 K** |

Exact values and point-by-point arrays are under `metrics/`.

## Held-out test data

`test_data/mixed_flux_v2_test_data.zip` contains the 200 held-out interpolation
scenarios plus the locked non-decaying scenario. The filtered manifest,
generation configuration, and checksums describe only the packaged test files.

- Archive SHA-256:
  `a202581d5da7d97ba5e44a50be04e2f349800c37fed8487c4b9ad21ae00e605f`
- Training and validation CSVs remain generated/ignored because they are not
  required to reproduce the reported evaluations.

## Boundary conditions and analytical reference

The versioned physical definition is in `../../core/constant_flux_physics.py`:

```text
T_t = alpha T_xx
-k T_x(0,t) = q_hot(t)
-k T_x(L,t) = h [T(L,t) - T_coolant]
T(x,0) = T_coolant
```

The exact reference is the Robin eigenfunction expansion with causal
step-response superposition. Mixed PINO training includes stratified switch
times/magnitudes, balanced upward/downward changes, 70% non-decaying steps, and
30% exponentially relaxing cases.
