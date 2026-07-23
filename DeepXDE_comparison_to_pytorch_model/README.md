# DeepXDE comparison to the PyTorch model

This is an independent copy of `DeepXDE/` configured with the full PyTorch
PINN's comparable training hyperparameters. The original folder is unchanged.

## Matched hyperparameters

| Setting | Comparison value |
|---|---:|
| Hidden layers | 3 |
| Units per hidden layer | 48 |
| Activation | tanh |
| Baseline Adam iterations | 1,200 |
| Adaptive windows | 5 |
| Adam iterations per adaptive window | 160 |
| Baseline learning rate | 0.002 |
| Adaptive learning rate | 0.0005 |
| Interior PDE points | 1,536 |
| Boundary points | 256 |
| Initial points | 256 |
| Sensor/data loss weight | 10 |

Additional training mechanics are also aligned:

- Glorot/Xavier uniform initialization;
- pseudorandom collocation sampling;
- fresh interior/initial physics points every Adam iteration (boundary points
  remain fixed because DeepXDE requires stable per-edge sample counts);
- physics loss weights of 1 for the PDE, 2 for each boundary, and 5 for the
  initial condition;
- four sensors on the 2-D centerline in the validated workflow.

The DeepXDE models retain three inputs `(X, Y, tau)` because they solve the
2-D paper problem. Consequently they have 4,945 trainable parameters, versus
4,897 in the two-input PyTorch model. DeepXDE also retains its analytical 2-D
sensor datasets; the PyTorch model's four 1-D sensor positions and synthetic
noise sweeps are not substituted here because doing so would change the
problem and invalidate the paper-data comparison. The compact paper workflow
must also retain its published center-point observations. The validated
workflow instead places four sensors at `x=(0.05, 0.25, 0.5, 0.75), y=0.5`;
the first sensor is slightly inside the domain because the DeepXDE `x=0` edge
has a prescribed temperature, unlike the PyTorch model's flux boundary.

This folder owns the comparison DeepXDE code, tests, and copied results in the
repository. The independent analytical literature reproduction and its
reference datasets live in `../Literature_results/`.

## Workflows

- `paper_data_model/` is the compact DeepXDE model that assimilates the
  literature paper's center-point reference values in sequential batches.
- `validated_adaptive_workflow/` is the larger full-field workflow containing
  analytical validation, offline PINN training, streaming adaptation,
  boundary-change experiments, and latency studies.

Each workflow has its own README, requirements, and results directory. Keep
new PINN outputs beside the workflow that generated them.
